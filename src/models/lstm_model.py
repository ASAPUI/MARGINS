"""
src/models/lstm_model.py

LSTM Neural Network model for gold price forecasting.
Implements the same interface as GBM, OU, Merton, Heston, and Regime models.

Author: Essabri Ali Rayan (@ASAPUI)
Version: 1.3
Date: March 2026
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Dict, Any
import hashlib
import json
import os
import random
from pathlib import Path
import warnings

# Import sequence_builder from features module
try:
    from src.data.features import sequence_builder
except ImportError:
    # Fallback if features module not available
    def sequence_builder(feature_matrix: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Build sequences for LSTM training."""
        N, F = feature_matrix.shape
        if N <= window:
            raise ValueError(f"Need more than {window} timesteps, got {N}")
        
        X, y = [], []
        for i in range(window, N):
            X.append(feature_matrix[i-window:i])
            y.append(feature_matrix[i, 0])
        
        return np.array(X), np.array(y)


class LSTMPricePredictor(nn.Module):
    """
    Internal PyTorch module - not exposed to the model registry.
    Two-layer stacked LSTM with dense projection head.
    
    Architecture:
        Input: (batch, seq_len, input_size)
        LSTM1: (input_size → hidden1) with dropout
        LSTM2: (hidden1 → hidden2) 
        FC1: (hidden2 → 32) + ReLU
        FC2: (32 → 1) - predicts log-return
    """
    
    def __init__(self, input_size: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # First LSTM layer - returns full sequence
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Second LSTM layer - returns last hidden state only
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(32, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize FC layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size,) - predicted log-returns
        """
        # LSTM layers
        out, _ = self.lstm1(x)  # (batch, seq_len, hidden1)
        out, _ = self.lstm2(out)  # (batch, seq_len, hidden2)
        
        # Take last timestep
        out = out[:, -1, :]  # (batch, hidden2)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze(-1)  # (batch,)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 15, delta: float = 1e-5, verbose: bool = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...")
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.val_loss_min = val_loss
    
    def load_best(self, model: nn.Module):
        """Load the best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class LSTMModel:
    """
    LSTM-based gold price model. Registered as 'lstm' in the model factory.
    Compatible with the Monte Carlo simulation interface.
    
    This model uses a two-layer LSTM architecture to predict log-returns,
    then uses residual bootstrapping to generate stochastic price paths.
    
    Attributes:
        MODEL_KEY: Class identifier for factory registration
        historical_data: Training price series
        seed: Random seed for reproducibility
        scaler: MinMaxScaler for feature normalization
        network: Trained LSTMPricePredictor instance
        residuals: In-sample prediction residuals for bootstrapping
        training_params: Dictionary of training metrics
    """
    
    MODEL_KEY = 'lstm'
    
    def __init__(self, historical_data: np.ndarray, seed: int = 42, cache_dir: Optional[str] = None):
        """
        Initialize LSTM model.
        
        Args:
            historical_data: Array of historical prices (1D or 2D with shape [n, 1])
            seed: Random seed for reproducibility
            cache_dir: Directory for caching trained model weights
            
        Raises:
            ValueError: If historical_data has fewer than 200 points
        """
        # Ensure 1D array
        self.historical_data = np.array(historical_data).flatten()
        
        if len(self.historical_data) < 200:
            raise ValueError(f"LSTMModel requires at least 200 data points for feature engineering, got {len(self.historical_data)}")
        
        self.seed = seed
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.network: Optional[LSTMPricePredictor] = None
        self.residuals: Optional[np.ndarray] = None
        self.training_params: Dict[str, Any] = {}
        self._feature_matrix: Optional[np.ndarray] = None
        self._window_size = 20
        
        # Setup cache directory
        if cache_dir is None:
            self._cache_dir = Path.home() / '.gold_option_cache' / 'lstm'
        else:
            self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        self._set_seed(seed)
        
        # Suppress PyTorch warnings
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def _set_seed(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        
        # For PyTorch >= 2.0
        if hasattr(torch, "generator"):
            g = torch.Generator()
            g.manual_seed(seed)
    
    def _build_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Build feature matrix from price series.
        
        Features (10 total):
            1. Log returns
            2. 10-day rolling volatility
            3. 30-day rolling volatility  
            4. RSI (14-day)
            5. Price / SMA 50
            6. Price / SMA 200
            7. Day of week (sin)
            8. Day of week (cos)
            9. Month (sin)
            10. Month (cos)
            
        Args:
            prices: Array of prices
            
        Returns:
            Feature matrix of shape (len(prices), 10)
        """
        n = len(prices)
        
        # 1. Log returns (primary target feature)
        log_returns = np.zeros(n)
        log_returns[1:] = np.diff(np.log(prices))
        
        # Helper for rolling statistics
        def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
            result = np.convolve(x, np.ones(window)/window, mode='same')
            result[:window-1] = np.nan
            return result
        
        def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
            result = np.array([np.std(x[max(0, i-window+1):i+1]) for i in range(len(x))])
            return result
        
        # 2. & 3. Rolling volatilities
        vol_10d = rolling_std(log_returns, 10)
        vol_30d = rolling_std(log_returns, 30)
        
        # 4. RSI (14-day)
        delta = np.zeros(n)
        delta[1:] = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = rolling_mean(gain, 14)
        avg_loss = rolling_mean(loss, 14)
        
        # Handle division by zero
        rs = np.zeros(n)
        mask = avg_loss > 0
        rs[mask] = avg_gain[mask] / avg_loss[mask]
        rsi = 100 - (100 / (1 + rs))
        
        # 5. & 6. Price to SMA ratios
        sma_50 = rolling_mean(prices, 50)
        sma_200 = rolling_mean(prices, 200)
        
        price_sma_50 = prices / (sma_50 + 1e-10)
        price_sma_200 = prices / (sma_200 + 1e-10)
        
        # 7. - 10. Cyclical time features (assuming daily trading data)
        # Day of week (0-4 for trading days)
        days_idx = np.arange(n)
        day_of_week = days_idx % 5
        month = (days_idx // 21) % 12  # Approximate trading months
        
        dow_sin = np.sin(2 * np.pi * day_of_week / 5)
        dow_cos = np.cos(2 * np.pi * day_of_week / 5)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Stack all features
        features = np.column_stack([
            log_returns,      # 0: Target feature
            vol_10d,          # 1
            vol_30d,          # 2
            rsi,              # 3
            price_sma_50,     # 4
            price_sma_200,    # 5
            dow_sin,          # 6
            dow_cos,          # 7
            month_sin,        # 8
            month_cos         # 9
        ])
        
        # Handle NaN/Inf - forward fill then backward fill
        for i in range(features.shape[1]):
            col = features[:, i]
            # Forward fill
            mask = np.isnan(col) | np.isinf(col)
            if mask.any():
                idx = np.where(~mask)[0]
                if len(idx) > 0:
                    col[mask] = np.interp(np.where(mask)[0], idx, col[idx])
                # Backward fill any remaining
                mask = np.isnan(col) | np.isinf(col)
                if mask.any():
                    col[mask] = col[~mask][0] if (~mask).any() else 0.0
            features[:, i] = col
        
        return features
    
    def _get_cache_key(self) -> str:
        """Generate cache key based on data hash and hyperparameters."""
        data_str = self.historical_data.tobytes()
        config_str = f"{self.seed}_{self._window_size}_v2.0"
        combined = data_str + config_str.encode()
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def _get_cache_path(self) -> Path:
        """Get path for cached model weights."""
        cache_key = self._get_cache_key()
        return self._cache_dir / f"{cache_key}.pt"
    
    def _get_params_cache_path(self) -> Path:
        """Get path for cached training params."""
        cache_key = self._get_cache_key()
        return self._cache_dir / f"{cache_key}_params.json"
    
    def _save_to_cache(self):
        """Save model state to cache."""
        if self.network is None:
            return
        
        cache_path = self._get_cache_path()
        params_path = self._get_params_cache_path()
        
        try:
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'scaler': self.scaler,
                'input_size': self.training_params.get('input_size', 10),
                'window': self._window_size,
                'residuals': self.residuals,
                'training_params': self.training_params,
                'historical_data_sample': self.historical_data[:10]  # For verification
            }, cache_path)
            
            with open(params_path, 'w') as f:
                json.dump(self.training_params, f, indent=2, default=str)
                
        except Exception as e:
            warnings.warn(f"Failed to cache model: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load model state from cache if available."""
        cache_path = self._get_cache_path()
        params_path = self._get_params_cache_path()
        
        if not cache_path.exists() or not params_path.exists():
            return False
        
        try:
            checkpoint = torch.load(cache_path, map_location='cpu', weights_only=False)
            
            # Verify data compatibility (check first 10 samples)
            cached_sample = checkpoint.get('historical_data_sample', [])
            current_sample = self.historical_data[:10]
            if not np.allclose(cached_sample, current_sample, rtol=1e-5):
                return False
            
            # Restore state
            self.scaler = checkpoint['scaler']
            self._window_size = checkpoint['window']
            self.residuals = checkpoint['residuals']
            self.training_params = checkpoint['training_params']
            
            input_size = checkpoint['input_size']
            self.network = LSTMPricePredictor(
                input_size=input_size,
                hidden1=self.training_params.get('hidden1', 128),
                hidden2=self.training_params.get('hidden2', 64),
                dropout=self.training_params.get('dropout', 0.2)
            )
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.network.eval()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load cache: {e}")
            return False
    
    def calibrate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the LSTM network on historical data.
        
        Training Configuration:
            - Optimizer: Adam (lr=1e-3)
            - Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
            - Loss: MSE on log-returns
            - Early Stopping: patience=15, delta=1e-5
            - Max Epochs: 100
            - Batch Size: 32
            - Train/Val Split: 80/20 (time-aware, no shuffle)
            
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training metrics and parameters
        """
        # Check cache first
        if self._load_from_cache():
            if verbose:
                print(f"Loaded cached LSTM model (val_loss: {self.training_params.get('val_loss', 'N/A'):.6f})")
            return self.training_params
        
        if verbose:
            print("Training LSTM model...")
        
        # 1. Build features
        self._feature_matrix = self._build_features(self.historical_data)
        n_features = self._feature_matrix.shape[1]
        
        # 2. Build sequences
        X, y = sequence_builder(self._feature_matrix, window=self._window_size)
        n_samples = len(X)
        
        if n_samples < 50:
            raise ValueError(f"Too few samples after sequencing: {n_samples}. Need at least 50.")
        
        # 3. Time-aware train/val split (no shuffling)
        train_size = int(0.8 * n_samples)
        if train_size < 30:
            train_size = int(0.7 * n_samples)  # Adjust if dataset is small
        
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 4. Normalize features (fit only on training data)
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        
        self.scaler.fit(X_train_reshaped)
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val_reshaped).reshape(X_val.shape)
        
        # 5. Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 6. Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 7. Initialize network
        self.network = LSTMPricePredictor(
            input_size=n_features,
            hidden1=128,
            hidden2=64,
            dropout=0.2
        )
        
        # 8. Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        early_stopping = EarlyStopping(patience=15, delta=1e-5, verbose=verbose)
        
        # 9. Training loop
        max_epochs = 100
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(max_epochs):
            # Training phase
            self.network.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.network.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.network(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if early_stopping(avg_val_loss, self.network):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        early_stopping.load_best(self.network)
        self.network.eval()
        
        # 10. Compute residuals on full dataset for simulation
        with torch.no_grad():
            X_full_scaled = self.scaler.transform(
                X.reshape(-1, n_features)
            ).reshape(X.shape)
            X_full_tensor = torch.FloatTensor(X_full_scaled)
            predictions = self.network(X_full_tensor).numpy()
        
        self.residuals = y - predictions
        
        # Store training parameters
        self.training_params = {
            'train_loss': float(history['train_loss'][-1]),
            'val_loss': float(early_stopping.val_loss_min),
            'best_val_loss': float(early_stopping.val_loss_min),
            'epochs': len(history['train_loss']),
            'input_size': n_features,
            'window': self._window_size,
            'hidden1': 128,
            'hidden2': 64,
            'dropout': 0.2,
            'final_lr': optimizer.param_groups[0]['lr'],
            'n_train_samples': train_size,
            'n_val_samples': n_samples - train_size,
            'residual_std': float(np.std(self.residuals)),
            'residual_mean': float(np.mean(self.residuals))
        }
        
        # Cache the model
        self._save_to_cache()
        
        if verbose:
            print(f"Training complete. Best val loss: {early_stopping.val_loss_min:.6f}")
        
        return self.training_params
    
    def simulate(self, S0: float, n_steps: int, n_paths: int) -> np.ndarray:
       if self.network is None:
        raise RuntimeError("Call calibrate() first.")
       if S0 <= 0 or n_steps <= 0 or n_paths <= 0:
        raise ValueError("S0, n_steps, n_paths must be positive.")

       self.network.eval()
       window = self._window_size
       n_features = self.training_params['input_size']

    # Build initial feature context ONCE using enough history for SMA-200
       context = self.historical_data[-(window + 220):].copy()
    # Anchor the last price to S0 so all paths start correctly
       context[-1] = S0

       base_features = self._build_features(context)  # one call, not 150k
       init_window = base_features[-window:]           # shape (window, n_features)

       paths = np.zeros((n_paths, n_steps + 1))
       paths[:, 0] = S0

       with torch.no_grad():
         for path_idx in range(n_paths):
            cur_window = init_window.copy()
            cur_prices = context[-window:].copy()
            cur_price = S0

            for step in range(1, n_steps + 1):
                scaled = self.scaler.transform(cur_window)
                tensor = torch.FloatTensor(scaled).unsqueeze(0)
                pred_log_ret = self.network(tensor).item()

                residual = np.random.choice(self.residuals)
                log_ret = pred_log_ret + residual
                next_price = max(cur_price * np.exp(log_ret), 1e-6)

                paths[path_idx, step] = next_price

                # Incremental window update — no full feature rebuild
                cur_prices = np.append(cur_prices[1:], next_price)
                new_feats = self._build_features(cur_prices)
                cur_window = new_feats[-window:]
                cur_price = next_price

       return paths
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return model parameters and training summary.
        
        Returns:
            Dictionary containing:
                - model_type: "LSTM"
                - architecture: Network architecture details
                - training: Training metrics (losses, epochs, etc.)
                - simulation: Residual statistics
                - status: "trained" or "uncalibrated"
        """
        if not self.training_params or self.network is None:
            return {
                "model_type": "LSTM",
                "status": "uncalibrated",
                "message": "Model has not been trained yet. Call calibrate() first."
            }
        
        return {
            "model_type": "LSTM",
            "status": "trained",
            "architecture": {
                "input_size": self.training_params.get('input_size'),
                "window_size": self.training_params.get('window'),
                "hidden_layers": [
                    self.training_params.get('hidden1'),
                    self.training_params.get('hidden2')
                ],
                "dropout": self.training_params.get('dropout'),
                "total_parameters": sum(p.numel() for p in self.network.parameters())
            },
            "training": {
                "final_train_loss": self.training_params.get('train_loss'),
                "final_val_loss": self.training_params.get('val_loss'),
                "best_val_loss": self.training_params.get('best_val_loss'),
                "epochs_trained": self.training_params.get('epochs'),
                "final_learning_rate": self.training_params.get('final_lr'),
                "train_samples": self.training_params.get('n_train_samples'),
                "val_samples": self.training_params.get('n_val_samples')
            },
            "simulation": {
                "residual_mean": float(np.mean(self.residuals)) if self.residuals is not None else None,
                "residual_std": float(np.std(self.residuals)) if self.residuals is not None else None,
                "residual_skew": float(np.mean(((self.residuals - np.mean(self.residuals)) / (np.std(self.residuals) + 1e-10))**3)) if self.residuals is not None else None
            },
            "cache": {
                "cache_dir": str(self._cache_dir),
                "cache_key": self._get_cache_key()
            }
        }
    
    def predict_single(self, recent_prices: np.ndarray) -> float:
        """
        Predict next log-return for a single price window.
        
        Args:
            recent_prices: Array of at least 'window' recent prices
            
        Returns:
            Predicted log-return for next step
        """
        if self.network is None:
            raise RuntimeError("Model not calibrated")
        
        if len(recent_prices) < self._window_size:
            raise ValueError(f"Need at least {self._window_size} prices, got {len(recent_prices)}")
        
        window_prices = recent_prices[-self._window_size:]
        features = self._build_features(window_prices)
        
        self.network.eval()
        with torch.no_grad():
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            pred = self.network(features_tensor).item()
        
        return pred


# Factory registration helper
if __name__ == "__main__":
    # Simple test
    print("LSTMModel implementation loaded successfully")
    print(f"Model key: {LSTMModel.MODEL_KEY}")