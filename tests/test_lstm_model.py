"""
tests/test_lstm_model.py

Unit tests for LSTM model implementation.
All tests from planning document T-01 through T-10.
"""

import numpy as np
import pytest
import torch
import tempfile
import shutil
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_model import LSTMModel, LSTMPricePredictor, EarlyStopping


class TestLSTMModel:
    """Test suite for LSTM Model - covering T-01 through T-10"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic price data for testing."""
        np.random.seed(42)
        # Generate 300 days of synthetic gold-like prices
        returns = np.random.normal(0.0001, 0.01, 300)
        prices = 1800 * np.exp(np.cumsum(returns))
        return prices
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_t01_instantiation(self, synthetic_data, temp_cache_dir):
        """T-01: Model instantiation with synthetic data - No exceptions raised."""
        try:
            model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
            assert model is not None
            assert model.historical_data is not None
            assert len(model.historical_data) == 300
        except Exception as e:
            pytest.fail(f"Instantiation failed: {e}")
    
    def test_t01_instantiation_fails_with_short_data(self, temp_cache_dir):
        """T-01b: Model instantiation fails with < 200 data points."""
        short_data = np.random.randn(100)
        with pytest.raises(ValueError, match="at least 200 data points"):
            LSTMModel(short_data, seed=42, cache_dir=temp_cache_dir)
    
    def test_t02_calibrate_returns_params(self, synthetic_data, temp_cache_dir):
        """T-02: calibrate() returns a params dict with required keys."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        params = model.calibrate(verbose=False)
        
        assert isinstance(params, dict)
        assert 'train_loss' in params
        assert 'val_loss' in params
        assert 'epochs' in params
        assert 'input_size' in params
        assert 'window' in params
        assert 'hidden1' in params
        assert 'hidden2' in params
    
    def test_t03_simulate_output_shape(self, synthetic_data, temp_cache_dir):
        """T-03: simulate() output shape (n_paths, n_steps+1)."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        model.calibrate(verbose=False)
        
        S0 = 1800.0
        n_steps = 30
        n_paths = 100
        
        paths = model.simulate(S0, n_steps, n_paths)
        
        assert isinstance(paths, np.ndarray)
        assert paths.shape == (n_paths, n_steps + 1)
        assert paths.dtype == np.float64 or paths.dtype == np.float32
    
    def test_t04_first_column_equals_s0(self, synthetic_data, temp_cache_dir):
        """T-04: simulate() first column equals S0 (±1e-6)."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        model.calibrate(verbose=False)
        
        S0 = 1800.0
        paths = model.simulate(S0, n_steps=30, n_paths=50)
        
        assert np.allclose(paths[:, 0], S0, atol=1e-6)
    
    def test_t05_paths_non_negative(self, synthetic_data, temp_cache_dir):
        """T-05: Paths are non-negative for positive S0."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        model.calibrate(verbose=False)
        
        S0 = 1800.0
        paths = model.simulate(S0, n_steps=30, n_paths=50)
        
        assert np.all(paths > 0), f"Min price: {np.min(paths)}"
    
    def test_t06_different_seeds_produce_different_paths(self, synthetic_data, temp_cache_dir):
        """T-06: Different seeds produce different paths."""
        model1 = LSTMModel(synthetic_data, seed=1, cache_dir=temp_cache_dir + "_1")
        model1.calibrate(verbose=False)
        paths1 = model1.simulate(1800.0, n_steps=30, n_paths=10)
        
        model2 = LSTMModel(synthetic_data, seed=2, cache_dir=temp_cache_dir + "_2")
        model2.calibrate(verbose=False)
        paths2 = model2.simulate(1800.0, n_steps=30, n_paths=10)
        
        assert not np.allclose(paths1, paths2), "Different seeds should produce different paths"
    
    def test_t07_same_seed_produces_identical_paths(self, synthetic_data, temp_cache_dir):
        """T-07: Same seed produces identical paths."""
        cache1 = os.path.join(temp_cache_dir, "seed42_1")
        cache2 = os.path.join(temp_cache_dir, "seed42_2")
        
        model1 = LSTMModel(synthetic_data, seed=42, cache_dir=cache1)
        model1.calibrate(verbose=False)
        paths1 = model1.simulate(1800.0, n_steps=30, n_paths=10)
        
        model2 = LSTMModel(synthetic_data, seed=42, cache_dir=cache2)
        model2.calibrate(verbose=False)
        paths2 = model2.simulate(1800.0, n_steps=30, n_paths=10)
        
        assert np.allclose(paths1, paths2), "Same seed should produce identical paths"
    
    def test_t08_model_weights_save_load(self, synthetic_data, temp_cache_dir):
        """T-08: Model weights save/load round-trip produces identical output."""
        # First model - train and save
        model1 = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        params1 = model1.calibrate(verbose=False)
        paths1 = model1.simulate(1800.0, n_steps=30, n_paths=10)
        
        # Second model - load from cache
        model2 = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        params2 = model2.calibrate(verbose=False)  # Should load from cache
        paths2 = model2.simulate(1800.0, n_steps=30, n_paths=10)
        
        assert np.allclose(paths1, paths2), "Cached model should produce identical output"
        assert params1['val_loss'] == params2['val_loss']
    
    def test_t09_simulate_single_path(self, synthetic_data, temp_cache_dir):
        """T-09: simulate() works with n_paths=1 - No shape errors."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        model.calibrate(verbose=False)
        
        paths = model.simulate(1800.0, n_steps=30, n_paths=1)
        
        assert paths.shape == (1, 31)
        assert paths[0, 0] == 1800.0
    
    def test_t10_factory_create_model(self, synthetic_data, temp_cache_dir):
        """T-10: Factory create_model('lstm', ...) returns LSTMModel instance."""
        try:
            from src.models import create_model
            model = create_model('lstm', historical_data=synthetic_data, seed=42, cache_dir=temp_cache_dir)
            assert isinstance(model, LSTMModel)
        except ImportError:
            pytest.skip("Factory not available in test environment")
    
    def test_get_parameters_uncalibrated(self, synthetic_data, temp_cache_dir):
        """Test get_parameters() before calibration."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        params = model.get_parameters()
        
        assert params['status'] == 'uncalibrated'
        assert 'message' in params
    
    def test_get_parameters_trained(self, synthetic_data, temp_cache_dir):
        """Test get_parameters() after calibration."""
        model = LSTMModel(synthetic_data, seed=42, cache_dir=temp_cache_dir)
        model.calibrate(verbose=False)
        params = model.get_parameters()
        
        assert params['status'] == 'trained'
        assert 'architecture' in params
        assert 'training' in params
        assert 'simulation' in params
        assert params['model_type'] == 'LSTM'
        assert 'total_parameters' in params['architecture']


class TestLSTMPricePredictor:
    """Tests for the internal PyTorch module."""
    
    def test_forward_pass_shape(self):
        """Test that LSTMPricePredictor forward pass produces correct shape."""
        batch_size = 16
        seq_len = 20
        input_size = 10
        
        model = LSTMPricePredictor(input_size, hidden1=128, hidden2=64)
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = model(x)
        
        assert output.shape == (batch_size,)
    
    def test_forward_pass_values(self):
        """Test that output is reasonable (finite values)."""
        model = LSTMPricePredictor(10, hidden1=32, hidden2=16)
        x = torch.randn(8, 20, 10)
        
        output = model(x)
        
        assert torch.all(torch.isfinite(output))
        assert output.dtype == torch.float32


class TestEarlyStopping:
    """Tests for EarlyStopping utility."""
    
    def test_early_stopping_trigger(self):
        """Test that early stopping triggers after patience exceeded."""
        es = EarlyStopping(patience=3, delta=0.01)
        model = LSTMPricePredictor(10, hidden1=32, hidden2=16)
        
        # Simulate increasing validation loss
        losses = [0.1, 0.11, 0.12, 0.13, 0.14]
        triggered = False
        
        for loss in losses:
            if es(loss, model):
                triggered = True
                break
        
        assert triggered, "Early stopping should trigger"
    
    def test_early_stopping_save_best(self):
        """Test that best model is saved."""
        es = EarlyStopping(patience=5, delta=0.01)
        model = LSTMPricePredictor(10, hidden1=32, hidden2=16)
        
        losses = [0.1, 0.09, 0.08, 0.085, 0.09]
        
        for loss in losses:
            es(loss, model)
        
        assert es.val_loss_min == 0.08


if __name__ == '__main__':
    pytest.main([__file__, '-v'])