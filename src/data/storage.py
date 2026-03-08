"""
Data Storage & Caching Module

Provides efficient local storage for:
- Raw market data caching (avoid repeated API calls)
- Processed feature storage
- Model parameter persistence
- Simulation results archiving

Supports multiple backends: Parquet (default), HDF5, CSV, Pickle

Author: Essabri Ali Rayan
Version: 1.0
"""

import pandas as pd
import numpy as np
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datetime import datetime, timedelta
import shutil

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Professional data storage manager with caching capabilities.
    
    Features:
    - Automatic cache expiration
    - Data versioning
    - Compression for large datasets
    - Metadata tracking
    """
    
    def __init__(
        self,
        base_path: str = './data_cache',
        default_format: str = 'parquet',
        cache_ttl_hours: int = 24
    ):
        """
        Initialize storage manager.
        
        Args:
            base_path: Root directory for all storage
            default_format: Default file format ('parquet', 'hdf5', 'csv', 'pickle')
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.base_path = Path(base_path)
        self.default_format = default_format
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Create subdirectories
        self.raw_path = self.base_path / 'raw'
        self.processed_path = self.base_path / 'processed'
        self.features_path = self.base_path / 'features'
        self.models_path = self.base_path / 'models'
        self.results_path = self.base_path / 'results'
        self.metadata_path = self.base_path / 'metadata'
        
        for path in [self.raw_path, self.processed_path, self.features_path, 
                     self.models_path, self.results_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self._metadata_file = self.metadata_path / 'storage_metadata.json'
        self._metadata = self._load_metadata()
        
        logger.info(f"DataStorage initialized at {base_path}")
    
    def _load_metadata(self) -> Dict:
        """Load storage metadata."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        return {'version': '1.0', 'entries': {}}
    
    def _save_metadata(self):
        """Save storage metadata."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)
    
    def _generate_key(self, data_identifier: str, params: Optional[Dict] = None) -> str:
        """Generate unique cache key from identifier and parameters."""
        key_string = data_identifier
        if params:
            # Sort params for consistent hashing
            param_str = json.dumps(params, sort_keys=True, default=str)
            key_string += f"_{param_str}"
        
        # Create hash
        hash_obj = hashlib.md5(key_string.encode())
        return hash_obj.hexdigest()[:16]
    
    def _get_file_path(
        self,
        key: str,
        data_type: str = 'raw',
        format: Optional[str] = None
    ) -> Path:
        """Get file path for storage."""
        fmt = format or self.default_format
        
        path_map = {
            'raw': self.raw_path,
            'processed': self.processed_path,
            'features': self.features_path,
            'models': self.models_path,
            'results': self.results_path
        }
        
        base_dir = path_map.get(data_type, self.raw_path)
        return base_dir / f"{key}.{fmt}"
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        data_type: str = 'processed',
        params: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        format: Optional[str] = None,
        compress: bool = True
    ) -> str:
        """
        Save DataFrame to storage.
        
        Args:
            df: DataFrame to save
            name: Human-readable name
            data_type: Type of data ('raw', 'processed', 'features', 'results')
            params: Parameters used to generate this data (for cache key)
            metadata: Additional metadata to store
            format: Override default format
            compress: Use compression for parquet
            
        Returns:
            Cache key for retrieval
        """
        key = self._generate_key(name, params)
        file_path = self._get_file_path(key, data_type, format)
        fmt = format or self.default_format
        
        try:
            if fmt == 'parquet':
                compression = 'snappy' if compress else None
                df.to_parquet(file_path, compression=compression, index=True)
                
            elif fmt == 'hdf5':
                df.to_hdf(file_path, key='data', mode='w', complevel=5 if compress else 0)
                
            elif fmt == 'csv':
                df.to_csv(file_path, index=True)
                
            elif fmt == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f)
            
            # Update metadata
            entry = {
                'name': name,
                'type': data_type,
                'format': fmt,
                'created': datetime.now().isoformat(),
                'rows': len(df),
                'columns': len(df.columns),
                'columns_list': list(df.columns),
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
                'params': params or {},
                'custom_metadata': metadata or {}
            }
            
            self._metadata['entries'][key] = entry
            self._save_metadata()
            
            logger.info(f"Saved {name} ({len(df)} rows) to {file_path}")
            return key
            
        except Exception as e:
            logger.error(f"Error saving {name}: {e}")
            raise
    
    def load_dataframe(
        self,
        key: str,
        data_type: str = 'processed',
        format: Optional[str] = None,
        check_ttl: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from storage.
        
        Args:
            key: Cache key or name
            data_type: Type of data
            format: Override format detection
            check_ttl: Check if cache has expired
            
        Returns:
            DataFrame or None if not found/expired
        """
        # Check if key exists in metadata (it's a generated key)
        if key not in self._metadata['entries']:
            # Try to find by name
            found_key = None
            for k, v in self._metadata['entries'].items():
                if v.get('name') == key and v.get('type') == data_type:
                    found_key = k
                    break
            if found_key:
                key = found_key
            else:
                logger.warning(f"No entry found for key/name: {key}")
                return None
        
        entry = self._metadata['entries'][key]
        
        # Check TTL
        if check_ttl:
            created = datetime.fromisoformat(entry['created'])
            if datetime.now() - created > self.cache_ttl:
                logger.info(f"Cache expired for {entry['name']}")
                return None
        
        file_path = self._get_file_path(key, data_type, entry.get('format', format))
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            fmt = entry.get('format', self.default_format)
            
            if fmt == 'parquet':
                df = pd.read_parquet(file_path)
            elif fmt == 'hdf5':
                df = pd.read_hdf(file_path, key='data')
            elif fmt == 'csv':
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif fmt == 'pickle':
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
            else:
                raise ValueError(f"Unknown format: {fmt}")
            
            logger.info(f"Loaded {entry['name']} ({len(df)} rows) from cache")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            return None
    
    def save_object(
        self,
        obj: Any,
        name: str,
        data_type: str = 'models',
        params: Optional[Dict] = None
    ) -> str:
        """
        Save arbitrary Python object (for model parameters, configs).
        
        Args:
            obj: Object to save
            name: Identifier
            data_type: Storage category
            params: Parameters dict
            
        Returns:
            Cache key
        """
        key = self._generate_key(name, params)
        file_path = self._get_file_path(key, data_type, 'pickle')
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            
            entry = {
                'name': name,
                'type': data_type,
                'format': 'pickle',
                'created': datetime.now().isoformat(),
                'object_type': type(obj).__name__,
                'params': params or {}
            }
            
            self._metadata['entries'][key] = entry
            self._save_metadata()
            
            logger.info(f"Saved object {name} ({type(obj).__name__}) to {file_path}")
            return key
            
        except Exception as e:
            logger.error(f"Error saving object {name}: {e}")
            raise
    
    def load_object(
        self,
        key: str,
        data_type: str = 'models',
        check_ttl: bool = True
    ) -> Optional[Any]:
        """Load Python object from storage."""
        # Resolve key
        if key not in self._metadata['entries']:
            for k, v in self._metadata['entries'].items():
                if v.get('name') == key and v.get('type') == data_type:
                    key = k
                    break
        
        if key not in self._metadata['entries']:
            return None
        
        entry = self._metadata['entries'][key]
        
        if check_ttl:
            created = datetime.fromisoformat(entry['created'])
            if datetime.now() - created > self.cache_ttl:
                return None
        
        file_path = self._get_file_path(key, data_type, 'pickle')
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            logger.info(f"Loaded object {entry['name']} from cache")
            return obj
        except Exception as e:
            logger.error(f"Error loading object {key}: {e}")
            return None
    
    def save_simulation_results(
        self,
        results: Dict[str, Any],
        model_type: str,
        simulation_params: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save Monte Carlo simulation results.
        
        Args:
            results: Dictionary with paths, statistics, etc.
            model_type: Type of model used
            simulation_params: Parameters for the simulation
            metadata: Additional info
            
        Returns:
            Cache key
        """
        name = f"sim_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert numpy arrays to DataFrames for storage
        storage_dict = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    storage_dict[key] = pd.DataFrame(value)
                else:
                    storage_dict[key] = pd.Series(value)
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                storage_dict[key] = value
            else:
                storage_dict[key] = value
        
        # Save as object
        full_metadata = {
            'model_type': model_type,
            'simulation_time': datetime.now().isoformat(),
            'params': simulation_params,
            'custom': metadata or {}
        }
        
        return self.save_object(
            storage_dict,
            name,
            data_type='results',
            params=simulation_params
        )
    
    def list_entries(
        self,
        data_type: Optional[str] = None,
        name_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List all stored entries with metadata.
        
        Args:
            data_type: Filter by type
            name_filter: Filter by name substring
            
        Returns:
            DataFrame with entry information
        """
        entries = []
        for key, meta in self._metadata['entries'].items():
            if data_type and meta.get('type') != data_type:
                continue
            if name_filter and name_filter.lower() not in meta.get('name', '').lower():
                continue
            
            entry = {
                'key': key,
                'name': meta.get('name'),
                'type': meta.get('type'),
                'format': meta.get('format'),
                'created': meta.get('created'),
                'rows': meta.get('rows'),
                'columns': meta.get('columns'),
                'size_bytes': meta.get('size_bytes')
            }
            entries.append(entry)
        
        return pd.DataFrame(entries)
    
    def delete_entry(self, key: str) -> bool:
        """Delete a specific entry from storage."""
        if key not in self._metadata['entries']:
            return False
        
        entry = self._metadata['entries'][key]
        file_path = self._get_file_path(key, entry['type'], entry.get('format'))
        
        try:
            if file_path.exists():
                file_path.unlink()
            del self._metadata['entries'][key]
            self._save_metadata()
            logger.info(f"Deleted entry {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting {key}: {e}")
            return False
    
    def clear_cache(self, data_type: Optional[str] = None, older_than_days: Optional[int] = None):
        """
        Clear cached entries.
        
        Args:
            data_type: Clear only specific type
            older_than_days: Clear entries older than N days
        """
        to_delete = []
        cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None
        
        for key, entry in list(self._metadata['entries'].items()):
            if data_type and entry.get('type') != data_type:
                continue
            
            if cutoff:
                created = datetime.fromisoformat(entry['created'])
                if created > cutoff:
                    continue
            
            to_delete.append(key)
        
        for key in to_delete:
            self.delete_entry(key)
        
        logger.info(f"Cleared {len(to_delete)} entries from cache")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache usage."""
        stats = {
            'total_entries': len(self._metadata['entries']),
            'by_type': {},
            'total_size_bytes': 0,
            'oldest_entry': None,
            'newest_entry': None
        }
        
        dates = []
        for entry in self._metadata['entries'].values():
            data_type = entry.get('type', 'unknown')
            stats['by_type'][data_type] = stats['by_type'].get(data_type, 0) + 1
            stats['total_size_bytes'] += entry.get('size_bytes', 0)
            dates.append(datetime.fromisoformat(entry['created']))
        
        if dates:
            stats['oldest_entry'] = min(dates).isoformat()
            stats['newest_entry'] = max(dates).isoformat()
        
        return stats


class CacheManager:
    """
    Simplified cache interface for quick data retrieval.
    """
    
    def __init__(self, storage: Optional[DataStorage] = None):
        self.storage = storage or DataStorage()
    
    def get_or_fetch(
        self,
        name: str,
        fetch_func,
        fetch_params: Optional[Dict] = None,
        force_refresh: bool = False,
        ttl_hours: int = 24
    ) -> pd.DataFrame:
        """
        Get data from cache or fetch if not available/expired.
        
        Args:
            name: Cache identifier
            fetch_func: Function to call if cache miss
            fetch_params: Parameters for fetch function
            force_refresh: Ignore cache and re-fetch
            ttl_hours: Cache TTL
            
        Returns:
            DataFrame
        """
        if not force_refresh:
            # Try to load from cache
            cached = self.storage.load_dataframe(name, check_ttl=True)
            if cached is not None:
                return cached
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {name}")
        fetch_params = fetch_params or {}
        df = fetch_func(**fetch_params)
        
        # Save to cache
        self.storage.save_dataframe(
            df,
            name,
            params=fetch_params,
            metadata={'ttl_hours': ttl_hours}
        )
        
        return df


def quick_cache(df: pd.DataFrame, name: str) -> str:
    """Quick function to cache a DataFrame."""
    storage = DataStorage()
    return storage.save_dataframe(df, name)


def quick_load(name: str) -> Optional[pd.DataFrame]:
    """Quick function to load cached DataFrame."""
    storage = DataStorage()
    return storage.load_dataframe(name)


if __name__ == "__main__":
    # Example usage
    print("Testing DataStorage...")
    
    storage = DataStorage(base_path='./test_cache')
    
    # Create sample data
    sample_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'price': np.random.randn(100).cumsum() + 2000,
        'volume': np.random.randint(1000, 10000, 100)
    }).set_index('date')
    
    # Save data
    key = storage.save_dataframe(
        sample_df,
        'gold_prices_test',
        data_type='raw',
        params={'symbol': 'GC=F', 'days': 100},
        metadata={'source': 'test'}
    )
    print(f"\nSaved with key: {key}")
    
    # Load data
    loaded = storage.load_dataframe(key, data_type='raw')
    print(f"\nLoaded data shape: {loaded.shape}")
    print(loaded.head())
    
    # List entries
    entries = storage.list_entries()
    print(f"\nStorage entries:")
    print(entries)
    
    # Cache stats
    stats = storage.get_cache_stats()
    print(f"\nCache stats: {stats}")
    
    # Cleanup
    storage.delete_entry(key)
    print("\nTest complete, cache cleaned up")