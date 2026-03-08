"""
Data Pipeline Package

Complete data infrastructure for the Monte Carlo Gold Predictor:
- fetcher: Market data acquisition from Yahoo Finance, FRED, etc.
- cleaner: Data quality and preprocessing
- features: Feature engineering for ML models
- storage: Caching and persistence layer

Usage:
    from data import GoldDataFetcher, DataCleaner, FeatureEngineer, DataStorage
    
    # Fetch data
    fetcher = GoldDataFetcher()
    raw_data = fetcher.fetch_gold_prices('GC=F', period='10y')
    
    # Clean data
    cleaner = DataCleaner()
    clean_data = cleaner.clean_price_data(raw_data)
    
    # Engineer features
    engineer = FeatureEngineer()
    features = engineer.create_all_features(clean_data)
    
    # Cache results
    storage = DataStorage()
    storage.save_dataframe(features, 'gold_features_v1')
"""

from fetcher import GoldDataFetcher, fetch_gold_data_simple
from cleaner import DataCleaner, clean_gold_data
from features import FeatureEngineer, engineer_features
from storage import DataStorage, CacheManager, quick_cache, quick_load

__version__ = '1.0.0'
__author__ = 'Essabri ali rayan'

__all__ = [
    # Fetcher
    'GoldDataFetcher',
    'fetch_gold_data_simple',
    
    # Cleaner
    'DataCleaner',
    'clean_gold_data',
    
    # Features
    'FeatureEngineer',
    'engineer_features',
    
    # Storage
    'DataStorage',
    'CacheManager',
    'quick_cache',
    'quick_load',
]