import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import requests
import zipfile
import json
import os
from urllib.parse import urljoin
import gdown
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

class BaseDataset(ABC):
    """Abstract base class for datasets."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.target = None
        self.feature_names = None
        self.target_names = None
    @abstractmethod
    
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and return the dataset."""
        pass


    def get_splits(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Splits the loaded data into train and test sets.
        
        This version is more robust: it checks if stratification is possible
        and falls back to a random split if not.
        """
        from sklearn.model_selection import train_test_split
        if self.data is None or self.target is None:
            self.load_data()
        
        if self.data.empty:
            return {'train': (pd.DataFrame(), pd.Series()), 'test': (pd.DataFrame(), pd.Series())}

        # --- START OF THE FIX ---
        stratify_target = None
        y_series = pd.Series(self.target)
        
        # Check if stratification is possible: more than one class and at least 2 samples in the smallest class
        if y_series.nunique() > 1 and y_series.value_counts().min() > 1:
            stratify_target = self.target
        else:
            logger.warning(f"[BaseDataLoader] Cannot stratify dataset. A class has only 1 sample. Falling back to random split")
        # --- END OF THE FIX ---

        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=test_size,
            random_state=random_state, stratify=stratify_target
        )
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }
