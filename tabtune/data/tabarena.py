import openml
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class TabArenaDataset:
    """
    Data loader for TabArena-v0.1 (OpenML Suite 457).
    Source: TabArena: A Living Benchmark for Machine Learning on Tabular Data [cite: 1102]
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.suite_id = 457 # TabArena-v0.1 Suite ID [cite: 2826]
        
    def get_available_datasets(self):
        """Returns the list of Task IDs in the TabArena suite."""
        try:
            suite = openml.study.get_suite(self.suite_id)
            # TabArena uses datasets derived from tasks. We list dataset IDs.
            return suite.data
        except Exception as e:
            logger.error(f"[TabArenaDataset] Failed to fetch suite {self.suite_id}: {e}")
            return []

    def get_data(self, dataset_id):
        """
        Fetches the full dataset (X, y) for Cross-Validation.
        TabArena requires IID stratified splitting, not pre-defined holdouts.
        """
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False)
            target_name = dataset.default_target_attribute
            
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe"
            )
            
            # Handle categorical features according to TabArena protocol
            # (Passing raw data is preferred, pipeline handles encoding)
            return X, y, categorical_indicator
            
        except Exception as e:
            logger.error(f"[TabArenaDataset] Error loading dataset {dataset_id}: {e}")
            raise e



    def get_splits(self, test_size=0.25):
        """
        Retrieves data and generates Train/Test splits.
        Required by BenchmarkPipeline.
        """
        # 1. Retrieve the dataset ID from the config
        # Note: BenchmarkPipeline might pass it as 'dataset_id' or 'dataset_name'
        dataset_id = self.config.get('dataset_id') or self.config.get('dataset_name')
        
        if dataset_id is None:
            raise ValueError("[TabArenaDataset] dataset_id not found in config provided to loader.")

        # 2. Fetch the raw data
        X, y, _ = self.get_data(dataset_id)

        # 3. Create the split
        # Note: TabArena officially uses 8-fold CV. This simple split acts as a proxy
        # to allow your current pipeline architecture to function without major refactoring.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }