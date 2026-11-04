import openml
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List

# We assume BaseDataset is in a central, accessible location.
# Let's import it from the talent_full.py file where you've defined it.
from .base import BaseDataset 

logger = logging.getLogger(__name__)

class OpenMLCC18Dataset(BaseDataset):
    """
    A data loader for datasets from the OpenML-CC18 benchmark suite.
    This class inherits from BaseDataset and implements the required methods.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_id = config.get('dataset_id')
        if not self.dataset_id:
            raise ValueError("'dataset_id' must be provided in the config for OpenML datasets.")
        
        # This list defines the entire CC-18 benchmark suite
        self.all_datasets = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49,
                    53, 219, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3573, 3902, 3903,
                    3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964,
                    9971, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954,
                    14965, 14969, 14970, 125920, 125922, 146195, 146800, 146817, 146819,
                    146820, 146821, 146822, 146824, 146825, 167119, 167120, 167121,
                    167124, 167125, 167140, 167141]

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads the specific OpenML dataset defined in the config.
        This method fulfills the abstract requirement of the BaseDataset class.
        """
        logger.info(f"[OpenMLLoader] Downloading and loading OpenML dataset ID: {self.dataset_id}...")
        
        try:
            # 1. Download from OpenML
            dataset = openml.datasets.get_dataset(self.dataset_id, download_data=True, download_qualities=False, download_all_files=True)
            X, y, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
            
            # 2. Perform basic data cleaning
            X.columns = X.columns.astype(str)
            
            target_name = y.name if y.name is not None else 'target'
            full_df = pd.concat([X, y.to_frame(name=target_name)], axis=1).dropna()
            
            if full_df.empty:
                logger.warning(f"[OpenMLLoader] Dataset {dataset.name} ({self.dataset_id}) is empty after dropping NaNs")
                self.data = pd.DataFrame()
                self.target = pd.Series(dtype='object')
                return self.data, self.target

            # 3. Set self.data and self.target as required by BaseDataset
            self.data = full_df.drop(columns=[target_name])
            self.target = full_df[target_name]
            
            logger.info(f"[OpenMLLoader] Loaded OpenML dataset '{dataset.name}' with shape {self.data.shape}")
            return self.data, self.target

        except Exception as e:
            logger.error(f"[OpenMLLoader] Failed to load OpenML dataset ID {self.dataset_id}: {e}")
            raise e

    def get_available_datasets(self) -> List[int]:
        """Returns the complete list of OpenML-CC18 dataset IDs."""
        return self.all_datasets