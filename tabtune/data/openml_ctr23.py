"""
OpenML-CTR23 Dataset Loader.

OpenML-CTR23 (Primary Curated Regression Standard) is a premier, highly curated
"gold standard" for tabular regression on OpenML.

Paper: https://openreview.net/pdf?id=HebAOoMm94
OpenML Suite/Study: https://www.openml.org/search?id=353&study_type=task&type=study
"""
import openml
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List
from .base import BaseDataset

logger = logging.getLogger(__name__)


class OpenMLCTR23Dataset(BaseDataset):
    """
    A data loader for datasets from the OpenML-CTR23 benchmark suite.
    
    OpenML-CTR23 (Primary Curated Regression Standard) is a premier, highly curated
    "gold standard" for tabular regression on OpenML, with tasks selected using strict criteria.
    
    Scale: It comprises 35 regression tasks.
    
    Paper: https://openreview.net/pdf?id=HebAOoMm94
    OpenML Suite/Study: https://www.openml.org/search?id=353&study_type=task&type=study
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_id = config.get('dataset_id')
        self.suite_id = 353  # OpenML-CTR23 suite ID
        if not self.dataset_id:
            raise ValueError("'dataset_id' must be provided in the config for OpenML-CTR23 datasets.")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads the specific OpenML-CTR23 dataset defined in the config.
        This method fulfills the abstract requirement of the BaseDataset class.
        Supports both dataset IDs and task IDs (converts task IDs to dataset IDs).
        """
        original_id = self.dataset_id
        logger.info(f"[OpenMLCTR23Loader] Downloading and loading OpenML-CTR23 ID: {original_id}...")
        
        try:
            # Try to load as dataset first
            dataset = openml.datasets.get_dataset(
                self.dataset_id, 
                download_data=True, 
                download_qualities=False, 
                download_all_files=True
            )
        except Exception as e:
            # If it fails with "Unknown dataset", try as task ID
            if "Unknown dataset" in str(e) or "code 111" in str(e):
                logger.warning(f"[OpenMLCTR23Loader] ID {original_id} not a dataset. Trying as TASK id...")
                try:
                    task = openml.tasks.get_task(int(original_id))
                    self.dataset_id = task.dataset_id
                    logger.info(f"[OpenMLCTR23Loader] Task {original_id} -> Dataset {self.dataset_id}. Retrying...")
                    dataset = openml.datasets.get_dataset(
                        self.dataset_id, 
                        download_data=True, 
                        download_qualities=False, 
                        download_all_files=True
                    )
                except Exception as task_error:
                    logger.error(f"[OpenMLCTR23Loader] Failed to load as task ID {original_id}: {task_error}")
                    raise e  # Raise original error
            else:
                raise e
        
        try:
            X, y, _, _ = dataset.get_data(
                dataset_format='dataframe', 
                target=dataset.default_target_attribute
            )
            
            # Basic data cleaning
            X.columns = X.columns.astype(str)
            
            target_name = y.name if y.name is not None else 'target'
            full_df = pd.concat([X, y.to_frame(name=target_name)], axis=1).dropna()
            
            if full_df.empty:
                logger.warning(f"[OpenMLCTR23Loader] Dataset {dataset.name} ({self.dataset_id}) is empty after dropping NaNs")
                self.data = pd.DataFrame()
                self.target = pd.Series(dtype='float64')
                return self.data, self.target
            
            # Set self.data and self.target
            self.data = full_df.drop(columns=[target_name])
            self.target = full_df[target_name]
            
            # Ensure target is numeric for regression
            self.target = pd.to_numeric(self.target, errors='coerce')
            
            logger.info(f"[OpenMLCTR23Loader] Loaded OpenML-CTR23 dataset '{dataset.name}' (ID: {self.dataset_id}) with shape {self.data.shape}")
            logger.info(f"[OpenMLCTR23Loader] Target variable: {target_name}, range: [{self.target.min():.4f}, {self.target.max():.4f}]")
            return self.data, self.target
            
        except Exception as e:
            logger.error(f"[OpenMLCTR23Loader] Failed to load OpenML-CTR23 ID {original_id} (dataset_id: {self.dataset_id}): {e}")
            raise e
    
    def get_available_datasets(self) -> List[int]:
        """
        Returns the list of OpenML-CTR23 dataset IDs from the suite.
        
        Returns:
            List[int]: List of dataset IDs in the OpenML-CTR23 suite
        """
        try:
            suite = openml.study.get_suite(self.suite_id)
            # Get task IDs and extract dataset IDs
            tasks = suite.tasks
            dataset_ids = []
            for task_id in tasks:
                try:
                    task = openml.tasks.get_task(task_id)
                    dataset_ids.append(task.dataset_id)
                except Exception as e:
                    logger.warning(f"[OpenMLCTR23Loader] Failed to get dataset ID for task {task_id}: {e}")
            logger.info(f"[OpenMLCTR23Loader] Found {len(dataset_ids)} datasets in OpenML-CTR23 suite")
            return dataset_ids
        except Exception as e:
            logger.error(f"[OpenMLCTR23Loader] Failed to fetch suite {self.suite_id}: {e}")
            return []
