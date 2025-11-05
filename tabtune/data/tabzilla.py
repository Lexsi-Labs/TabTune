import logging
from typing import Dict, Any
import openml
from .openml import OpenMLCC18Dataset

logger = logging.getLogger(__name__)

class TabZillaDataset(OpenMLCC18Dataset):
    """
    Reuses the OpenML loader but adds Dataset→Task fallback for TabZilla IDs.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def load_data(self):
        try:
            return super().load_data()
        except Exception as e:
            # If openml.py wasn't patched globally, do a local fallback here:
            if "Unknown dataset" in str(e) or "code 111" in str(e):
                tid = int(self.dataset_id)
                logger.warning(f"[TabZillaLoader] ID {tid} not a dataset. Trying as TASK id...")
                task = openml.tasks.get_task(tid)
                self.dataset_id = task.dataset_id
                logger.info(f"[TabZillaLoader] Task {tid} -> Dataset {self.dataset_id}. Retrying")
                return super().load_data()
            raise
