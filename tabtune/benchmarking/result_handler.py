import pandas as pd
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class ResultsHandler:
    """
    Manages the collection, display, and incremental saving of benchmark results.
    
    MODIFIED: This class now loads existing results from the CSV file upon 
    initialization to make the benchmark pipeline resumable.
    """
    def __init__(self, filename: str | None = None):
        """
        Initializes the handler and loads any existing results from the CSV file.

        Args:
            filename (str, optional): The path to the output CSV file. 
                                      If None, a timestamped name is generated.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        self.filename = filename
        
        # --- START MODIFICATION ---
        # Load existing results from the file, if it exists
        self.results_df = self._load_existing_results()
        # --- END MODIFICATION ---
        
        self._header_written = os.path.exists(self.filename)
        
        if self._header_written and not self.results_df.empty:
            logger.info(f"[ResultsHandler] ResultsHandler initialized. Loaded {len(self.results_df)} existing results from '{self.filename}'")
        else:
            logger.info(f"[ResultsHandler] ResultsHandler initialized. Will save new results to '{self.filename}'")

    def _load_existing_results(self) -> pd.DataFrame:
        """Helper to load existing CSV data if it exists."""
        if os.path.exists(self.filename):
            try:
                # Try to read the CSV
                return pd.read_csv(self.filename)
            except pd.errors.EmptyDataError:
                # Handle case where file exists but is empty
                logger.warning(f"[ResultsHandler] Results file '{self.filename}' is empty. Starting fresh")
                return pd.DataFrame()
            except Exception as e:
                # Handle other read errors
                logger.error(f"[ResultsHandler] Failed to load existing results from '{self.filename}': {e}. Starting fresh")
                return pd.DataFrame()
        else:
            # File doesn't exist, return an empty DataFrame
            return pd.DataFrame()

    def add_result(self, model_name: str, dataset_id: any, dataset_name: str, metrics: dict, fit_time: float, eval_time: float):
        """
        Adds a new result, appends it to the in-memory DataFrame, and writes it to the CSV file immediately.
        """
        result_record = {
            "model_name": model_name,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "fit_time_seconds": round(fit_time, 2),
            "eval_time_seconds": round(eval_time, 2),
            **{k: round(v, 4) for k, v in metrics.items()}
        }
        
        # Create a single-row DataFrame for the new result
        df_to_append = pd.DataFrame([result_record])

        # --- START MODIFICATION ---
        # Add to the in-memory DataFrame
        self.results_df = pd.concat([self.results_df, df_to_append], ignore_index=True)
        # --- END MODIFICATION ---

        # --- Append to CSV immediately ---
        try:
            # If the header hasn't been written yet, write it.
            # Otherwise, append without the header.
            if not self._header_written:
                df_to_append.to_csv(self.filename, index=False, mode='a')
                self._header_written = True
            else:
                df_to_append.to_csv(self.filename, index=False, mode='a', header=False)
                
        except Exception as e:
            logger.error(f"[ResultsHandler] Failed to write result to CSV '{self.filename}'. Error: {e}")

    def print_summary(self):
        """Prints a formatted summary table of ALL collected results (old and new) to the console."""
        # --- MODIFICATION: Use self.results_df ---
        if self.results_df.empty:
            logger.info("[ResultsHandler] No results to display")
            return
            
        df = self.results_df
        
        # Dynamically get the benchmark name from the filename for a cleaner title
        benchmark_title = "Benchmark Summary"
        try:
            # Extracts 'openml-cc18_TabPFN' from 'benchmark_results_openml-cc18_TabPFN.csv'
            base_name = os.path.basename(self.filename).replace('benchmark_results_', '').replace('.csv', '')
            benchmark_title = f"--- {base_name.upper()} Benchmark Summary ---"
        except IndexError:
            pass # Keep default title if filename format is unexpected

        print("\n" + "="*80)
        print(benchmark_title)
        print("="*80)
        # Use .to_string() to ensure the whole dataframe is printed
        print(df.to_string()) 
        print("="*80)

    def finalize(self):
        """Prints a final confirmation message."""
        logger.info(f"[ResultsHandler] All results have been saved incrementally to '{self.filename}'")