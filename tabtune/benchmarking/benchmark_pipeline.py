import time
import logging
import pandas as pd
import torch
import gc
from sklearn.model_selection import train_test_split

from ..TabularPipeline.pipeline import TabularPipeline
from .result_handler import ResultsHandler

# Import all data loader classes
from ..data.talent_full import TALENTFullDataset
from ..data.openml import OpenMLCC18Dataset
from ..data.tabzilla import TabZillaDataset

logger = logging.getLogger(__name__)

class BenchmarkPipeline:
    """
    An orchestrator for running benchmark suites on multiple models.
    """
    def __init__(self, models_to_benchmark: dict, benchmark_name: str, data_config: dict = None):
        """
        Initializes the BenchmarkPipeline.
        """
        self.models_to_benchmark = models_to_benchmark
        self.benchmark_name = benchmark_name
        self.data_config = data_config or {}
        
        filename = f"benchmark_results_{self.benchmark_name}.csv"
        self.results_handler = ResultsHandler(filename=filename)
        
        self.loader_map = {
            'talent': TALENTFullDataset,
            'openml-cc18': OpenMLCC18Dataset,
            'tabzilla': TabZillaDataset,
        }

        if self.benchmark_name not in self.loader_map:
            raise ValueError(f"Benchmark '{self.benchmark_name}' is not supported. Available: {list(self.loader_map.keys())}")
            
        self.DatasetLoaderClass = self.loader_map[self.benchmark_name]
        
        logger.info(f"[BenchmarkPipeline] BenchmarkPipeline initialized for '{self.benchmark_name}' benchmark")
        logger.info(f"[BenchmarkPipeline] Models to be tested: {list(self.models_to_benchmark.keys())}")

    def _cleanup_gpu_memory(self):
        """Comprehensive GPU memory cleanup"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                logger.debug("[BenchmarkPipeline] GPU memory cleaned successfully")
            except Exception as e:
                logger.warning(f"[BenchmarkPipeline] GPU cleanup failed: {e}")

    # --- THIS IS THE CORRECTED METHOD DEFINITION ---
    def run(self, dataset_list: list = None, test_size: float = 0.25):
        """
        Executes the full benchmark run, saving results into a separate file for each model.
        """
        # Auto-discover datasets if no list is provided
        if dataset_list is None:
            logger.info("[BenchmarkPipeline] No specific dataset list provided. Attempting to discover datasets automatically...")
            discovery_loader = self.DatasetLoaderClass(config=self.data_config)
            dataset_list = discovery_loader.get_available_datasets()
            if not dataset_list:
                logger.error(f"[BenchmarkPipeline] Auto-discovery failed: No datasets found for '{self.benchmark_name}'")
                return

        logger.info(f"[BenchmarkPipeline] Benchmark will run on {len(dataset_list)} datasets")

        # --- MODIFIED: Outer loop is now over models ---
        for model_key, model_config in self.models_to_benchmark.items():
            
            # 1. Create a dedicated ResultsHandler for this specific model
            filename = f"benchmark_results_{self.benchmark_name}_{model_key}.csv"
            model_results_handler = ResultsHandler(filename=filename)
            logger.info(f"\n[BenchmarkPipeline] {'='*30} Starting runs for model: {model_key} {'='*30}")
            logger.info(f"[BenchmarkPipeline] Results for this model will be saved to '{filename}'")

            # 2. Inner loop over datasets
            for dataset_identifier in dataset_list:
                
                # --- START MODIFICATION: Check for existing results ---
                
                # Get the dataset name for logging
                if self.benchmark_name in ('openml-cc18', 'tabzilla'):
                    dataset_name_for_log = f"OpenML-ID-{dataset_identifier}"
                else:
                    dataset_name_for_log = dataset_identifier

                # Check if this combo (model_key, dataset_identifier) is already in the DataFrame
                is_completed = False
                if not model_results_handler.results_df.empty:
                    try:
                        # We use .astype(str) to handle cases where one is int and other is str
                        existing_run = (
                            (model_results_handler.results_df['model_name'] == model_key) &
                            (model_results_handler.results_df['dataset_id'].astype(str) == str(dataset_identifier))
                        )
                        is_completed = existing_run.any()
                    except KeyError as e:
                        logger.warning(f"[BenchmarkPipeline] Could not check for existing results: Missing column {e}. Will run the benchmark")
                        is_completed = False # Run if columns are missing

                if is_completed:
                    logger.info(f"[BenchmarkPipeline] SKIPPING: Results for '{model_key}' on '{dataset_name_for_log}' (ID: {dataset_identifier}) already exist")
                    continue # Skip to the next dataset
                
                try:
                    data_loader_config = {**self.data_config}
                    if self.benchmark_name in ('openml-cc18', 'tabzilla'):
                        data_loader_config['dataset_id'] = dataset_identifier
                        dataset_name_for_log = f"OpenML-ID-{dataset_identifier}"
                    else:
                        data_loader_config['dataset_name'] = dataset_identifier
                        dataset_name_for_log = dataset_identifier

                    logger.info(f"\n[BenchmarkPipeline] Processing Dataset: {dataset_name_for_log}")
                    
                    data_loader = self.DatasetLoaderClass(config=data_loader_config)
                    splits = data_loader.get_splits(test_size=test_size)
                    X_train, y_train, X_test, y_test = (*splits['train'], *splits['test'])
                    
                    if X_train.empty:
                        logger.warning("[BenchmarkPipeline] Skipping dataset because it has no data")
                        continue

                    # 3. Run the model and pass the dedicated results handler
                    self._run_model_on_data(
                        model_key, model_config, dataset_name_for_log, dataset_identifier,
                        X_train, y_train, X_test, y_test,
                        results_handler=model_results_handler # Pass the correct handler
                    )
                except Exception as data_e:
                    logger.error(f"[BenchmarkPipeline] FAILED to process dataset '{dataset_identifier}'. Reason: {data_e}", exc_info=False)
                    # Cleanup GPU memory even on data loading errors
                    self._cleanup_gpu_memory()
                    continue  # Continue to next dataset instead of crashing

            # 4. Finalize results for the current model
            logger.info(f"\n[BenchmarkPipeline] Finalizing results for model: {model_key}")
            model_results_handler.print_summary()
            model_results_handler.finalize()

    def _run_model_on_data(self, model_key, model_config, dataset_name, dataset_id, X_train, y_train, X_test, y_test, results_handler: ResultsHandler):
        """
        Enhanced helper function with robust error handling and GPU cleanup.
        """
        logger = logging.getLogger(__name__)
        
        # Pre-run GPU cleanup
        self._cleanup_gpu_memory()
        
        pipeline = None
        try:
            logger.info(f"[BenchmarkPipeline] Running Model: '{model_key}' on '{dataset_name}'")
            
            # Handle ContextTab special case
            if model_key == 'ContextTab':
                X_train = X_train.copy()
                X_test = X_test.copy()
                
                cat_cols = X_train.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    logger.info(f"[BenchmarkPipeline] Cleaning {len(cat_cols)} categorical columns for ContextTab")
                    for col in cat_cols:
                        X_train[col] = X_train[col].fillna('_missing_').astype(str)
                        X_test[col] = X_test[col].fillna('_missing_').astype(str)
            
            # Create pipeline
            pipeline = TabularPipeline(**model_config)
            
            # Fit model
            start_fit = time.time()
            pipeline.fit(X_train, y_train)
            end_fit = time.time()
            
            # Evaluate model
            start_eval = time.time()
            metrics = pipeline.evaluate(X_test, y_test, output_format='json')
            end_eval = time.time()
            
            # Calibration evaluation
            start_calib = time.time()
            calibration_metrics = pipeline.evaluate_calibration(
                X_test=X_test,
                y_test=y_test,
                n_bins=15,
                output_format='json'
            )
            end_calib = time.time()

            combined_metrics = {
                **metrics,  # Standard metrics (accuracy, precision, recall, F1, AUC, MCC)
                **calibration_metrics  # Calibration metrics (brier_score_loss, expected_calibration_error, maximum_calibration_error)
            }

            fit_time = end_fit - start_fit
            eval_time = (end_eval - start_eval) + (end_calib - start_calib)
            
            # Save results
            results_handler.add_result(
                model_key, dataset_id, dataset_name, combined_metrics, fit_time, eval_time
            )
            
            logger.info(f"[BenchmarkPipeline] SUCCESS: Model '{model_key}' on '{dataset_name}'. Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[BenchmarkPipeline] GPU OOM: Model '{model_key}' on '{dataset_name}'. Reason: {e}")
            logger.info("[BenchmarkPipeline] Attempting GPU cleanup...")
            self._cleanup_gpu_memory()
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e) or "invalid configuration" in str(e):
                logger.error(f"[BenchmarkPipeline] CUDA ERROR: Model '{model_key}' on '{dataset_name}'. Reason: {e}")
                logger.info("[BenchmarkPipeline] Attempting GPU cleanup...")
                self._cleanup_gpu_memory()
            else:
                logger.error(f"[BenchmarkPipeline] RUNTIME ERROR: Model '{model_key}' on '{dataset_name}'. Reason: {e}")
                
        except Exception as e:
            logger.error(f"[BenchmarkPipeline] GENERAL ERROR: Model '{model_key}' on '{dataset_name}'. Reason: {e}")
            
        finally:
            # Always cleanup GPU memory after each dataset
            self._cleanup_gpu_memory()
            
            # Explicitly delete pipeline to free memory
            if pipeline is not None:
                del pipeline
                pipeline = None
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"[BenchmarkPipeline] Memory cleanup completed for '{dataset_name}'")
