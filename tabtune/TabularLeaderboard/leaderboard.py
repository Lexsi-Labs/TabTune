import pandas as pd
import numpy as np
import logging
from IPython.display import display, Markdown

from ..TabularPipeline.pipeline import TabularPipeline
from ..logger import setup_logger

logger = logging.getLogger('tabtune')

class TabularLeaderboard:
    """
    A leaderboard utility to benchmark multiple TabTune pipeline configurations
    on a single, pre-split dataset.
    """
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, task_type: str = "classification",):
        """
        Initializes the leaderboard with a user-provided, pre-split dataset.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target.
            y_test (pd.Series): Testing target.
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type
        
        self.models_to_run = []
        self.results = []
        
        logger.info("[Leaderboard] Leaderboard initialized with custom dataset")
        logger.info(f"[Leaderboard] Data prepared with {self.X_train.shape[0]} training samples and {self.X_test.shape[0]} test samples")


    def add_model(self, model_name: str, tuning_strategy: str = 'inference', 
                  model_params: dict = None, tuning_params: dict = None, finetune_mode: str = None, task_type: str = None,):
        """
        Adds a model configuration to the list of contestants for the leaderboard.
        """
        if task_type is None:
            task_type = self.task_type

        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")
            
        config = {
            "model_name": model_name,
            "tuning_strategy": tuning_strategy,
            "model_params": model_params or {},
            "tuning_params": tuning_params or {},
            "finetune_mode": finetune_mode,
            "task_type": task_type
        }
        self.models_to_run.append(config)
        logger.info(f"[Leaderboard] Added to leaderboard: {model_name} (Strategy: {tuning_strategy})")

    def run(self, rank_by: str = 'roc_auc_score'):
        """
        Runs all added model configurations, evaluates them, and displays a sorted leaderboard.
        """
        logger.info("\n" + "="*60)
        logger.info("[Leaderboard] Starting Leaderboard Run")
        
        for i, config in enumerate(self.models_to_run):
            logger.info("\n" + "="*40)
            logger.info(f"[Leaderboard] [{i+1}/{len(self.models_to_run)}] Running: {config['model_name']} (Strategy: {config['tuning_strategy']})")
            logger.info("="*40)
            
            try:
                pipeline = TabularPipeline(
                    model_name=config['model_name'],
                    task_type=config['task_type'],
                    tuning_strategy=config['tuning_strategy'],
                    model_params=config['model_params'],
                    tuning_params=config['tuning_params'],
                    finetune_mode=config['finetune_mode']
                )
                
                pipeline.fit(self.X_train, self.y_train)
                
                metrics = pipeline.evaluate(self.X_test, self.y_test)
                
                result_row = {
                    'Model': config['model_name'],
                    'Strategy': config['tuning_strategy']
                }
                
                if config['task_type'] == 'classification':
                    result_row.update({
                        'Accuracy': metrics.get('accuracy', 0),
                        'F1 Score': metrics.get('f1_score', 0),
                        'ROC AUC': metrics.get('roc_auc_score', 0)
                    })
                elif config['task_type'] == 'regression':
                    result_row.update({
                        'MSE': metrics.get('mse', float('nan')),
                        'RMSE': metrics.get('rmse', float('nan')),
                        'MAE': metrics.get('mae', float('nan')),
                        'R2 Score': metrics.get('r2_score', float('nan'))
                    })
                    
                self.results.append(result_row)

            except Exception as e:
                logger.error(f"[Leaderboard] Error running {config['model_name']}: {e}")
                result_row = {
                    'Model': config['model_name'],
                    'Strategy': config['tuning_strategy'],
                    'Finetune Mode': config['finetune_mode'],
                    'Status': 'Failed'
                }
                if config['task_type'] == 'classification':
                    result_row.update({'Accuracy': 'Failed', 'F1 Score': 'Failed', 'ROC AUC': 'Failed'})
                elif config['task_type'] == 'regression':
                    result_row.update({'MSE': 'Failed', 'RMSE': 'Failed'})

                self.results.append(result_row)

        logger.info("\n" + "="*60)
        logger.info("[Leaderboard] Leaderboard Complete")
        
        leaderboard_df = pd.DataFrame(self.results)
        
        leaderboard_df = pd.DataFrame(self.results)
        
        sort_map = {
            'accuracy': 'Accuracy', 'f1_score': 'F1 Score', 'roc_auc_score': 'ROC AUC',
            'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2_score': 'R2 Score'
        }
        sort_column = sort_map.get(rank_by, rank_by)
        
        # Determine sort order: descending for scores, ascending for errors
        ascending = False
        if sort_column in ['MSE', 'RMSE', 'MAE']:
            ascending = True

        
        if sort_column in leaderboard_df.columns:
            # Coerce errors to NaN to handle 'Failed' strings during sort, then sort
            temp_col = pd.to_numeric(leaderboard_df[sort_column], errors='coerce')
            leaderboard_df = leaderboard_df.iloc[temp_col.sort_values(ascending=ascending).index].reset_index(drop=True)
            
            leaderboard_df.index = leaderboard_df.index + 1
            leaderboard_df.index.name = 'Rank'

        display(Markdown("Leaderboard Results"))
        display(leaderboard_df)
        return leaderboard_df
