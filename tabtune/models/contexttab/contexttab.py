# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Union

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from .constants import ZMQ_PORT_DEFAULT, ModelSize
from .scripts.start_embedding_server import start_embedding_server
from .data.tokenizer import Tokenizer
from .torch_model import ConTextTab

warnings.filterwarnings('ignore', message='.*not support non-writable tensors.*')


def to_device(x, device: Union[torch.device, int], dtype: Optional[torch.dtype] = None, raise_on_unexpected=True):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            target_dtype = dtype if v.dtype == torch.float32 else v.dtype
            x[k] = v.to(device, dtype=target_dtype)
        elif isinstance(v, dict):
            x[k] = to_device(v, device, dtype=dtype)
        elif v is not None and raise_on_unexpected:
            raise ValueError(f'Unknown type, {type(v)}')
    return x


class ConTextTabEstimator(BaseEstimator, ABC):
    """ConTextTabEstimator class.

    Args:
        checkpoint: path to the checkpoint file; must be of size base model size
        bagging: number of bagging iterations; if 1 there is no bagging. If 'auto', then:
            - There is no bagging if the number of samples is less than max_context_size - just use everything
            - Otherwise, the training data is split into chunks size max_context_size rows:
            ceil(len(dataset) / max_context_size) overlapping chunks and each chunk is then used as a bagging
            iteration (capped at MAX_AUTO_BAGS = 16).
        max_context_size: maximum number of samples to use for training
        num_regression_bins: number of bins to use for regression (to convert into classification).
            Unused if regression_type is 'l2'.
        regression_type: regression type that was used in the specified model
            - reg-as-classif - binned regression where bin is associated with the quantile of a given column
            - l2 - direct prediction of the target value with L2 loss during training
        classification_type: classification type that was used in the specified model
            - cross-entropy - class likelihood prediction using cross entropy loss during training
            - clustering - class prediction using similarity between context and query vectors
            - clustering-cosine - class prediction using cosine similarity between context and query vectors 
        is_drop_constant_columns: flag to indicate to drop constant columns in the input dataframe
        test_chunk_size: number of test rows to use for prediction at once
    """
    classification_or_regression: str
    MAX_AUTO_BAGS = 16
    MAX_NUM_COLUMNS = 500

    def __init__(self,
                 checkpoint: str = '2025-11-04_sap-rpt-one-oss.pt',
                 checkpoint_revision: str = None,
                 bagging: Union[Literal['auto'], int] = 1,
                 max_context_size: int = 8192,
                 num_regression_bins: int = 16,
                 regression_type: Literal['reg-as-classif', 'l2'] = 'l2',
                 classification_type: Literal['cross-entropy', 'clustering', 'clustering-cosine'] = 'cross-entropy',
                 is_drop_constant_columns: bool = True,
                 test_chunk_size: int = 1000):

        # Use new repository: SAP/sap-rpt-1-oss (formerly SAP/contexttab)
        self.model_size = ModelSize.base  # New repo always uses base model size
        self.checkpoint_revision = checkpoint_revision
        self.checkpoint = checkpoint
        # Update to use new repository
        if checkpoint_revision:
            self._checkpoint_path = hf_hub_download(repo_id="SAP/sap-rpt-1-oss", filename=checkpoint, revision=checkpoint_revision)
        else:
            self._checkpoint_path = hf_hub_download(repo_id="SAP/sap-rpt-1-oss", filename=checkpoint)
        self.bagging = bagging
        if not isinstance(bagging, int) and bagging != 'auto':
            raise ValueError('bagging must be an integer or "auto"')
        self.max_context_size = max_context_size
        self.num_regression_bins = num_regression_bins
        self.model = ConTextTab(self.model_size, regression_type=regression_type, classification_type=classification_type)
        # We're using a single GPU here, even if more are available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_embedding_server(Tokenizer.sentence_embedding_model_name)
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability(0)[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(Path(self._checkpoint_path), self.device)
        self.regression_type = regression_type
        self.seed = 42
        # Use drop_constant_columns to match official implementation
        self.drop_constant_columns = is_drop_constant_columns
        self.tokenizer = Tokenizer(
            regression_type=regression_type,
            classification_type=classification_type,
            zmq_port=ZMQ_PORT_DEFAULT,  # Only one GPU supported
            random_seed=self.seed,
            num_regression_bins=num_regression_bins,
            is_valid=True)
        self.model.to(self.device).eval()
        self.classification_type = classification_type
        self.test_chunk_size = test_chunk_size

    @abstractmethod
    def task_specific_fit(self):
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fit the model.

        Args:
            X: The input dataframe.
            y: The target column.
        """
        if len(X) != len(y):
            raise ValueError('X and y must have the same length')
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            # Ensure y has the same index as X to avoid alignment issues
            y = pd.Series(y, name='TARGET', index=X.index)

        self.X_ = X

        self.bagging_config = self.bagging
        if X.shape[0] < self.max_context_size:
            self.bagging_config = 1

        self.y_ = y
        # Return the classifier
        self.task_specific_fit()
        return self

    @property
    def bagging_number(self):
        check_is_fitted(self)
        if self.bagging_config == 'auto':
            return min(self.MAX_AUTO_BAGS, ceil(len(self.X_) / self.max_context_size))
        else:
            assert isinstance(self.bagging_config, int)
            return self.bagging_config

    def get_tokenized_data(self, X_test, bagging_index):
        X_train = self.X_
        y_train = self.y_

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
        y_test = pd.Series([y_train.iloc[0]] * len(X_test), name=self.y_.name, index=X_test.index)

        df_train = pd.concat([X_train, y_train.to_frame()], axis=1)
        df_test = pd.concat([X_test, y_test.to_frame()], axis=1)
        
        # Debug: Check before bagging
        import logging
        debug_logger = logging.getLogger(__name__)
        if df_train.iloc[:, -1].isna().sum() > 0:
            debug_logger.error(
                f"[ConTextTabRegressor] df_train target has NaN BEFORE bagging! "
                f"Bagging index: {bagging_index}, NaN count: {df_train.iloc[:, -1].isna().sum()}, "
                f"Shape: {df_train.shape}, X_train index: {X_train.index.tolist()[:5]}, y_train index: {y_train.index.tolist()[:5]}"
            )

        if isinstance(self.bagging_config, int) and self.bagging_config > 1:
            # For bagging, we use replacement
            df_train = df_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)
            
            # Debug: Check after bagging
            if df_train.iloc[:, -1].isna().sum() > 0:
                debug_logger.error(
                    f"[ConTextTabRegressor] df_train target has NaN AFTER bagging sample! "
                    f"Bagging index: {bagging_index}, NaN count: {df_train.iloc[:, -1].isna().sum()}"
                )
        elif len(df_train) > self.max_context_size:
            if isinstance(self.bagging_config, str):
                assert self.bagging_config == 'auto'
                # Split the data into overlapping chunks of size max_context_size
                # Randomize order as well, to have balanced bags
                # bagging_index = 0 --> select 0:max_context_size
                # ... (linearly spaced like np.linspace(0, len(df_train) - max_context_size, self.bagging_number))
                # bagging_index = self.bagging_number - 1 --> select (len(df_train) - max_context_size):len(df_train)
                start = int((len(df_train) - self.max_context_size) / (self.bagging_number - 1) * bagging_index)
                # We need a fixed seed, so across diffent "bagging folds" we select the correct indices
                # (as non-overlapping as possible)
                np.random.seed(self.seed)
                indices = np.random.permutation(df_train.index)
                end = start + self.max_context_size
                df_train = df_train.loc[indices[start:end]]
            else:
                # There is no bagging, but we still have to sample because there are too many points
                df_train = df_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)

        df = pd.concat([df_train, df_test], ignore_index=True)
        
        # Store the original training data length before any modifications
        original_train_len = len(df_train)
        
        # Debug: Check target column after initial concat
        import logging
        debug_logger = logging.getLogger(__name__)
        target_col_after_concat = df.iloc[:, -1]
        if target_col_after_concat.isna().sum() > 0:
            debug_logger.error(
                f"[ConTextTabRegressor] Target column has NaN after initial concat! "
                f"Bagging index: {bagging_index}, NaN count: {target_col_after_concat.isna().sum()}, "
                f"Total rows: {len(df)}, Train rows: {original_train_len}"
            )

        if self.drop_constant_columns:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            constant_cols = list(X.columns[X.nunique() == 1])
            if constant_cols:
                X = X.drop(columns=constant_cols)
                # Ensure indices align when concatenating
                df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
                # Debug: Check after dropping constant columns
                if df.iloc[:, -1].isna().sum() > 0:
                    debug_logger.error(
                        f"[ConTextTabRegressor] Target column has NaN after dropping constant columns! "
                        f"Bagging index: {bagging_index}, NaN count: {df.iloc[:, -1].isna().sum()}"
                    )

        if df.shape[1] > self.MAX_NUM_COLUMNS:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            X = X.sample(n=self.MAX_NUM_COLUMNS - 1, axis=1, random_state=self.seed + bagging_index, replace=False)
            # Ensure indices align when concatenating
            df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            # Debug: Check after column sampling
            if df.iloc[:, -1].isna().sum() > 0:
                debug_logger.error(
                    f"[ConTextTabRegressor] Target column has NaN after column sampling! "
                    f"Bagging index: {bagging_index}, NaN count: {df.iloc[:, -1].isna().sum()}"
                )

        # Use the stored original_train_len to correctly split train and test
        df_train = df.iloc[:original_train_len].copy()
        df_test = df.iloc[original_train_len:].copy()
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1:]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1:]

        # Debug: Check y_train and y_test before tokenization
        if y_train.isna().sum().sum() > 0:
            debug_logger.error(
                f"[ConTextTabRegressor] y_train has NaN before tokenization! "
                f"Bagging index: {bagging_index}, NaN count: {y_train.isna().sum().sum()}, "
                f"Shape: {y_train.shape}, Columns: {y_train.columns.tolist()}"
            )
        if y_test.isna().sum().sum() > 0:
            debug_logger.error(
                f"[ConTextTabRegressor] y_test has NaN before tokenization! "
                f"Bagging index: {bagging_index}, NaN count: {y_test.isna().sum().sum()}"
            )

        data, labels, label_classes = self.tokenizer(X_train, y_train, X_test, y_test,
                                                     self.classification_or_regression)

        # Debug: Check target data immediately after tokenization
        if 'target' in data and isinstance(data['target'], torch.Tensor):
            nan_after_tokenizer = torch.sum(~torch.isfinite(data['target'])).item()
            if nan_after_tokenizer > 0:
                debug_logger.error(
                    f"[ConTextTabRegressor] Found {nan_after_tokenizer} NaN/Inf in target AFTER tokenizer! "
                    f"Bagging index: {bagging_index}, Shape: {data['target'].shape}, "
                    f"Min: {data['target'].min().item()}, Max: {data['target'].max().item()}"
                )

        target_mean, target_std = 0, 0
        is_regression = self.classification_or_regression == 'regression'
        if is_regression and self.regression_type == 'l2':
            # Debug: Check inputs to standard_scale_column
            debug_logger.debug(
                f"[ConTextTabRegressor] Calling standard_scale_column with "
                f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}, "
                f"y_train NaN: {y_train.isna().sum().sum()}, y_test NaN: {y_test.isna().sum().sum()}"
            )
            _, target_mean, target_std = self.tokenizer.standard_scale_column(y_train, y_test)
            
            # Debug: Check target_mean and target_std
            if isinstance(target_mean, torch.Tensor) and not torch.isfinite(target_mean).all():
                debug_logger.error(f"[ConTextTabRegressor] target_mean has NaN/Inf: {target_mean}")
            if isinstance(target_std, torch.Tensor) and not torch.isfinite(target_std).all():
                debug_logger.error(f"[ConTextTabRegressor] target_std has NaN/Inf: {target_std}")
            elif isinstance(target_std, torch.Tensor) and target_std.item() < 1e-10:
                debug_logger.warning(f"[ConTextTabRegressor] target_std is very small: {target_std.item()}")

        return {
            'data': data,
            'num_rows': df.shape[0],
            'num_cols': df.shape[1],
            'labels': None,
            'is_regression': torch.tensor(is_regression),
            'label_classes': np.asarray(label_classes),
            'target_mean': target_mean,
            'target_std': target_std
        }

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[list, np.ndarray]:
        pass


class ConTextTabClassifier(ClassifierMixin, ConTextTabEstimator):
    classification_or_regression = 'classification'

    def task_specific_fit(self):
        # Store the classes seen during fit
        self.classes_ = unique_labels(self.y_)

    def reorder_logits(self, logits, tokenized_classes):
        class_mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        indices = np.array([class_mapping[cls] for cls in tokenized_classes])
        new_logits = torch.full((logits.shape[0], len(self.classes_)),
                                float('-inf'),
                                dtype=logits.dtype,
                                device=logits.device)
        new_logits[:, indices] = logits[:, :len(tokenized_classes)]
        return new_logits

    @torch.no_grad()
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]):
        # Check if fit has been called
        check_is_fitted(self)

        all_logits = []

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)

            try:
                tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False, dtype=self.dtype)
            except TypeError:
                # Legacy compatibility
                tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False)
            logits_classif = self.model(**tokenized_data)

            _, logits = self.model.extract_prediction_classification(logits_classif, tokenized_data['data']['target'],
                                                                     tokenized_data['label_classes'])

            all_logits.append(self.reorder_logits(logits, tokenized_data['label_classes']))

        all_logits = torch.stack(all_logits)
        if self.classification_type in ['clustering', 'clustering-cosine']:
            # Pick the class of the most similar element across folds, then softmax
            all_logits = torch.max(all_logits, dim=0).values.cpu().float()
            probs = torch.nn.functional.softmax(all_logits, dim=-1)
        else:
            # Average probabilities across folds, i.e. first softmax, then mean
            all_probs = torch.nn.functional.softmax(all_logits, dim=-1)
            probs = all_probs.mean(dim=0)

        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> list:
        """Predict the class labels for the provided input dataframe.

        Args:
            X: The input dataframe.

        Returns:
            The predicted class labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            probs.append(pred)
        probs = torch.cat(probs)

        preds = probs.argmax(dim=-1).numpy()
        return [self.classes_[p] for p in preds]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input dataframe.

        Args:
            X: The input data.

        Returns:
            The predicted probabilities of the classes.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            probs.append(pred)
        probs = torch.cat(probs)
        return probs.numpy()


class ConTextTabRegressor(RegressorMixin, ConTextTabEstimator):
    classification_or_regression = 'regression'

    def task_specific_fit(self):
        self.y_ = self.y_.astype(float)

    @torch.no_grad()
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable.

        Args:
            X: The input dataframe.

        Returns:
            The predicted target variable.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if fit has been called
        check_is_fitted(self)

        all_preds = []

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)
            
            # Debug: Check target data before device transfer
            if 'data' in tokenized_data and isinstance(tokenized_data['data'], dict):
                target_before = tokenized_data['data'].get('target')
                if target_before is not None and isinstance(target_before, torch.Tensor):
                    nan_before = torch.sum(~torch.isfinite(target_before)).item()
                    if nan_before > 0:
                        logger.error(
                            f"[ConTextTabRegressor] Found {nan_before} NaN/Inf in target BEFORE device transfer! "
                            f"Bagging index: {bagging_index}, Shape: {target_before.shape}, "
                            f"Min: {target_before.min().item()}, Max: {target_before.max().item()}"
                        )
            
            tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False)
            
            # Debug: Check target data after device transfer
            if 'data' in tokenized_data and isinstance(tokenized_data['data'], dict):
                target_after = tokenized_data['data'].get('target')
                if target_after is not None and isinstance(target_after, torch.Tensor):
                    nan_after = torch.sum(~torch.isfinite(target_after)).item()
                    if nan_after > 0:
                        logger.error(
                            f"[ConTextTabRegressor] Found {nan_after} NaN/Inf in target AFTER device transfer! "
                            f"Bagging index: {bagging_index}, Shape: {target_after.shape}, "
                            f"Device: {target_after.device}, Dtype: {target_after.dtype}"
                        )
            
            # Validate input data before forward pass
            if 'data' in tokenized_data:
                data_tensor = tokenized_data['data']
                if isinstance(data_tensor, dict):
                    for key, value in data_tensor.items():
                        if isinstance(value, torch.Tensor):
                            if torch.any(~torch.isfinite(value)):
                                nan_count = torch.sum(~torch.isfinite(value)).item()
                                logger.warning(
                                    f"[ConTextTabRegressor] Found {nan_count} NaN/Inf values in input data['{key}']. "
                                    f"Replacing with zeros."
                                )
                                data_tensor[key] = torch.where(
                                    ~torch.isfinite(value),
                                    torch.zeros_like(value),
                                    value
                                )
            
            # Ensure model is in eval mode
            self.model.eval()
            
            logits_reg = self.model(**tokenized_data)
            label_classes = tokenized_data['label_classes']

            if self.regression_type != 'l2':
                if len(label_classes) != self.num_regression_bins:
                    raise ValueError(f'Expected {self.num_regression_bins} classes, got {len(label_classes)}')

            # Check if logits are NaN/Inf before extraction
            if isinstance(logits_reg, torch.Tensor):
                if torch.any(~torch.isfinite(logits_reg)):
                    nan_count = torch.sum(~torch.isfinite(logits_reg)).item()
                    total_count = logits_reg.numel()
                    
                    if nan_count == total_count:
                        # ALL logits are NaN - this is a serious issue, skip this bagging iteration
                        logger.error(
                            f"[ConTextTabRegressor] ALL logits are NaN/Inf in bagging iteration {bagging_index}. "
                            f"Shape: {logits_reg.shape}. Skipping this iteration and using mean prediction."
                        )
                        # Skip this bagging iteration - will use mean as fallback
                        num_test = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
                        preds = np.full(num_test, float(self.y_.mean()), dtype=np.float32)
                        all_preds.append(preds)
                        continue
                    else:
                        logger.warning(
                            f"[ConTextTabRegressor] Found {nan_count}/{total_count} NaN/Inf values in logits. "
                            f"Shape: {logits_reg.shape}, Replacing with zeros (normalized mean)."
                        )
                        # Replace NaN/Inf logits with zeros (which represent normalized mean after denormalization)
                        logits_reg = torch.where(
                            ~torch.isfinite(logits_reg),
                            torch.zeros_like(logits_reg),
                            logits_reg
                        )

            # Get target_mean and target_std, with validation
            target_mean = tokenized_data.get('target_mean')
            target_std = tokenized_data.get('target_std')
            
            # Validate target_mean and target_std are finite
            if target_mean is not None:
                if isinstance(target_mean, torch.Tensor):
                    if not torch.isfinite(target_mean).all():
                        # Fallback to mean of training data
                        target_mean = torch.tensor(float(self.y_.mean()), dtype=target_mean.dtype, device=target_mean.device)
                elif not np.isfinite(target_mean):
                    target_mean = float(self.y_.mean())
            
            if target_std is not None:
                if isinstance(target_std, torch.Tensor):
                    if not torch.isfinite(target_std).all() or target_std.item() < 1e-10:
                        # Fallback to std of training data, or 1.0 if std is too small
                        fallback_std = max(float(self.y_.std()), 1e-10)
                        target_std = torch.tensor(fallback_std, dtype=target_std.dtype, device=target_std.device)
                elif not np.isfinite(target_std) or target_std < 1e-10:
                    target_std = max(float(self.y_.std()), 1e-10)
            
            preds, _ = self.model.extract_prediction_regression(logits_reg,
                                                                tokenized_data['data']['target'],
                                                                tokenized_data['label_classes'],
                                                                target_mean=target_mean,
                                                                target_std=target_std)
            
            # Validate predictions before appending
            if len(preds) == 0:
                # Empty predictions - use mean as fallback
                num_test = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
                preds = np.full(num_test, float(self.y_.mean()), dtype=np.float32)
            elif np.any(~np.isfinite(preds)):
                # Replace NaN/Inf with mean
                nan_inf_mask = ~np.isfinite(preds)
                replacement_value = float(self.y_.mean())
                preds[nan_inf_mask] = replacement_value
            
            all_preds.append(preds)

        # Average predictions across bagging iterations
        if len(all_preds) > 0:
            # Filter out empty arrays
            valid_preds = [p for p in all_preds if len(p) > 0]
            if len(valid_preds) > 0:
                preds = np.mean(valid_preds, axis=0)
            else:
                # All predictions were empty - use mean as fallback
                num_test = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
                preds = np.full(num_test, float(self.y_.mean()), dtype=np.float32)
        else:
            # No predictions at all - use mean as fallback
            num_test = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
            preds = np.full(num_test, float(self.y_.mean()), dtype=np.float32)

        return preds

    @torch.no_grad()
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable.

        Args:
            X: The input dataframe.

        Returns:
            The predicted target variable.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        preds = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            preds.append(pred)
        preds = np.concatenate(preds)
        return preds
