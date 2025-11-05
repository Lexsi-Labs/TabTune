from __future__ import annotations

import warnings
from pathlib import Path
from packaging import version
from typing import Optional, List, Dict

import numpy as np
import torch

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from .preprocessing import TransformToNumerical, EnsembleGenerator
from ..model.inference_config import InferenceConfig
from ..model.orion_bix import OrionBix


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


class OrionBixClassifier(ClassifierMixin, BaseEstimator):
    """Tabular In-Context Learning with scikit-learn interface.
    """

    def __init__(
        self,
        n_estimators: int = 32,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        class_shift: bool = True,
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        use_hierarchical: bool = True,
        use_amp: bool = True,
        batch_size: Optional[int] = 8,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "Orion-BiX-v1.1.ckpt",
        device: Optional[str | torch.device] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shift = class_shift
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.use_hierarchical = use_hierarchical
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config

    def _more_tags(self):
        """Mark classifier as non-deterministic to bypass certain sklearn tests."""
        return dict(non_deterministic=True)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _load_model(self):
        """Load a model from a given path or download it if not available.

        It uses `model_path` and `checkpoint_version` to determine the source.
         - If `model_path` is specified and exists, it's used directly.
         - If `model_path` is specified but doesn't exist (and auto-download is enabled),
           the version specified by `checkpoint_version` is downloaded to `model_path`.
         - If `model_path` is None, the version specified by `checkpoint_version` is downloaded
           from Hugging Face Hub and cached in the default Hugging Face cache directory.

        Raises
        ------
        AssertionError
            If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.

        ValueError
            If a checkpoint cannot be found or downloaded based on the settings.
        """

        repo_id = "Lexsi/Orion-BiX"
        filename = self.checkpoint_version
        ckpt = "Orion-BiX-v1.1.ckpt"

        if filename == ckpt:
            info_message = (
                f"INFO: You are downloading '{ckpt}', the version used in the original Orion-BiX paper.\n"
            )
        else:
            raise ValueError(
                f"Invalid checkpoint version '{filename}'. Available ones are: '{ckpt}'."
            )

        if self.model_path is None:
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(info_message)
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
            if model_path_:
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                
        else:
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                if self.allow_auto_download:
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                    )

        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."
        
        self.model_path_ = model_path_
        self.model_ = OrionBix(**checkpoint["config"])

        state_dict = checkpoint["state_dict"]
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace("_orig_mod.", "").replace("module.", "")
            cleaned_state_dict[cleaned_key] = value

        # Load the state dict
        self.model_.load_state_dict(cleaned_state_dict)        
        self.model_ = self.model_.float()
        self.model_.eval()
        if hasattr(self.model_, 'use_amp'):
            self.model_.use_amp = False


    def fit(self, X, y):
        """Fit the classifier to training data.

        Prepares the model for prediction by:
        1. Encoding class labels using LabelEncoder
        2. Converting input features to numerical values
        3. Fitting the ensemble generator to create transformed dataset views
        4. Loading the pre-trained model

        The model itself is not trained on the data; it uses in-context learning
        at inference time. This method only prepares the data transformations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Training target labels.

        Returns
        -------
        self : OrionBixClassifier
            Fitted classifier instance.

        Raises
        ------
        ValueError
            If the number of classes exceeds the model's maximum supported classes
            and hierarchical classification is disabled.
        """
        
        check_classification_targets(y)

        if self.device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

        # Load the pre-trained model
        self._load_model()
        self.model_.to(self.device_)

        # Inference configuration
        init_config = {
            "COL_CONFIG": {"device": self.device_, "use_amp": False, "verbose": self.verbose},
            "ROW_CONFIG": {"device": self.device_, "use_amp": False, "verbose": self.verbose},
            "ICL_CONFIG": {"device": self.device_, "use_amp": False, "verbose": self.verbose},
            #"ICL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
        }
        # If None, default settings in InferenceConfig
        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)
        # If dict, update default settings
        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)
        # If InferenceConfig, use as is
        else:
            self.inference_config_ = self.inference_config

        # Encode class labels
        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.y_encoder_.classes_)

        if self.n_classes_ > self.model_.max_classes and not self.use_hierarchical:
            raise ValueError(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Consider enabling hierarchical classification."
            )

        if self.n_classes_ > self.model_.max_classes and self.verbose:
            print(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Therefore, hierarchical classification is used."
            )

        #  Transform input features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        # Fit ensemble generator to create multiple dataset views
        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            class_shift=self.class_shift,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y)

        return self

    def _batch_forward(self, Xs, ys, shuffle_patterns=None):
        """Process model forward passes in batches to manage memory efficiently.

        This method handles the batched inference through the model,
        dividing the ensemble members into smaller batches to avoid out-of-memory errors.

        Parameters
        ----------
        Xs : np.ndarray
            Input features of shape (n_datasets, n_samples, n_features), where n_datasets
            is the number of ensemble members.

        ys : np.ndarray
            Training labels of shape (n_datasets, train_size), where train_size is the
            number of samples used for in-context learning.

        shuffle_patterns : List or None, default=None
            Lists of feature shuffle patterns to be applied to each ensemble member.
            If None, no feature shuffling is applied.

        Returns
        -------
        np.ndarray
            Model outputs (logits or probabilities) of shape (n_datasets, test_size, n_classes)
            where test_size = n_samples - train_size.
        """

        batch_size = self.batch_size or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)
        if shuffle_patterns is None:
            shuffle_patterns = [None] * n_batches
        else:
            shuffle_patterns = np.array_split(shuffle_patterns, n_batches)

        outputs = []
        for X_batch, y_batch, pattern_batch in zip(Xs, ys, shuffle_patterns):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            if pattern_batch is not None:
                pattern_batch = pattern_batch.tolist()

            with torch.no_grad():
                out = self.model_(
                    X_batch,
                    y_batch,
                    feature_shuffles=pattern_batch,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for test samples.

        Applies the ensemble of models to make predictions, with each ensemble
        member providing predictions that are then averaged. The method:
        1. Transforms input data using the fitted encoders
        2. Applies the ensemble generator to create multiple views
        3. Forwards each view through the model
        4. Corrects for class shifts
        5. Averages predictions across ensemble members

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities for each test sample.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            # Reject 1D arrays to maintain sklearn compatibility
            raise ValueError(f"The provided input X is one-dimensional. Reshape your data.")
            
        self.device_ = next(self.model_.parameters()).device

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()

            import multiprocessing as mp

            n_logical_cores = mp.cpu_count()

            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"OrionBix got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = min(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, mp.cpu_count() + 1 + self.n_jobs)

            torch.set_num_threads(n_threads)

        X = self.X_encoder_.transform(X)

        data = self.ensemble_generator_.transform(X)
        outputs = []
        for norm_method, (Xs, ys) in data.items():
            shuffle_patterns = self.ensemble_generator_.feature_shuffle_patterns_[norm_method]
            outputs.append(self._batch_forward(Xs, ys, shuffle_patterns))
        outputs = np.concatenate(outputs, axis=0)

        # Extract class shift offsets from ensemble generator
        class_shift_offsets = []
        for offsets in self.ensemble_generator_.class_shift_offsets_.values():
            class_shift_offsets.extend(offsets)

        # Determine actual number of ensemble members
        # May be fewer than requested if dataset has quite limited features and classes
        n_estimators = len(class_shift_offsets)

        # Aggregate predictions from all ensemble members, correcting for class shifts
        avg = None
        for i, offset in enumerate(class_shift_offsets):
            out = outputs[i]
            # Reverse the class shift
            out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)

            if avg is None:
                avg = out
            else:
                avg += out

        # Calculate ensemble average
        avg /= n_estimators

        # Convert logits to probabilities if required
        if self.average_logits:
            avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        # Normalize probabilities to sum to 1
        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels for test samples.

        Uses predict_proba to get class probabilities and returns the class with
        the highest probability for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels for each test sample.
        """
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)

        return self.y_encoder_.inverse_transform(y)

    @staticmethod
    def softmax(x, axis: int = -1, temperature: float = 0.9):
        """Compute softmax values with temperature scaling using NumPy.

        Parameters
        ----------
        x : ndarray
            Input array of logits.

        axis : int, default=-1
            Axis along which to compute softmax.

        temperature : float, default=0.9
            Temperature scaling parameter.

        Returns
        -------
        ndarray
            Softmax probabilities along the specified axis, with the same shape as the input.
        """
        x = x / temperature
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        # Compute softmax
        return e_x / np.sum(e_x, axis=axis, keepdims=True)