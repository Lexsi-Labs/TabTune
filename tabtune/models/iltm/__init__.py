"""
Vendored iLTM (Integrated Large Tabular Model) package for TabTune.

The full upstream engine (https://github.com/AI-sandbox/iLTM, Apache-2.0 --
see the LICENSE file in this directory) is vendored under this package:

* ``iltm_model.py`` -- the real ``iLTM`` ``torch.nn.Module`` (initial
  transformation block + meta-trained hypernetwork that *generates* an MLP
  predictor from a dataset representation);
* ``inference_interface.py`` -- the upstream sklearn-style engine
  (``iLTMClassifier`` / ``iLTMRegressor``): internal preprocessing,
  ensembling, optional retrieval and per-predictor finetuning;
* ``model_checkpoints.py`` -- Hugging Face checkpoint resolution (repo
  ``dbonet/iLTM``, cache dir via the ``ILTM_CKPT_DIR`` env var; local ``.pth``
  paths are used as-is with **no** download);
* ``tree_embedding.py`` -- OPTIONAL XGBoost/CatBoost tree embeddings (imports
  are guarded with a clear ImportError; only the tree-embedding checkpoints
  such as ``xgbrconcat`` / ``cbrconcat`` need them);
* ``realmlp_td_s_preprocessing.py`` / ``utils.py`` /
  ``hyperparameter_search_space.py`` / ``log_config.py`` -- upstream support code;
* ``engine.py`` -- thin engine subclasses that pin the backbone per wrapper
  instance (so fine-tuning / LoRA never mutates a process-wide cached module);
* ``episode_features.py`` -- numeric episode featurizer shared by the
  classification and regression wrappers for TuningManager fine-tuning;
* ``classifier.py`` -- the TabTune-contract :class:`ILTMClassifier` wrapper.

Only the lightweight wrapper is exported here; the torch-heavy engine is
imported lazily inside ``ILTMClassifier._load_model`` so ``import tabtune``
never pulls the vendored stack eagerly.
"""
from .classifier import ILTMClassifier

__all__ = ["ILTMClassifier"]
