"""
Vendored xRFM (Recursive Feature Machines) package for TabTune.

The full upstream engine (https://github.com/dmbeaglehole/xRFM, MIT licensed --
see the LICENSE file in this directory) is vendored under this package:

* ``xrfm.py`` / ``tree_utils.py`` -- the tree-splittable ``xRFM`` estimator;
* ``rfm_src/`` -- the core ``RFM`` algorithm (kernels, AGOP, EigenPro, ...);
* ``classifier.py`` -- the TabTune-contract :class:`XRFMClassifier` wrapper;
* ``preprocessing.py`` -- numeric feature encoding shared by the classification
  and regression wrappers.

Only the lightweight wrapper is exported here; the torch-heavy engine is
imported lazily inside ``XRFMClassifier._load_model`` so ``import tabtune``
never needs the optional stack eagerly.
"""
from .classifier import XRFMClassifier

__all__ = ["XRFMClassifier"]
