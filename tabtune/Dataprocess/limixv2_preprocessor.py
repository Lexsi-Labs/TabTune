"""LimiX-2 preprocessor for TabTune's DataProcessor.

LimiX-2 encodes categoricals, handles missing values and datetimes, and
normalises features per ensemble member itself, so features pass through
untouched. Classification targets are label-encoded so ``classes_`` is
contiguous; regression targets stay as floats.
"""

from .tabldm_preprocessor import TabLDMPreprocessor


class LimiXV2Preprocessor(TabLDMPreprocessor):
    pass
