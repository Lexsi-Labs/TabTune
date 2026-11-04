"""TabAbax preprocessor for TabTune's DataProcessor.

TabAbax encodes string and categorical columns inside ``fit`` and builds its
own normalised, feature-permuted ensemble views, so features pass through
untouched.
"""

from .tabldm_preprocessor import TabLDMPreprocessor


class TabAbaxPreprocessor(TabLDMPreprocessor):
    pass
