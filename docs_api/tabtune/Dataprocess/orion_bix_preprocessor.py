from .tabicl_preprocessor import TabICLPreprocessor

class OrionBixPreprocessor(TabICLPreprocessor):
    """
    The preprocessing pipeline for the TabBiaxial model.
    
    This model uses the same 4-step preprocessing pipeline as TabICL,
    so this class inherits directly from TabICLPreprocessor.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        print("Fitting OrionBix Preprocessor...")
        super().fit(X, y)
        print(" OrionBix Preprocessor fitted.")
        return self