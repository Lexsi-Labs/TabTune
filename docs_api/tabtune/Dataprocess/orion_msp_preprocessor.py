from .tabicl_preprocessor import TabICLPreprocessor

class OrionMSPPreprocessor(TabICLPreprocessor):
    """
    The preprocessing pipeline for OrionMSP model.
    
    This model uses the same 4-step preprocessing pipeline as TabICL,
    so this class inherits directly from TabICLPreprocessor.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        print("Fitting OrionMSP Preprocessor...")
        super().fit(X, y)
        print(" OrionMSP Preprocessor fitted.")
        return self