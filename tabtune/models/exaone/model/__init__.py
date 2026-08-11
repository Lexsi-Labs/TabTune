"""Public neural-network components used for tabular inference."""

from importlib import import_module

__all__ = [
    "ClassificationModel",
    "RegressionModel",
    "CrossAxisSummaryTransformer",
    "CrossAxisSummaryLayer",
    "TensorAttention",
    "FeedForward",
    "FeatureEncoder",
    "LabelEncoder",
    "PeakMemoryEstimate",
    "PeakMemoryMode",
    "estimated_peak",
]

_PUBLIC_BINDINGS = {
    "ClassificationModel": (".heads", "ClassificationModel"),
    "RegressionModel": (".heads", "RegressionModel"),
    "CrossAxisSummaryTransformer": (".transformer", "CrossAxisSummaryTransformer"),
    "CrossAxisSummaryLayer": (".layer", "CrossAxisSummaryLayer"),
    "TensorAttention": (".attention", "TensorAttention"),
    "FeedForward": (".mlp", "FeedForward"),
    "FeatureEncoder": (".encoders", "FeatureEncoder"),
    "LabelEncoder": (".encoders", "LabelEncoder"),
    "PeakMemoryEstimate": (".memory", "PeakMemoryEstimate"),
    "PeakMemoryMode": (".memory", "PeakMemoryMode"),
    "estimated_peak": (".memory", "estimated_peak"),
}


def __getattr__(name: str):
    """Resolve public component aliases only when they are requested."""
    try:
        module_name, attribute_name = _PUBLIC_BINDINGS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
