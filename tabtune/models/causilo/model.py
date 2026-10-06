"""Checkpoint-defined network components; ModelRunner defines their execution order."""

from dataclasses import asdict, dataclass

from torch import nn

from .nn.column import ColumnBlock
from .nn.embeddings import FeatureEmbedding, TargetEmbedding
from .nn.prediction import Head, PredictionBlock
from .nn.row import RowBlock, RowPool


@dataclass(frozen=True)
class ModelConfig:
    """Architecture dimensions loaded from the official checkpoint configuration.

    The two column depths describe stages before and after row mixing; the two
    row depths describe row mixing and pooling. Prediction width is
    ``width * row_latents``. ``outputs`` is the class capacity or the number of
    regression output channels averaged by Engine.
    """

    task: str
    width: int
    expansion: int
    group_size: int
    frequencies: int
    column_latents: int
    column_heads: int
    column_depths: tuple[int, int]
    row_heads: int
    row_latents: int
    row_depths: tuple[int, int]
    prediction_heads: int
    prediction_depth: int
    outputs: int

    def record(self) -> dict:
        """Produce the versioned configuration stored with weights and fitted state."""
        return {
            "architecture": "causilo-v1.0",
            "format_version": 1,
            "config_version": 1,
            "model": asdict(self),
        }


class Model(nn.Module):
    """Register pretrained modules without coupling them to an execution policy.

    Some latent/embedding parameters use empty initialization: load a checkpoint
    before executing this model. There is no training loop or forward method;
    ModelRunner handles direct and cached evaluation of the registered blocks.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width, expansion = config.width, config.expansion
        classes = config.outputs if config.task == "classification" else 0
        self.feature_embedding = FeatureEmbedding(width, config.group_size, config.frequencies)
        self.feature_target = TargetEmbedding(width, classes)
        self.row_target = TargetEmbedding(width * config.row_latents, classes)
        self.columns = nn.ModuleList(
            ColumnBlock(width, config.column_heads, expansion, config.column_latents, depth)
            for depth in config.column_depths
        )
        self.row = RowBlock(width, config.row_heads, expansion, config.row_latents, config.row_depths[0])
        self.pool = RowPool(width, config.row_heads, expansion, config.row_latents, config.row_depths[1])
        self.prediction = PredictionBlock(
            width * config.row_latents, config.prediction_heads, expansion, config.prediction_depth
        )
        self.head = Head(width * config.row_latents, config.outputs)
