import logging
import torch

from tabtune.benchmarking.benchmark_pipeline import BenchmarkPipeline
from tabtune.benchmarking.benchmarking_config import BENCHMARK_DATASETS
from tabtune.logger import setup_logger


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

setup_logger(use_rich=True)
logger = logging.getLogger("tabtune")

logger.info("=" * 80)
logger.info("Running Mitra on Grinsztajn Tabular Benchmark (Regression)")
logger.info("=" * 80)


# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

MODELS_TO_BENCHMARK = {
    "Mitra-Inference": {
        "model_name": "Mitra",
        "tuning_strategy": "inference",
        "tuning_params": {
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "processor_params": {
            "resampling_strategy": "none"
        }
    }
}

logger.info("Model configuration:")
for model in MODELS_TO_BENCHMARK:
    logger.info(f"   - {model}")


# ---------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------

benchmark_name = "openml-ctr23"

datasets = BENCHMARK_DATASETS[benchmark_name]

logger.info(f"\nTotal datasets in benchmark: {len(datasets)}")


# ---------------------------------------------------------
# OPTIONAL: quick test first
# ---------------------------------------------------------

# comment this line to run full benchmark
datasets = datasets[:3]

logger.info(f"Running on {len(datasets)} datasets for test:")
for d in datasets:
    logger.info(f"   OpenML ID {d}")


# ---------------------------------------------------------
# Data Config
# ---------------------------------------------------------

DATA_CONFIG = {}


# ---------------------------------------------------------
# Initialize BenchmarkPipeline
# ---------------------------------------------------------

benchmark = BenchmarkPipeline(
    models_to_benchmark=MODELS_TO_BENCHMARK,
    benchmark_name=benchmark_name,
    data_config=DATA_CONFIG
)


# ---------------------------------------------------------
# Run Benchmark
# ---------------------------------------------------------

benchmark.run(
    dataset_list=datasets,
    test_size=0.25
)


logger.info("\nBenchmark completed successfully!")

logger.info(
    f"Results saved to: benchmark_results_{benchmark_name}_Mitra-Inference.csv"
)