"""
Example 9: Running Comprehensive Benchmarks
===========================================

This example demonstrates TabTune's BenchmarkPipeline for running comprehensive
benchmarks across multiple datasets and model configurations.

Key Learning Points:
- Use BenchmarkPipeline for large-scale evaluations
- Configure benchmark suites (talent, openml-cc18)
- Run model comparisons across multiple datasets
- Systematic evaluation workflow

Dataset: Multiple datasets from benchmark suites
Note: Demonstrates benchmark infrastructure usage
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import logging
import argparse

# Import TabTune benchmarking components
from tabtune.benchmarking.benchmark_pipeline import BenchmarkPipeline
from tabtune.benchmarking.benchmarking_config import BENCHMARK_DATASETS
from tabtune.logger import setup_logger

# ============================================================================
# SETUP: Logging
# ============================================================================

setup_logger(use_rich=True)
logger = logging.getLogger('tabtune')

# ============================================================================
# MAIN BENCHMARKING DEMONSTRATION
# ============================================================================

logger.info("="*80)
logger.info("EXAMPLE 9: Running Comprehensive Benchmarks")
logger.info("="*80)
logger.info("\n📊 This example demonstrates TabTune's benchmarking infrastructure")
logger.info("   for systematic evaluation across multiple datasets\n")

# ============================================================================
# DEMONSTRATION 1: Single Dataset Benchmark
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 1: Benchmarking on Single Dataset")
logger.info("="*80)
logger.info("\n📝 Benchmarking allows you to:")
logger.info("   - Run systematic evaluations")
logger.info("   - Compare models across datasets")
logger.info("   - Track results automatically\n")

# Configure models to benchmark
MODELS_TO_BENCHMARK = {
    "OrionMSP-Inference": {
        "model_name": "OrionMSP",
        "tuning_strategy": "inference",
        "tuning_params": {'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        "processor_params": {'resampling_strategy': 'none'}
    },
}

logger.info("🔧 Model Configurations to Benchmark:")
for model_key, config in MODELS_TO_BENCHMARK.items():
    logger.info(f"   - {model_key}: {config.get('tuning_strategy', 'N/A')} strategy")

# ============================================================================
# DEMONSTRATION 2: Using BenchmarkPipeline
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 2: Using BenchmarkPipeline")
logger.info("="*80)
logger.info("\n📝 BenchmarkPipeline provides:")
logger.info("   - Automated dataset loading")
logger.info("   - Model training and evaluation")
logger.info("   - Results collection and reporting")
logger.info("   - Progress tracking\n")

logger.info("💡 Note: Full benchmarking requires dataset setup")
logger.info("   This example shows the configuration structure")
logger.info("   For actual benchmarking, ensure datasets are available\n")

# Example benchmark configuration
# Note: This is a demonstration - actual execution may require dataset preparation
DATA_CONFIG = {
    'talent': {'data_path': './talent_data', 'skip_download': False},
    'openml-cc18': {},  # Use default OpenML-CC18 configuration
}

logger.info("📋 Benchmark Configuration:")
logger.info("   Models: Multiple configurations defined")
logger.info("   Datasets: Can use talent or openml-cc18 suites")
logger.info("   Results: Automatically collected and ranked")

# ============================================================================
# DEMONSTRATION 3: Running a Benchmark (Conceptual)
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 3: Running Benchmark (Conceptual Example)")
logger.info("="*80)
logger.info("\n📝 To run a full benchmark:")

logger.info("\n   1️⃣  Initialize BenchmarkPipeline:")
logger.info("      benchmark = BenchmarkPipeline(")
logger.info("          models_to_benchmark=MODELS_TO_BENCHMARK,")
logger.info("          benchmark_name='your_benchmark',")
logger.info("          data_config=DATA_CONFIG")
logger.info("      )")

logger.info("\n   2️⃣  Run the benchmark:")
logger.info("      benchmark.run()")

logger.info("\n   3️⃣  Results are automatically:")
logger.info("      - Collected per dataset and model")
logger.info("      - Ranked by specified metrics")
logger.info("      - Saved for analysis")

# ============================================================================
# ACTUAL BENCHMARK EXECUTION: Run a small benchmark demonstration
# ============================================================================

logger.info("\n" + "="*80)
logger.info("RUNNING ACTUAL BENCHMARK: Small Demo")
logger.info("="*80)
logger.info("\n💡 Running a small benchmark with 1-2 datasets for demonstration...")

try:
    # Initialize benchmark with openml-cc18 - run on multiple datasets for proper benchmark
    from tabtune.benchmarking.benchmarking_config import BENCHMARK_DATASETS
    
    # Pick 2-3 small datasets for a proper benchmark demonstration
    demo_datasets = None
    if 'openml-cc18' in BENCHMARK_DATASETS and len(BENCHMARK_DATASETS['openml-cc18']) > 0:
        # Use first 3 datasets for a proper benchmark demo
        demo_datasets = BENCHMARK_DATASETS['openml-cc18'][:3]
        logger.info(f"   Running benchmark on {len(demo_datasets)} datasets:")
        for ds_id in demo_datasets:
            logger.info(f"      - OpenML ID {ds_id}")
    else:
        # If config doesn't have datasets, use well-known small OpenML datasets
        demo_datasets = [3, 31, 1489]  # kr-vs-kp, German Credit, Phoneme
        logger.info(f"   Running benchmark on {len(demo_datasets)} datasets:")
        logger.info(f"      - OpenML ID 3 (kr-vs-kp)")
        logger.info(f"      - OpenML ID 31 (German Credit)")
        logger.info(f"      - OpenML ID 1489 (Phoneme)")
    
    logger.info(f"\n   Model: OrionMSP-Inference")
    logger.info("   This will take several minutes...\n")
    
    # Run actual benchmark
    benchmark = BenchmarkPipeline(
        models_to_benchmark=MODELS_TO_BENCHMARK,
        benchmark_name='openml-cc18',
        data_config=DATA_CONFIG['openml-cc18']
    )
    
    # Run on multiple datasets for a proper benchmark
    benchmark.run(dataset_list=demo_datasets, test_size=0.25)
    
    logger.info("\n✅ Benchmark execution completed!")
    logger.info("   Results saved to: benchmark_results_openml-cc18_OrionMSP-Inference.csv")
    logger.info("   Check the CSV file for detailed results across all datasets")
    
except Exception as e:
    logger.warning(f"⚠️  Could not run actual benchmark: {e}")
    logger.info("   This is expected if:")
    logger.info("   - Datasets are not available/downloaded")
    logger.info("   - OpenML API is not accessible")
    logger.info("   - Or for demonstration purposes")
    logger.info("\n   The example code above shows the correct usage pattern.")

# ============================================================================
# DEMONSTRATION 4: Benchmark Suites
# ============================================================================

logger.info("\n" + "="*80)
logger.info("DEMONSTRATION 4: Available Benchmark Suites")
logger.info("="*80)

logger.info("\n📊 TabTune supports multiple benchmark suites:\n")

logger.info("   1. Talent Benchmark Suite:")
logger.info("      - Multiple tabular datasets")
logger.info("      - Various domains and sizes")
logger.info("      - Good for comprehensive evaluation")

logger.info("\n   2. OpenML-CC18:")
logger.info("      - 72 classification datasets")
logger.info("      - Standard ML benchmark suite")
logger.info("      - Widely used in research")

logger.info("\n💡 Choose benchmark suite based on:")
logger.info("   - Your evaluation needs")
logger.info("   - Dataset availability")
logger.info("   - Computational resources")
logger.info("   - Domain requirements")

# ============================================================================
# EXAMPLE: Minimal Benchmark Execution Pattern
# ============================================================================

logger.info("\n" + "="*80)
logger.info("EXAMPLE: Minimal Benchmark Execution Pattern")
logger.info("="*80)

logger.info("\n📝 Complete code pattern for running benchmarks:\n")

example_code = """
# 1. Define models to benchmark
MODELS_TO_BENCHMARK = {
    "Model-Name": {
        "model_name": "TabICL",
        "tuning_strategy": "inference",
        "processor_params": {'resampling_strategy': 'none'}
    },
    # Add more model configurations...
}

# 2. Configure data source
DATA_CONFIG = {
    'openml-cc18': {},  # Use OpenML-CC18 suite
    # or
    'talent': {'data_path': './talent_data'}
}

# 3. Initialize and run benchmark
benchmark = BenchmarkPipeline(
    models_to_benchmark=MODELS_TO_BENCHMARK,
    benchmark_name='openml-cc18',
    data_config=DATA_CONFIG['openml-cc18']
)

# 4. Run on specific datasets or all datasets
benchmark.run(dataset_list=['dataset1', 'dataset2'])
# or
# benchmark.run()  # Run on all datasets in suite
"""

logger.info(example_code)

# ============================================================================
# SUMMARY: Benchmarking Benefits
# ============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY: Benchmarking Benefits")
logger.info("="*80)
logger.info("\n✨ Key Takeaways:")
logger.info("   1. Systematic Evaluation: Consistent evaluation across datasets")
logger.info("   2. Model Comparison: Fair comparison with same methodology")
logger.info("   3. Reproducibility: Standardized benchmark protocols")
logger.info("   4. Scalability: Evaluate multiple models/datasets efficiently")
logger.info("   5. Results Tracking: Automatic collection and reporting")

logger.info("\n💡 Use Cases:")
logger.info("   - Research: Compare new methods against baselines")
logger.info("   - Model Selection: Find best model for your domain")
logger.info("   - Hyperparameter Tuning: Systematic parameter search")
logger.info("   - Publication: Standardized evaluation for papers")
logger.info("   - Production: Validate model across diverse scenarios")

logger.info("\n📋 Best Practices:")
logger.info("   - Use standard benchmark suites for reproducibility")
logger.info("   - Document configuration clearly")
logger.info("   - Track results systematically")
logger.info("   - Report all relevant metrics")
logger.info("   - Consider computational constraints")

logger.info("\n" + "="*80)
logger.info("✅ Example 9 Complete: Benchmarking Demonstration")
logger.info("="*80)
logger.info("\n💡 Note: This example shows the structure and usage pattern.")
logger.info("   For actual benchmarking execution, ensure:")
logger.info("   - Datasets are prepared/downloaded")
logger.info("   - Sufficient computational resources")
logger.info("   - Appropriate model configurations")
logger.info("="*80)
