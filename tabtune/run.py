import logging
import torch
import argparse
from tabtune.benchmarking.benchmark_pipeline import BenchmarkPipeline
from tabtune.benchmarking.benchmarking_config import BENCHMARK_DATASETS
from tabtune.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TabTune Benchmark Suites.")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        default="talent", 
        choices=["talent", "openml-cc18", "tabzilla"],
        help="The name of the benchmark suite to run."
    )
    args = parser.parse_args()

    setup_logger(use_rich=True)
    logger = logging.getLogger('tabtune')
    
    MODELS_TO_BENCHMARK = {
        "TabDPT-SFT": {
            "model_name": "TabDPT",
            "tuning_strategy": "finetune",
            "finetune_mode":"sft",
            "processor_params": {'resampling_strategy': 'none'},
            "model_params": {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        },
    }

    DATA_CONFIG = {
        'talent': {'data_path': './talent_data', 'skip_download': False},
        'openml-cc18': {},
    }
    
    logger.info(f"[BenchmarkRunner] Starting Benchmark Suite: {args.benchmark.upper()}")

    benchmark = BenchmarkPipeline(
        models_to_benchmark=MODELS_TO_BENCHMARK,
        benchmark_name=args.benchmark,
        data_config=DATA_CONFIG.get(args.benchmark)
    )

    # --- MODIFIED: Simplified logic for running ---
    if args.benchmark == 'talent':
        # For 'talent', call run() without arguments to trigger auto-discovery
        benchmark.run()
    else:
        # For other benchmarks, get the list from the config file
        datasets_to_run = BENCHMARK_DATASETS[args.benchmark]
        benchmark.run(dataset_list=datasets_to_run)
    
    logger.info(f"[BenchmarkRunner] Benchmark Suite Finished: {args.benchmark.upper()}")
