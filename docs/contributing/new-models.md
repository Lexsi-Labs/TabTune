# Contributing: Adding New Models

## Steps
1. Implement model under `tabtune/models/<model_name>/`
2. Add corresponding preprocessor in `tabtune/Dataprocess/`
3. Register model in `TabularPipeline` selection logic
4. Document parameters and defaults
5. Add example usage in docs/examples

## Checklist
- Inference path works on a small dataset
- Optional: fine-tuning via `TuningManager`
- Docs updated
