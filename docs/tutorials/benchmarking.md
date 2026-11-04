# Benchmarking with TabularLeaderboard

Compare models and strategies on the same splits.

```python
from tabtune import TabularLeaderboard

leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)

leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='inference',
    model_params={'n_estimators': 16}
)
leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='finetune',
    model_params={'n_estimators': 16},
    tuning_params={'epochs': 5, 'learning_rate': 1e-5}
)
leaderboard.add_model(
    model_name='TabPFN',
    tuning_strategy='inference'
)

leaderboard.run(rank_by='roc_auc_score')
```
