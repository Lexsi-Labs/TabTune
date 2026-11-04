# Logging

Every pipeline, leaderboard and benchmark reports through one logger, `tabtune`. You
configure it once; nothing else in the library touches handlers.

```python
from tabtune import setup_logger

setup_logger()                                   # Rich console on stderr, INFO
setup_logger(level="DEBUG", log_file="runs/tabtune.log")
setup_logger(json_format=True)                   # JSON on console and file
setup_logger(console_format="plain", use_color=False)
```

## What you get

- Console output is Rich when a terminal is attached and plain otherwise.
  `console_format` forces `auto`, `rich`, `plain` or `json`.
- `log_file` adds a rotating UTF-8 file handler (`max_bytes=5 MB`, `backup_count=5`);
  `file_format` is `plain` or `json`.
- `console_level` and `file_level` set the two sinks separately.
- Under distributed training only rank zero logs by default (`rank_zero_only=True`). Use
  one log file per process if you turn that off.
- `show_banner=False` suppresses the product banner.
- Handlers that other libraries or your application installed are left alone; `setup_logger`
  only manages the ones it owns, and calling it twice replaces its own handlers rather than
  stacking them.

## Reading the output

Fine-tuning runs log the epoch, the training loss and, for the validated loop, the
validation score and whether it beat the baseline. Leaderboards log each entry's fit and
predict time. The registry logs envelope warnings at `WARNING`, so a quiet run means your
data was inside every documented limit.

```python
import logging
logging.getLogger("tabtune").setLevel(logging.WARNING)   # only problems
```

`examples/logging_showcase.py` prints every format side by side.
