# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.2.x | yes |
| < 0.2.0 | no |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Email **[security@lexsi.ai](mailto:security@lexsi.ai)** with:

1. a description of the problem,
2. steps to reproduce it,
3. the impact as you understand it,
4. the affected versions, if you know them,
5. a suggested fix, if you have one.

You will get an acknowledgement within three working days and a status update within
fourteen. We ask that you give us reasonable time to fix the problem before disclosing it,
and we will credit you in the release notes unless you prefer otherwise.

## Scope

TabTune downloads model checkpoints from the Hugging Face Hub and loads them with
`safetensors` or `torch.load`. Checkpoints are code-adjacent data; load only checkpoints
from sources you trust, and pin them with `model_params={"model_path": ...}` in
production. Vendored model code under `tabtune/models/` is covered by this policy; the
upstream projects it was copied from have their own.
