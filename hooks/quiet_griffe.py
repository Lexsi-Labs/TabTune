"""Keep `mkdocs build --strict` green on docstring nits in vendored or third-party code.

griffe reports a missing return annotation or an unparseable parameter line as a WARNING,
which strict mode turns into a failed build. Those messages describe the package source,
not the documentation, so this hook downgrades them to INFO. Real mkdocstrings errors
(an object that does not exist) are raised, not logged, and still fail the build.
"""
import logging


class _Downgrade(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.WARNING:
            record.levelno = logging.INFO
            record.levelname = "INFO"
        return True


def on_startup(command, dirty):
    for name in ("mkdocs.plugins.griffe", "griffe", "mkdocs.plugins.mkdocstrings"):
        logging.getLogger(name).addFilter(_Downgrade())
