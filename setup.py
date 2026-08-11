"""Compatibility shim for tools that still invoke ``setup.py`` directly.

All packaging metadata lives in ``pyproject.toml``. This file previously
duplicated it - and drifted: it declared version 0.1.17 while pyproject
declared 0.1.18 and ``tabtune/__init__.py`` fell back to a third value. Keeping
one source of truth removes that failure mode entirely.
"""

from setuptools import setup

setup()
