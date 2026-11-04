# Contributing: Development Setup

## Prerequisites
- Python 3.10+
- Git

## Setup
```bash
git clone <your-fork>
cd TabTune_Internal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-mkdocs.txt
pip install -e .[dev]
```

## Running docs locally
```bash
mkdocs serve
```
