# docs_api

A source-only snapshot of the `tabtune` package (0.2.2) that the API Reference pages read
their docstrings from. mkdocstrings is pointed here (`paths: [docs_api]` in `mkdocs.yml`)
so `mkdocs build` and `mkdocs gh-deploy` work on any branch, including the `docs` branch,
whose own `tabtune/` copy is older. Nothing imports it; the vendored model packages under
`tabtune/models/` are left out because no reference page documents them.

Refresh it after a release, from a checkout of `main`:

    rm -rf docs_api/tabtune
    (cd /path/to/main && find tabtune -name "*.py" -not -path "tabtune/models/*" -o -path tabtune/models/__init__.py \
      | tar cf - -T -) | tar xf - -C docs_api
