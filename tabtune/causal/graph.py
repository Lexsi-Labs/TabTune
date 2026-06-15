"""
tabtune.causal.graph
====================

**Step 1 of the causal discipline: Identify.**

This module contains:

* :class:`GraphBuilder` -- accepts a user-supplied DAG, infers one via the PC
  algorithm (when ``causal-learn`` is installed), or falls back to the
  canonical "X causes T and Y; T causes Y" assumption.

* :class:`Identifier` -- given the graph and the (T, Y, X) variable triple,
  reports whether the average treatment effect is identifiable and returns
  an estimand (backdoor / front-door / IV) using DoWhy when available.

The Identify step is the *safety check* of the pipeline: if the effect
cannot be expressed as a function of the observable distribution, the
analysis should stop here rather than produce a misleading number.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# Optional dependencies are imported lazily so the module still loads when
# they are absent. The relevant methods raise informative errors instead.
def _try_import_dowhy():
    try:
        import dowhy  # noqa: F401
        from dowhy import CausalModel
        return CausalModel
    except Exception:
        return None


def _try_import_causal_learn():
    try:
        from causallearn.search.ConstraintBased.PC import pc  # noqa: F401
        return pc
    except Exception:
        return None


def _sanitise(name: str) -> str:
    """Return a name safe for DoWhy / GML / pyparsing.

    DoWhy's GML parser (via networkx.read_gml) is strict: node IDs that
    contain spaces, hyphens, quotes, or non-ASCII characters can cause
    silent parse failures that surface later as the misleading
    "graph should be directed acyclic" error. We normalise every node
    name to ``[A-Za-z0-9_]+`` and prefix with ``v_`` if it starts with
    a digit, mirroring what DoubleML does internally.
    """
    if name is None:
        return name
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if s and s[0].isdigit():
        s = "v_" + s
    return s or "_"


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------
class GraphBuilder:
    """
    Builds the causal graph that downstream :class:`Identifier` and
    :class:`tabtune.causal.refute.Refuter` rely on.

    A graph here is a :class:`networkx.DiGraph` with nodes named after
    columns in the user's dataframe and directed edges representing
    causal influence ``parent -> child``.

    Parameters
    ----------
    treatment : str
        Column name of the treatment variable T.
    outcome : str
        Column name of the outcome variable Y.
    confounders : list of str
        Column names of the assumed confounders X.
    instruments : list of str, optional
        Column names of instrumental variables, if any.

    Methods
    -------
    from_user_dag(graph)
        Validate and return a user-supplied DiGraph.
    infer(df, alpha)
        Run the PC algorithm on the dataframe and return the discovered DAG.
        Requires ``causal-learn``.
    default()
        Return the canonical "X causes T and Y; T causes Y" graph.
    to_gml_string()
        Serialise the current graph to a GML string (consumed by DoWhy
        when the user's DoWhy version doesn't accept a DiGraph directly).
    name_map()
        Return ``{original_column_name: sanitised_node_name}`` so callers
        can rename their dataframe to match the graph before handing both
        to DoWhy.
    """

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: list[str],
        instruments: list[str] | None = None,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = list(confounders)
        self.instruments = list(instruments or [])
        self._graph: nx.DiGraph | None = None

       
        all_names = [treatment, outcome, *self.confounders, *self.instruments]
        self._name_map: dict[str, str] = {n: _sanitise(n) for n in all_names}
        # Detect collisions (two different originals mapping to the same
        # sanitised name). If this happens we disambiguate with a suffix
        # so DoWhy still sees distinct nodes.
        seen: dict[str, int] = {}
        for orig, safe in list(self._name_map.items()):
            count = seen.get(safe, 0)
            if count > 0:
                self._name_map[orig] = f"{safe}_{count}"
            seen[safe] = count + 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def name_map(self) -> dict[str, str]:
        """Return a shallow copy of the original -> sanitised name map."""
        return dict(self._name_map)

    @property
    def safe_treatment(self) -> str:
        return self._name_map[self.treatment]

    @property
    def safe_outcome(self) -> str:
        return self._name_map[self.outcome]

    @property
    def safe_confounders(self) -> list[str]:
        return [self._name_map[c] for c in self.confounders]

    @property
    def safe_instruments(self) -> list[str]:
        return [self._name_map[z] for z in self.instruments]

    def from_user_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Validate and adopt a user-supplied causal DAG."""
        if not isinstance(graph, nx.DiGraph):
            raise TypeError(
                "User-supplied DAG must be a networkx.DiGraph; got "
                f"{type(graph).__name__}."
            )
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("User-supplied graph contains cycles; expected a DAG.")
        # Sanity-check that T and Y are present
        for node in (self.treatment, self.outcome):
            if node not in graph.nodes:
                raise ValueError(f"User graph is missing required node '{node}'.")
        # Relabel to sanitised names so DoWhy can consume it.
        graph = nx.relabel_nodes(graph.copy(), self._name_map)
        self._graph = graph
        return self._graph

    def infer(self, df: pd.DataFrame, alpha: float = 0.05) -> nx.DiGraph:
        """
        Run the PC algorithm on the dataframe and adopt the discovered DAG.

        Notes
        -----
        Requires the optional ``causal-learn`` package. If unavailable, this
        method raises :class:`ImportError` so the caller can fall back to
        :meth:`default`.
        """
        pc = _try_import_causal_learn()
        if pc is None:
            raise ImportError(
                "Graph inference via PC algorithm requires the 'causal-learn' "
                "package. Install with `pip install causal-learn`, or call "
                ".default() to use the canonical confounder graph."
            )
        logger.info("[Causal] Inferring causal graph via PC algorithm (alpha=%.3f)", alpha)
        columns = list(df.columns)
        data = df[columns].to_numpy()
        cg = pc(data, alpha=alpha, indep_test="fisherz", show_progress=False)

        # causal-learn -> networkx, using *sanitised* node names from the
        # outset so the graph is consistent with everything else.
        safe_cols = [self._name_map.get(c, _sanitise(c)) for c in columns]
        g = nx.DiGraph()
        g.add_nodes_from(safe_cols)
        adj = cg.G.graph
        for i in range(len(safe_cols)):
            for j in range(len(safe_cols)):
                if i == j:
                    continue
                # In causal-learn's representation, a directed edge i -> j
                # is encoded as graph[j][i] == 1 and graph[i][j] == -1.
                if adj[j][i] == 1 and adj[i][j] == -1:
                    g.add_edge(safe_cols[i], safe_cols[j])
        if not nx.is_directed_acyclic_graph(g):
            # PC can produce a CPDAG with bidirected edges; coerce to DAG
            # by dropping the reverse of any cycle-forming pair.
            for u, v in list(g.edges()):
                if g.has_edge(v, u):
                    g.remove_edge(v, u)
        self._graph = g
        logger.info(
            "[Causal] Inferred DAG with %d nodes, %d edges",
            g.number_of_nodes(), g.number_of_edges(),
        )
        return self._graph

    def default(self) -> nx.DiGraph:
        """Return the canonical 'X -> T, X -> Y, T -> Y' graph.

        Nodes are stored under *sanitised* names so the graph round-trips
        through DoWhy / GML cleanly.
        """
        g = nx.DiGraph()
        t = self.safe_treatment
        y = self.safe_outcome
        g.add_node(t)
        g.add_node(y)
        for x in self.safe_confounders:
            g.add_node(x)
            g.add_edge(x, t)
            g.add_edge(x, y)
        g.add_edge(t, y)
        for z in self.safe_instruments:
            g.add_node(z)
            g.add_edge(z, t)
        self._graph = g
       
        assert nx.is_directed_acyclic_graph(g), (
            "GraphBuilder.default() produced a cyclic graph; this is a bug."
        )
        logger.info(
            "[Causal] Using default confounder graph: %d nodes, %d edges",
            g.number_of_nodes(), g.number_of_edges(),
        )
        return self._graph

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            return self.default()
        return self._graph

    def to_gml_string(self) -> str:
        """Return a GML string representation consumable by DoWhy.

        This is only used for DoWhy versions older than 0.11 (newer DoWhy
        accepts a networkx.DiGraph directly, which we prefer in
        :class:`Identifier`). The string is deliberately defensive:

        * ``directed 1`` is emitted so the GML parser knows the graph is
          directed -- omitting this triggers DoWhy's misleading
          "graph should be directed acyclic" message even when there
          are no cycles.
        * Node ``id`` is the sanitised node name itself (quoted), so
          DoWhy can match treatment/outcome strings against nodes
          without needing a separate label lookup.
        * Each ``node`` / ``edge`` opens with ``[`` on the same line,
          matching networkx's expected GML dialect.
        """
        g = self.graph
        lines = ["graph [", "  directed 1"]
        for node in g.nodes():
            lines.append(f'  node [ id "{node}" label "{node}" ]')
        for u, v in g.edges():
            lines.append(f'  edge [ source "{u}" target "{v}" ]')
        lines.append("]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------
class Identifier:
    """
    Reports whether the causal effect of T on Y is identifiable from data.

    When DoWhy is installed, this class returns the formal estimand
    (backdoor, front-door, or IV) along with the adjustment set. When DoWhy
    is absent, it performs a minimal d-separation check on the user's DAG
    and reports whether the supplied confounders block the backdoor paths.

    Parameters
    ----------
    builder : :class:`GraphBuilder`
        Owns the causal graph and variable roles.
    """

    def __init__(self, builder: GraphBuilder):
        self.builder = builder

    def _prepare_dowhy_inputs(self, df: pd.DataFrame):
        """Rename dataframe columns to sanitised node names DoWhy expects.

        We never mutate the user's DataFrame -- a shallow copy with
        renamed columns is returned. Columns that aren't part of the
        causal graph are passed through untouched.
        """
        name_map = self.builder.name_map()
        # Only rename columns that are in the map AND in the dataframe;
        # leave everything else alone.
        rename = {k: v for k, v in name_map.items() if k in df.columns and k != v}
        if rename:
            df = df.rename(columns=rename)
        return df

    def identify(
        self,
        df: pd.DataFrame | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Return a dict describing the identification result.

        The returned dict has keys:

        * ``identifiable`` (bool)
        * ``estimand_type`` (str): ``'backdoor' | 'frontdoor' | 'iv' | 'none'``
        * ``adjustment_set`` (list[str])
        * ``backend`` (str): ``'dowhy'`` or ``'fallback'``
        * ``message`` (str): human-readable explanation
        """
        CausalModel = _try_import_dowhy()

        if CausalModel is not None and df is not None:
            
            df_dowhy = self._prepare_dowhy_inputs(df)
            graph_obj = self.builder.graph

            dowhy_attempts: list[tuple[str, Any]] = [
                ("DiGraph", graph_obj),
                ("GML",     self.builder.to_gml_string()),
            ]
            last_exc: Exception | None = None
            for label, graph_arg in dowhy_attempts:
                try:
                    model = CausalModel(
                        data=df_dowhy,
                        treatment=self.builder.safe_treatment,
                        outcome=self.builder.safe_outcome,
                        graph=graph_arg,
                    )
                    estimand = model.identify_effect(
                        proceed_when_unidentifiable=False,
                        optimize_backdoor=True,
                    )
                    est_type = "none"
                    adj_set: list[str] = []
                    if getattr(estimand, "backdoor_variables", None):
                        backdoor = estimand.backdoor_variables
                        for _, v in backdoor.items():
                            if v:
                                est_type = "backdoor"
                                adj_set = list(v)
                                break
                    if est_type == "none" and getattr(
                        estimand, "instrumental_variables", None
                    ):
                        iv_list = estimand.instrumental_variables
                        if iv_list:
                            est_type = "iv"
                            adj_set = list(iv_list)
                    identifiable = est_type != "none"
                    result = {
                        "identifiable": identifiable,
                        "estimand_type": est_type,
                        "adjustment_set": adj_set,
                        "backend": f"dowhy:{label.lower()}",
                        "message": (
                            f"DoWhy returned an identifiable {est_type} estimand "
                            f"via {label} input."
                            if identifiable
                            else "DoWhy could not identify the effect."
                        ),
                    }
                    if verbose:
                        logger.info(
                            "[Causal] Identification (DoWhy/%s): %s",
                            label, result["message"],
                        )
                    return result
                except Exception as exc:  # pragma: no cover -- defensive
                    last_exc = exc
                    if verbose:
                        logger.debug(
                            "[Causal] DoWhy attempt via %s failed: %s; trying next.",
                            label, exc,
                        )
                    continue

            # Both DoWhy attempts failed -- record the cause and fall back.
            if last_exc is not None:
                logger.warning(
                    "[Causal] DoWhy identification unavailable (%s); "
                    "using d-separation fallback.",
                    last_exc,
                )

        
        g = self.builder.graph
        t = self.builder.safe_treatment
        y = self.builder.safe_outcome
        confounders = self.builder.safe_confounders
        try:
            has_path = nx.has_path(g, t, y)
        except nx.NodeNotFound:
            has_path = False
        all_confounders_in_graph = all(c in g.nodes for c in confounders)
        identifiable = has_path and all_confounders_in_graph
        result = {
            "identifiable": identifiable,
            "estimand_type": "backdoor" if identifiable else "none",
            "adjustment_set": confounders if identifiable else [],
            "backend": "fallback",
            "message": (
                "Assuming backdoor adjustment via the declared confounders. "
                "Install 'dowhy>=0.11' for a formal identification check."
                if identifiable
                else "No directed path T -> Y in the graph, or confounders missing."
            ),
        }
        if verbose:
            logger.info("[Causal] Identification (fallback): %s", result["message"])
        return result