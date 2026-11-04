"""Leaderboard: compare models and strategies on a shared split.

Example:
    >>> from tabtune import TabularLeaderboard
    >>> board = TabularLeaderboard(X_train, X_test, y_train, y_test)  # doctest: +SKIP
    >>> board.add_models(["TabICLv2", "OrionMSP", "Mitra"])           # doctest: +SKIP
    >>> board.run()                                                   # doctest: +SKIP
    >>> board.to_html("leaderboard.html")                             # doctest: +SKIP
"""

from .leaderboard import LeaderboardEntry, TabularLeaderboard
from .report import render_leaderboard_html, render_pareto_svg

__all__ = [
    "TabularLeaderboard",
    "LeaderboardEntry",
    "render_leaderboard_html",
    "render_pareto_svg",
]
