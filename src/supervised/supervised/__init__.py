"""Supervised learning utilities for tabular pandas workflows."""

from supervised.describe import (  # noqa: F401
    association_with_target,
    correlation_ratio,
    infer_variable_types,
    missingness_profile,
    plot_distributions,
    plot_missingness,
    plot_target_relationships,
)

__all__ = [
    "infer_variable_types",
    "correlation_ratio",
    "association_with_target",
    "missingness_profile",
    "plot_missingness",
    "plot_distributions",
    "plot_target_relationships",
]
