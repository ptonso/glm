from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_missingness(missing_df: pd.DataFrame) -> None:
    """Bar plot of missing-value proportions."""
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(missing_df) + 1))
    sns.barplot(data=missing_df, x="pct_faltantes", y="variavel", ax=ax, color="#7AA6DC")
    ax.set_xlabel("Missing values proportions")
    ax.set_ylabel("")
    ax.set_xlim(0, missing_df["pct_faltantes"].max() * 1.05 + 0.01)
    plt.show()


def plot_distributions(df: pd.DataFrame, target: str, types: Mapping[str, Sequence[str]]) -> None:
    """Histograms for numeric variables and counts for categoricals."""
    num_cols = [target] + list(types.get("numerical", []))
    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(4.5 * len(num_cols), 4), constrained_layout=True)
        if len(num_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, num_cols):
            sns.histplot(df[col], kde=True, ax=ax, color="#5E81AC")
            ax.set_title(col)
    cat_cols = list(types.get("categorical", []))
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(4 * len(cat_cols), 4), constrained_layout=True)
        if len(cat_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cat_cols):
            sns.countplot(data=df, x=col, hue=col, palette="Set2", legend=False, ax=ax)
            ax.set_title(col)
    plt.show()


def plot_target_relationships(df: pd.DataFrame, target: str, types: Mapping[str, Sequence[str]]) -> None:
    """Pair target with predictors using regplots for numerics and boxplots for categoricals."""
    num_cols = list(types.get("numerical", []))
    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4), constrained_layout=True)
        if len(num_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, num_cols):
            sns.regplot(
                data=df,
                x=col,
                y=target,
                scatter_kws={"alpha": 0.4},
                line_kws={"color": "black"},
                ax=ax,
            )
            ax.set_title(f"{target} ~ {col}")
    cat_cols = list(types.get("categorical", []))
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(4.5 * len(cat_cols), 4), constrained_layout=True)
        if len(cat_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cat_cols):
            sns.boxplot(data=df, x=col, y=target, hue=col, palette="Set2", legend=False, ax=ax)
            ax.set_title(f"{target} por {col}")
    plt.show()
