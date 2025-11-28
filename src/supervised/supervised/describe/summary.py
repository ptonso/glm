from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def infer_variable_types(df: pd.DataFrame, target: str, cat_threshold: int = 10) -> Dict[str, list[str]]:
    """Split predictors into categorical and numerical buckets based on dtype and unique counts."""
    features = [c for c in df.columns if c != target]
    categorical: list[str] = []
    numerical: list[str] = []
    for col in features:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or series.dtype.name in ("category", "object"):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(series) and series.nunique() <= cat_threshold:
            categorical.append(col)
        else:
            numerical.append(col)
    return {"categorical": categorical, "numerical": numerical}


def correlation_ratio(y: pd.Series, x: pd.Series) -> float:
    """Compute eta-squared for numeric target against categorical predictor."""
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if data.empty:
        return float("nan")
    mean_total = data["y"].mean()
    ss_total = ((data["y"] - mean_total) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - mean_total) ** 2 for _, g in data.groupby("x")["y"])
    return float(ss_between / ss_total) if ss_total else float("nan")


def association_with_target(
    df: pd.DataFrame, target: str, types: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Quantify monotonic associations of predictors with the target."""
    rows: list[dict[str, object]] = []
    for col in types.get("numerical", []):
        clean = df[[col, target]].dropna()
        corr = clean[target].rank().corr(clean[col].rank()) if len(clean) else np.nan
        rows.append(
            {
                "variavel": col,
                "tipo": "numerica",
                "metrica": "spearman",
                "associacao": float(corr) if pd.notna(corr) else np.nan,
            }
        )
    for col in types.get("categorical", []):
        clean = df[[col, target]].dropna()
        eta = correlation_ratio(clean[target], clean[col]) if clean[col].nunique() > 1 else np.nan
        rows.append({"variavel": col, "tipo": "categorica", "metrica": "eta2", "associacao": eta})
    return pd.DataFrame(rows)


def missingness_profile(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Summarize missing values and optional rank-correlation with the target."""
    miss = df.isna().sum().to_frame("faltantes")
    miss["pct_faltantes"] = miss["faltantes"] / len(df)
    if target:
        miss["corr_target"] = [
            df[target].rank().corr(df[col].isna().astype(int).rank()) if miss.loc[col, "faltantes"] else np.nan
            for col in miss.index
        ]
    return miss.reset_index().rename(columns={"index": "variavel"})
