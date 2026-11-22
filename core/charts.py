"""Pick a sensible default chart for a query result."""
from __future__ import annotations

import re

import pandas as pd
import plotly.express as px


def auto_chart(df: pd.DataFrame):
    if df is None or len(df) < 2:
        return None
    df = df.copy()
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and not re.search(r"(^|_)(id|key)$", str(c), re.I)]
    if not nums:
        return None
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dates:
        for c in df.columns:
            if df[c].dtype == object and re.search(r"date|day|month|week|period", str(c), re.I):
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.9:
                    df[c] = parsed
                    dates.append(c)
    cats = [c for c in df.columns if c not in nums and c not in dates]

    if dates:
        d = dates[0]
        if cats and df[cats[0]].nunique() <= 8:
            return px.line(df.sort_values(d), x=d, y=nums[0], color=cats[0], markers=len(df) < 60)
        return px.line(df.sort_values(d), x=d, y=nums[:3], markers=len(df) < 60)
    if cats:
        c = cats[0]
        top = df.groupby(c, dropna=False)[nums[0]].sum().sort_values(ascending=False).head(25).reset_index()
        top[c] = top[c].astype(str)
        return px.bar(top, x=c, y=nums[0])
    if len(nums) >= 2:
        return px.scatter(df, x=nums[0], y=nums[1])
    return None
