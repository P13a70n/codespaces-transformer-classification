from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px


def make_intent_distribution_chart(label_counts: pd.DataFrame) -> Any:
    chart_df = label_counts.head(12).copy()
    return px.bar(
        chart_df,
        x="intent",
        y="count",
        color="count",
        title="Top banking intents in the dataset",
        text_auto=True,
        color_continuous_scale="Viridis",
    )


def make_query_length_chart(lengths: List[int]) -> Any:
    df = pd.DataFrame({"query_length": lengths})
    return px.histogram(df, x="query_length", nbins=25, title="Customer message length distribution")


def make_confusion_matrix_chart(confusion_df: pd.DataFrame) -> Any:
    return px.imshow(
        confusion_df.values,
        labels=dict(x="Predicted intent", y="Actual intent", color="Count"),
        x=confusion_df.columns,
        y=confusion_df.index,
        color_continuous_scale="Blues",
        title="Confusion matrix",
    )


def make_prediction_chart(predictions: List[Dict[str, Any]]) -> Any:
    chart_df = pd.DataFrame(predictions)
    if chart_df.empty:
        return None
    return px.bar(
        chart_df,
        x="intent",
        y="confidence_pct",
        color="confidence_pct",
        title="Top predicted intents",
        text_auto=True,
        color_continuous_scale="Cividis",
    )
