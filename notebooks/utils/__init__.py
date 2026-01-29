"""
Notebook display utilities for planner and multi-agent evaluation.

Separates display/visualization code from notebook logic.
"""

from .tables import (
    # Notebook 04 — Planner evaluation
    build_token_usage_table,
    build_metrics_comparison_table,
    build_cost_analysis_table,
    build_roi_analysis_markdown,
    build_summary_table,
    # Notebook 05 — Multi-agent workflow
    build_react_eval_summary,
    build_tool_usage_table,
    build_azure_eval_summary,
    build_workflow_summary,
)
from .charts import (
    # Notebook 04
    plot_metrics_comparison,
    plot_cost_vs_performance,
    # Notebook 05
    plot_react_distributions,
)

__all__ = [
    # Tables — Notebook 04
    "build_token_usage_table",
    "build_metrics_comparison_table",
    "build_cost_analysis_table",
    "build_roi_analysis_markdown",
    "build_summary_table",
    # Tables — Notebook 05
    "build_react_eval_summary",
    "build_tool_usage_table",
    "build_azure_eval_summary",
    "build_workflow_summary",
    # Charts — Notebook 04
    "plot_metrics_comparison",
    "plot_cost_vs_performance",
    # Charts — Notebook 05
    "plot_react_distributions",
]
