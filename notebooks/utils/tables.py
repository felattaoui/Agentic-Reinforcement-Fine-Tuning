"""
Markdown table builders for notebook display.
"""

from typing import Dict, Any, List
from pathlib import Path
from collections import Counter
import pandas as pd


def build_token_usage_table(usage_data: Dict[str, Dict[str, Any]]) -> str:
    """
    Build markdown table for token usage & latency summary.

    Args:
        usage_data: Dict mapping model name to usage dict with keys:
            - input_tokens, output_tokens, reasoning_tokens, total_tokens
            - elapsed_seconds, avg_latency_ms

    Returns:
        Markdown string for display
    """
    md = "### 📈 Token Usage & Latency Summary\n\n"
    md += "| Model | Input | Output | Reasoning | Total | Time (s) | Avg Latency (ms) |\n"
    md += "|-------|------:|-------:|----------:|------:|---------:|-----------------:|\n"

    for name, usage in usage_data.items():
        elapsed = usage.get('elapsed_seconds', 0)
        avg_lat = usage.get('avg_latency_ms', 0)
        md += (
            f"| {name} "
            f"| {usage['input_tokens']:,} "
            f"| {usage['output_tokens']:,} "
            f"| {usage['reasoning_tokens']:,} "
            f"| {usage['total_tokens']:,} "
            f"| {elapsed:.1f} "
            f"| {avg_lat:,.0f} |\n"
        )

    return md


def build_metrics_comparison_table(
    all_results: Dict[str, pd.DataFrame],
    col_mapping: Dict[str, str]
) -> str:
    """
    Build markdown table for Recall/Precision/F2 comparison.

    Args:
        all_results: Dict mapping model name to DataFrame with evaluation results
        col_mapping: Dict mapping metric name to column name in DataFrame

    Returns:
        Markdown string for display
    """
    md = "### 📊 Results Comparison\n\n"

    # Header row
    md += "| Metric |"
    for name in all_results.keys():
        md += f" {name} |"
    md += " **Best** |\n"

    # Separator row
    md += "|--------|"
    for _ in all_results.keys():
        md += "-------:|"
    md += "--------|\n"

    # Data rows
    for metric_name, col in col_mapping.items():
        md += f"| **{metric_name.capitalize()}** |"
        scores = {}
        for name, df in all_results.items():
            score = df[col].mean()
            scores[name] = score
            md += f" {score:.3f} |"
        best = max(scores, key=scores.get)
        md += f" {best} |\n"

    return md


def build_cost_analysis_table(
    cost_results: Dict[str, Dict[str, float]],
    usage_data: Dict[str, Dict[str, Any]],
    config: Dict[str, Any]
) -> str:
    """
    Build markdown table for cost analysis.

    Args:
        cost_results: Dict mapping model name to cost dict from calculate_model_cost()
        usage_data: Dict mapping model name to usage dict
        config: Configuration dict with keys:
            - training_runs, training_hours_per_run, training_rate
            - hosting_hours_monthly, hosting_rate
            - monthly_requests, projection_months

    Returns:
        Markdown string for display
    """
    training_cost = config['training_runs'] * config['training_hours_per_run'] * config['training_rate']
    hosting_cost = config['hosting_hours_monthly'] * config['hosting_rate']

    md = f"""### 💰 COST ANALYSIS

**Configuration:**
- Training: {config['training_runs']} run × {config['training_hours_per_run']}h @ ${config['training_rate']}/h = **${training_cost:.0f}** (one-time)
- Hosting: {config['hosting_hours_monthly']}h/month @ ${config['hosting_rate']}/h = **${hosting_cost:,.0f}**/month
- Projection: **{config['monthly_requests']:,} requests/month** over **{config['projection_months']} months**

| Model | Reasoning | Total Tokens | Cost/100k req | Training | Hosting/mo | **{config['projection_months']}-mo Total** |
|-------|----------:|-------------:|--------------:|---------:|-----------:|------------------:|
"""

    model_order = [
        "o4-mini (vanilla)",
        "gpt-5.2 (none)",
        "gpt-5.2 (low)",
        "gpt-5.2 (medium)",
        "gpt-5.2 (high)",
        "o4-mini (fine-tuned)"
    ]

    for name in model_order:
        if name not in cost_results:
            continue

        usage = usage_data[name]
        cost = cost_results[name]

        reasoning = usage["reasoning_tokens"]
        total = usage["total_tokens"]

        # Cost for 100k requests
        cost_100k = cost["inference_cost_per_1k"] * 100

        # Training (one-time)
        training = cost["training_cost_total"]
        training_str = f"${training:,.0f}" if training > 0 else "-"

        # Hosting per month
        hosting = cost["hosting_cost_monthly"]
        hosting_str = f"${hosting:,.0f}" if hosting > 0 else "-"

        # N-month projection
        inference_monthly = cost["inference_cost_per_1k"] * (config['monthly_requests'] / 1000)
        total_projection = training + (hosting + inference_monthly) * config['projection_months']

        md += (
            f"| {name} "
            f"| {reasoning:,} "
            f"| {total:,} "
            f"| ${cost_100k:,.0f} "
            f"| {training_str} "
            f"| {hosting_str} "
            f"| **${total_projection:,.0f}** |\n"
        )

    # Summary
    baseline_total = cost_results["o4-mini (vanilla)"]["inference_cost_per_1k"] * (config['monthly_requests'] / 1000) * config['projection_months']

    finetuned_cost = cost_results["o4-mini (fine-tuned)"]
    finetuned_total = finetuned_cost["training_cost_total"] + \
        (finetuned_cost["hosting_cost_monthly"] + \
         finetuned_cost["inference_cost_per_1k"] * (config['monthly_requests'] / 1000)) * config['projection_months']

    gpt52_high_total = cost_results["gpt-5.2 (high)"]["inference_cost_per_1k"] * (config['monthly_requests'] / 1000) * config['projection_months']

    md += f"""
**📊 {config['projection_months']}-Month Cost Comparison @ {config['monthly_requests']:,} requests/month:**
- o4-mini (vanilla): **${baseline_total:,.0f}**
- o4-mini (fine-tuned): **${finetuned_total:,.0f}** (includes ${finetuned_cost["training_cost_total"]:.0f} training + ${finetuned_cost["hosting_cost_monthly"]:,.0f}/mo hosting)
- gpt-5.2 (high): **${gpt52_high_total:,.0f}**
"""

    return md


def build_roi_analysis_markdown(
    f2_scores: Dict[str, float],
    cost_results: Dict[str, Dict[str, float]],
    breakeven: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """
    Build markdown for ROI analysis: fine-tuned vs gpt-5.2 (high).

    Args:
        f2_scores: Dict mapping model name to F2 score
        cost_results: Dict mapping model name to cost dict
        breakeven: Break-even analysis dict from calculate_breakeven()
        config: Configuration dict with monthly_requests, projection_months

    Returns:
        Markdown string for display
    """
    finetuned_cost = cost_results["o4-mini (fine-tuned)"]
    gpt52_high_cost = cost_results["gpt-5.2 (high)"]

    training_cost_total = finetuned_cost["training_cost_total"]
    hosting_cost_monthly = finetuned_cost["hosting_cost_monthly"]
    finetuned_cost_per_1k = finetuned_cost["inference_cost_per_1k"]
    gpt52_high_cost_per_1k = gpt52_high_cost["inference_cost_per_1k"]

    md = f"""### 📈 ROI ANALYSIS: o4-mini (fine-tuned) vs gpt-5.2 (high)

**Why gpt-5.2 (high)?** Closest in F2 performance:
- o4-mini (fine-tuned): F2 = {f2_scores["o4-mini (fine-tuned)"]:.3f}
- gpt-5.2 (high): F2 = {f2_scores["gpt-5.2 (high)"]:.3f}

**💵 Fine-tuning Investment:**
- Training: **${training_cost_total:,.0f}** (one-time)
- Hosting: **${hosting_cost_monthly:,.0f}**/month

**📊 Inference Cost Comparison (per 1,000 requests):**
| Model | Cost/1k req |
|-------|------------:|
| o4-mini (fine-tuned) | ${finetuned_cost_per_1k:.2f} |
| gpt-5.2 (high) | ${gpt52_high_cost_per_1k:.2f} |
| **Inference savings** | **${gpt52_high_cost_per_1k - finetuned_cost_per_1k:.2f}** |
"""

    if breakeven["is_viable"]:
        md += f"""
**🎯 Break-even Analysis:**

To cover the **${hosting_cost_monthly:,.0f}/month hosting cost**, you need enough requests
so that inference savings = hosting cost:

```
break_even = hosting_monthly / savings_per_1k × 1000
           = ${hosting_cost_monthly:,.0f} / ${breakeven['savings_per_1k']:.2f} × 1000
           = {breakeven['breakeven_monthly']:,} requests/month
           = {breakeven['breakeven_daily']:,} requests/day
```

**💡 At your projected volume ({config['monthly_requests']:,} requests/month):**
"""
        inference_savings_monthly = breakeven['savings_per_1k'] * (config['monthly_requests'] / 1000)
        net_monthly = inference_savings_monthly - hosting_cost_monthly

        md += f"""
| Component | Amount |
|-----------|-------:|
| Inference savings | +${inference_savings_monthly:,.0f} |
| Hosting cost | -${hosting_cost_monthly:,.0f} |
| **Net monthly** | **${net_monthly:+,.0f}** |
"""

        if net_monthly > 0:
            md += f"\n✅ **Above break-even!** You save ${net_monthly:,.0f}/month vs gpt-5.2 (high)."
        else:
            md += f"\n⚠️ **Below break-even.** You lose ${-net_monthly:,.0f}/month vs gpt-5.2 (high). Need {breakeven['breakeven_monthly']:,} req/month to break even."
    else:
        md += f"""
**⚠️ o4-mini (fine-tuned) is MORE expensive per request than gpt-5.2 (high)!**
- Additional cost: ${-breakeven['savings_per_1k']:.2f} per 1,000 requests
- Recommendation: Use gpt-5.2 (high) directly, or try Developer tier (no hosting cost).
"""

    # Total cost projection
    finetuned_monthly_total = hosting_cost_monthly + (finetuned_cost_per_1k * config['monthly_requests'] / 1000)
    gpt52_high_monthly = gpt52_high_cost_per_1k * config['monthly_requests'] / 1000

    finetuned_total = training_cost_total + (finetuned_monthly_total * config['projection_months'])
    gpt52_high_total = gpt52_high_monthly * config['projection_months']

    diff_total = gpt52_high_total - finetuned_total

    md += f"""
---
**📊 {config['projection_months']}-Month Total Cost @ {config['monthly_requests']:,} requests/month:**

| Model | Monthly | {config['projection_months']}-Month Total |
|-------|--------:|----------------:|
| o4-mini (fine-tuned) | ${finetuned_monthly_total:,.0f} | ${finetuned_total:,.0f} |
| gpt-5.2 (high) | ${gpt52_high_monthly:,.0f} | ${gpt52_high_total:,.0f} |
| **Difference** | | **${diff_total:+,.0f}** |

*o4-mini (fine-tuned) total = ${training_cost_total:,.0f} training + ({config['projection_months']} × ${finetuned_monthly_total:,.0f})*
"""

    return md


def build_summary_table(
    f2_scores: Dict[str, float],
    model_configs: Dict[str, str],
    num_samples: int
) -> str:
    """
    Build summary markdown with F2 scores and delta vs baseline.

    Args:
        f2_scores: Dict mapping model name to F2 score
        model_configs: Dict mapping display name to deployment name
        num_samples: Number of evaluation samples

    Returns:
        Markdown string for display
    """
    baseline_f2 = f2_scores.get("o4-mini (vanilla)", 0)

    md = f"""### 🏆 EVALUATION COMPLETE

| Config | Model |
|--------|-------|
"""
    for display_name, deployment in model_configs.items():
        md += f"| {display_name} | {deployment} |\n"

    md += f"""
**Samples:** {num_samples}

**F2 Scores (vs o4-mini vanilla):**
| Model | F2 Score | Delta |
|-------|----------|-------|
"""

    for name, f2 in f2_scores.items():
        delta = ((f2 - baseline_f2) / baseline_f2 * 100) if baseline_f2 > 0 else 0
        md += f"| {name} | {f2:.3f} | {delta:+.1f}% |\n"

    # Determine winner
    winner = max(f2_scores, key=f2_scores.get)
    winner_score = f2_scores[winner]

    if winner == "o4-mini (fine-tuned)":
        md += "\n🎉 **o4-mini (fine-tuned) wins!** RFT provides value even vs larger models with reasoning."
    elif "5.2" in winner:
        md += f"\n⚠️ **{winner} wins** with F2={winner_score:.3f}. Consider cost/latency tradeoffs vs using a larger model with reasoning."
    else:
        md += "\n🤔 **o4-mini (vanilla) wins.** Something may be wrong with the fine-tuning."

    md += "\n\n🚀 Run `05_multiagent_with_tool_calling.ipynb` to test the multi-agent workflow!"

    return md


# =========================================================================
# Notebook 05 — Multi-agent workflow tables
# =========================================================================


def build_react_eval_summary(results: List[Dict[str, Any]]) -> str:
    """
    Build markdown summary for ReAct workflow evaluation results.

    Args:
        results: List of per-case result dicts with keys:
            success, recall, precision, f2, iterations, tools_passed,
            num_mutations, execution_time_s
    """
    n = len(results)
    success_count = sum(1 for r in results if r["success"])
    avg_recall = sum(r["recall"] for r in results) / n
    avg_precision = sum(r["precision"] for r in results) / n
    avg_f2 = sum(r["f2"] for r in results) / n
    total_iterations = sum(r.get("iterations", 0) for r in results)
    total_mutations = sum(r.get("num_mutations", 0) for r in results)
    avg_time = sum(r.get("execution_time_s", 0) for r in results) / n
    avg_iterations = total_iterations / n
    avg_tools_passed = sum(r.get("tools_passed", 0) for r in results) / n

    md = "### 📊 Evaluation Results (ReAct with Filtered Tools)\n\n"

    md += "| Metric | Value |\n"
    md += "|--------|------:|\n"
    md += f"| Cases evaluated | {n} |\n"
    md += f"| Workflow success rate | {success_count}/{n} ({100*success_count/n:.1f}%) |\n"
    md += f"| Total iterations (tool calls) | {total_iterations} |\n"
    md += f"| Avg iterations per case | {avg_iterations:.1f} |\n"
    md += f"| Avg tools passed to Executor | {avg_tools_passed:.1f} / 15 |\n"
    md += f"| Total database mutations | {total_mutations} |\n"
    md += f"| Avg execution time | {avg_time:.2f}s |\n"

    md += "\n**Tool Prediction Metrics (Planner):**\n\n"
    md += "| Metric | Score |\n"
    md += "|--------|------:|\n"
    md += f"| Recall | {avg_recall:.3f} |\n"
    md += f"| Precision | {avg_precision:.3f} |\n"
    md += f"| **F2 Score** | **{avg_f2:.3f}** |\n"

    return md


def build_tool_usage_table(results: List[Dict[str, Any]]) -> str:
    """
    Build markdown table comparing expected vs planned vs executed tools.

    Args:
        results: List of per-case result dicts with keys:
            tools_planned, tools_executed, tools_expected
    """
    all_planned = []
    all_executed = []
    all_expected = []

    for r in results:
        all_planned.extend(r.get("tools_planned", []))
        all_executed.extend(r.get("tools_executed", []))
        all_expected.extend(r.get("tools_expected", []))

    planned_counts = Counter(all_planned)
    executed_counts = Counter(all_executed)
    expected_counts = Counter(all_expected)

    all_tools = sorted(
        set(planned_counts.keys()) | set(executed_counts.keys()) | set(expected_counts.keys()),
        key=lambda t: expected_counts.get(t, 0),
        reverse=True
    )

    md = "### 🔧 Tool Usage Comparison\n\n"
    md += "| Tool | Expected | Planned | Executed |\n"
    md += "|------|--------:|---------:|---------:|\n"

    for tool in all_tools:
        exp = expected_counts.get(tool, 0)
        pln = planned_counts.get(tool, 0)
        exe = executed_counts.get(tool, 0)
        md += f"| `{tool}` | {exp} | {pln} | {exe} |\n"

    md += f"| **Total** | **{sum(expected_counts.values())}** | **{sum(planned_counts.values())}** | **{sum(executed_counts.values())}** |\n"

    return md


def build_azure_eval_summary(eval_results: List[Dict[str, Any]]) -> str:
    """
    Build markdown summary for Azure AI Evaluation SDK metrics.

    Args:
        eval_results: List of per-case eval dicts with keys:
            task_adherence, intent_resolution, tool_call_accuracy, summary
    """
    def safe_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    ta_scores = [s for s in (safe_float(e["task_adherence"].get("task_adherence"))
                             for e in eval_results) if s is not None]
    ir_scores = [s for s in (safe_float(e["intent_resolution"].get("intent_resolution"))
                             for e in eval_results) if s is not None]
    tca_scores = [s for s in (safe_float(e["tool_call_accuracy"].get("tool_call_accuracy"))
                              for e in eval_results) if s is not None]

    md = "### 🤖 Azure AI Evaluation SDK Metrics\n\n"
    md += "| Evaluator | Scale | Score | Pass Rate |\n"
    md += "|-----------|-------|------:|----------:|\n"

    if ta_scores:
        pass_rate = sum(ta_scores) / len(ta_scores)
        md += f"| Task Adherence | binary | {int(sum(ta_scores))}/{len(ta_scores)} | {pass_rate:.1%} |\n"
    else:
        md += "| Task Adherence | binary | - | - |\n"

    if ir_scores:
        mean_ir = sum(ir_scores) / len(ir_scores)
        ir_pass = sum(1 for s in ir_scores if s >= 3) / len(ir_scores)
        md += f"| Intent Resolution | 1-5 | {mean_ir:.2f} | {ir_pass:.1%} (>=3) |\n"
    else:
        md += "| Intent Resolution | 1-5 | - | - |\n"

    if tca_scores:
        mean_tca = sum(tca_scores) / len(tca_scores)
        tca_pass = sum(1 for s in tca_scores if s >= 3) / len(tca_scores)
        md += f"| Tool Call Accuracy | 1-5 | {mean_tca:.2f} | {tca_pass:.1%} (>=3) |\n"
    else:
        md += "| Tool Call Accuracy | 1-5 | - | - |\n"

    all_passed = sum(1 for e in eval_results if e.get("summary", {}).get("all_passed"))
    if eval_results:
        md += f"\n**All 3 passed:** {all_passed}/{len(eval_results)} ({100*all_passed/len(eval_results):.0f}%)\n"

    return md


def build_workflow_summary(
    planner_model: str,
    executor_model: str,
    results: List[Dict[str, Any]],
    outputs_dir: Path
) -> str:
    """
    Build final workflow summary markdown.

    Args:
        planner_model: Planner deployment name
        executor_model: Executor deployment name
        results: List of per-case result dicts
        outputs_dir: Path to output directory
    """
    n = len(results)
    success_count = sum(1 for r in results if r["success"])
    avg_f2 = sum(r["f2"] for r in results) / n
    avg_iterations = sum(r.get("iterations", 0) for r in results) / n
    avg_tools_passed = sum(r.get("tools_passed", 0) for r in results) / n
    avg_time = sum(r.get("execution_time_s", 0) for r in results) / n
    total_mutations = sum(r.get("num_mutations", 0) for r in results)

    md = f"""### 🏆 Final Summary

**Architecture:**
> Planner (RFT fine-tuned) → ExecutorAgent (ReAct with filtered tools)

**Key Features:**
- Planner predicts which tools are needed
- ExecutorAgent receives ONLY predicted tools (not all 15)
- Tool descriptions via API (docstrings) + context via prompt (tool_definitions.json)

| | Model |
|--|-------|
| **Planner** | `{planner_model}` |
| **Executor** | `{executor_model}` (ReAct, reasoning_effort=None) |

**Results ({n} cases):**

| Metric | Value |
|--------|------:|
| Success rate | {100*success_count/n:.1f}% |
| F2 Score | {avg_f2:.3f} |
| Avg tools passed | {avg_tools_passed:.1f} / 15 |
| Avg iterations | {avg_iterations:.1f} |
| Avg execution time | {avg_time:.2f}s |
| Total mutations | {total_mutations} |

**Outputs:**
- `{outputs_dir / 'react_results.csv'}`
- `{outputs_dir / 'react_traces.json'}`
- `{outputs_dir / 'react_summary.json'}`
- `{outputs_dir / 'react_distributions.png'}`
"""

    return md
