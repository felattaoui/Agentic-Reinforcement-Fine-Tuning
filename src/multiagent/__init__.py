"""
Multi-agent workflow module for the RFT workshop.

This module provides:
- Real retail tools adapted from tau-bench (15 tools)
- Database layer with users, orders, products (snapshot isolation)
- ExecutorAgent with ReAct pattern and dynamic tool filtering
- Agent creation utilities
- Workflow orchestration
- Evaluation functions
"""

from src.multiagent.tools import (
    get_all_tools,
    get_tools_by_names,
    get_account_tools,
    get_order_tools,
    get_refund_tools,
    get_utility_tools,
    reset_tool_log,
    get_tool_log,
    set_active_database,
    get_active_database,
    KNOWN_TOOLS,
    print_tools_summary
)

from src.multiagent.database import (
    RetailDatabase,
    load_database,
    copy_tau_bench_data
)

from src.multiagent.agents import (
    create_executor_with_tools,
    create_planner,
    create_planners
)

from src.multiagent.workflow import (
    extract_tools_from_plan,
    run_react_workflow,
    run_planner_only
)

from src.multiagent.evaluation import evaluate_plan

__all__ = [
    # Tools
    "get_all_tools",
    "get_tools_by_names",
    "get_account_tools",
    "get_order_tools",
    "get_refund_tools",
    "get_utility_tools",
    "reset_tool_log",
    "get_tool_log",
    "set_active_database",
    "get_active_database",
    "KNOWN_TOOLS",
    "print_tools_summary",
    # Database
    "RetailDatabase",
    "load_database",
    "copy_tau_bench_data",
    # Agents
    "create_executor_with_tools",
    "create_planner",
    "create_planners",
    # Workflow
    "extract_tools_from_plan",
    "run_react_workflow",
    "run_planner_only",
    # Evaluation
    "evaluate_plan",
]
