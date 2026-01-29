# CLAUDE.md - Project Reference

## Overview

**Agentic-Reinforcement-Fine-Tuning** is a workshop and tutorial implementation of Reinforcement Fine-Tuning (RFT) for training an intelligent retail customer service planner using Azure OpenAI's o4-mini model and the tau-bench dataset.

## Project Structure

```
Agentic-Reinforcement-Fine-Tuning/
├── notebooks/                    # Jupyter tutorials (6 notebooks)
│   ├── 00_optional_RFT_graders_tutorial.ipynb
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   ├── 03_deployment.ipynb
│   ├── 04_planner_evaluation.ipynb
│   └── 05_multiagent_with_tool_calling.ipynb
│
├── src/                          # Main application code
│   ├── settings.py               # Centralized configuration
│   ├── azure_client.py           # Azure OpenAI client
│   ├── data_utils.py             # Data loading/saving utilities
│   │
│   ├── config/                   # Configuration files
│   │   ├── planner_schema.json   # JSON schema for structured planner output
│   │   ├── executor_prompt.txt   # ExecutorAgent system prompt
│   │   ├── tool_definitions.json # 15 retail tools definition
│   │   └── enrichment_prompt.txt # Dataset enrichment prompt
│   │
│   ├── graders/                  # RFT training grader
│   │   ├── grader.py             # F1 score grader
│   │   └── tests.py
│   │
│   ├── evaluation/               # Model evaluation module
│   │   ├── evaluators.py         # Recall, Precision, F1 evaluators
│   │   ├── agent_evaluators.py   # Azure AI Eval SDK (TaskAdherence, IntentResolution)
│   │   ├── generate.py
│   │   ├── deployment.py
│   │   └── content_filter.py
│   │
│   ├── training/                 # Training utilities
│   │   └── job_utils.py
│   │
│   ├── cost/                     # Cost analysis utilities
│   │   ├── __init__.py
│   │   └── pricing.py            # ROI and break-even calculations
│   │
│   ├── multiagent/               # Multi-agent orchestration
│   │   ├── agents.py             # Agent creation (Planner, ExecutorAgent)
│   │   ├── workflow.py           # ReAct workflow orchestration
│   │   ├── models.py             # Pydantic models for structured output
│   │   ├── evaluation.py         # F2 metrics (re-export from evaluators)
│   │   ├── tools.py
│   │   ├── database/             # Retail database layer
│   │   │   ├── store.py          # RetailDatabase with snapshots
│   │   │   └── loader.py         # JSON data loader
│   │   └── tools/                # 15 retail tools
│   │       ├── base.py           # Global database bridge
│   │       ├── account.py        # User lookup/management
│   │       ├── order.py          # Order operations
│   │       ├── refund.py         # Returns/exchanges
│   │       └── utility.py        # Transfer, calculate
│
├── data/                         # Input data
│   ├── datasets/                 # RFT training datasets
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── tau_bench/                # Real retail database
│       ├── users.json
│       ├── orders.json
│       └── products.json
│
├── outputs/                      # Generated artifacts
│   ├── job_history.json          # RFT training job history
│   ├── planner/                  # Planner evaluation results
│   │   ├── responses/            # Model responses (eval_*.jsonl)
│   │   ├── eval_*.csv
│   │   ├── eval_summary.json
│   │   └── eval_results.png
│   ├── multiagent/               # Multi-agent workflow results
│   │   ├── multiagent_*.csv
│   │   ├── multiagent_*.json
│   │   └── multiagent_*.png
│   └── eda/                      # EDA plots from data prep
│       └── eda_*.png
│
└── article/                      # Educational articles
    ├── article_1.md              # RFT Fundamentals
    └── article_2.md              # Multi-Agent Orchestration
```

## Notebooks

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| `00_optional_RFT_graders_tutorial` | Tutorial on RFT graders (string_check, text_similarity, python, model) using Responses API | Self-contained |
| `01_data_preparation` | Load tau-bench data, enrich with gpt-5.2, split train/val/test | `src/settings`, `src/data_utils` |
| `02_training` | Launch RFT fine-tuning job on Azure OpenAI | `src/settings`, `src/azure_client`, `src/training/` |
| `03_deployment` | Deploy fine-tuned model to Azure endpoint | `src/settings`, `src/evaluation/deployment` |
| `04_planner_evaluation` | Compare 3 planner configs (baseline, gpt-5.2, fine-tuned) with F2 metrics | `src/settings`, `src/evaluation/` |
| `05_multiagent_with_tool_calling` | Full workflow with ExecutorAgent, ReAct pattern, tau-bench database | `src/multiagent/`, `src/evaluation/` |

## Technologies

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Azure OpenAI** | Base model (o4-mini) + RFT training |
| **Microsoft Agent Framework** | Multi-agent orchestration |
| **tau-bench** | Retail benchmark dataset |
| **pandas/numpy** | Data processing |
| **openai SDK** | Azure OpenAI API (>=1.40.0) |
| **azure-identity** | Azure authentication |
| **azure-ai-evaluation** | Evaluation metrics |

## Agent Framework with Azure

### Thread Isolation (Critical for Multi-Agent Workflows)

**Problem:** Agent Framework agents accumulate conversation history between `agent.run()` calls, causing context contamination when the same agent handles multiple independent requests.

**Root Cause:** From Agent Framework docs:
> "Agents themselves are stateless. All state is preserved in the AgentThread object."

When calling `agent.run(task)` without specifying a thread, the agent uses an implicit default thread that persists across calls.

**Solution:** Create a fresh thread for each independent request:

```python
# ❌ WRONG: Reuses implicit thread, causes memory leak
async def run_agent(agent, task: str):
    result = await agent.run(task)
    return result.text

# ✅ CORRECT: Fresh thread per request, isolated context
async def run_agent(agent, task: str):
    thread = agent.get_new_thread()  # Fresh context
    result = await agent.run(task, thread=thread)
    return result.text
```

**Symptoms of Missing Thread Isolation:**
- `tools_executed >> tools_planned` in evaluation (agents recall previous conversations)
- Tools executed for wrong users (cross-session contamination)
- Certain tools never executed despite being planned (agent confused by accumulated context)

**Best Practice:** Always use `agent.get_new_thread()` when:
- Processing independent requests in a loop (evaluation, batch processing)
- Handling concurrent users in production
- Any scenario where requests should have isolated context

### OpenAI Responses API on Microsoft Foundry

Microsoft Foundry exposes an OpenAI-compatible endpoint at `/openai/v1/` that supports the Responses API.
Use `OpenAIResponsesClient` (standard OpenAI client) with `base_url` pointing to Azure:

```python
from agent_framework.openai import OpenAIResponsesClient
from src.settings import AZURE_API_KEY, AZURE_ENDPOINT

def create_client(deployment_name):
    return OpenAIResponsesClient(
        api_key=AZURE_API_KEY,
        base_url=f"{AZURE_ENDPOINT}/openai/v1/",
        model_id=deployment_name
    )

client = create_client("gpt-5.2")
```

**Why not `AzureOpenAIResponsesClient`?**
- `AzureOpenAIResponsesClient` uses a native Azure endpoint that may not exist on all Foundry deployments (404 errors)
- `OpenAIResponsesClient` with `base_url` works reliably on Microsoft Foundry

**Why Responses API over Chat Completions?**
- Responses API is the latest OpenAI API with native reasoning support
- Consistent with `src/evaluation/generate.py` which uses `AsyncOpenAI` with the same pattern
- Standardized across notebooks 04 and 05

### Known Issue: `instruction_role` Parameter Ignored

**Problem:** Agent Framework's `instruction_role` parameter is defined but **ignored** at the base class level.

```python
# This parameter EXISTS but is NOT USED
client = OpenAIResponsesClient(
    api_key=AZURE_API_KEY,
    base_url=f"{AZURE_ENDPOINT}/openai/v1/",
    model_id="gpt-5.2",
    instruction_role="developer"  # Stored but ignored!
)
```

**Root Cause:** In `agent_framework/_clients.py` (~line 572), the `BaseChatClient` hardcodes `role="system"`:

```python
# BUG: Always uses "system", ignores instruction_role
system_msg = ChatMessage(role="system", text=chat_options.instructions)
```

**Impact:**
- Training data uses `role="developer"` (OpenAI recommended)
- Agent Framework inference uses `role="system"` (hardcoded)
- Result: ~8.8% precision degradation for vanilla models (gpt-5.2)
- Fine-tuned models are less affected (learned patterns during training)

**Workaround:** For precision-critical planners, use direct API calls:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=AZURE_API_KEY, base_url=f"{AZURE_ENDPOINT}/openai/v1/")

response = await client.responses.create(
    model="gpt-5.2",
    input=[
        {"role": "developer", "content": system_prompt},  # Explicit control
        {"role": "user", "content": user_request}
    ]
)
```

**Full Investigation:** See `docs/DEBUG_agent_framework_instruction_role.md`

## Architecture

```
User Request → [Planner] → [ExecutorAgent] → [Database] → [Response]
                  ↓              ↓
           Fine-tuned o4-mini  ReAct pattern (gpt-5.2)
```

- **Planner**: RFT-trained model predicting needed tools
- **ExecutorAgent**: Single agent with all 15 tools, ReAct pattern (iterates automatically)
- **Database**: tau-bench data with snapshot isolation

## Code Conventions

### Type Hints
```python
def find_user_id_by_email(email: Annotated[str, "User email"]) -> str:
```

### Configuration
- All settings centralized in `src/settings.py`
- Environment variables via `.env` file
- Paths: `ROOT_DIR`, `DATA_DIR`, `DATASETS_DIR`, `CONFIG_DIR`, `OUTPUTS_DIR`, `PLANNER_OUTPUTS_DIR`, `MULTIAGENT_OUTPUTS_DIR`, `EDA_OUTPUTS_DIR`

### Database Access Pattern
```python
# Snapshot isolation for each workflow
db_snapshot = database.snapshot()
set_active_database(db_snapshot)

# Global bridge for tools
db = get_active_database()
```

### Grading (F2 Score)
```python
F2 = 5 * (precision * recall) / (4 * precision + recall)
```

## Key Files

| File | Description |
|------|-------------|
| `src/settings.py` | Centralized configuration, `load_system_prompt()` (from JSONL), `load_planner_schema()` |
| `data/datasets/train.jsonl` | Training data with embedded system prompt (single source of truth) |
| `src/config/planner_schema.json` | JSON schema for structured planner output |
| `src/config/executor_prompt.txt` | ExecutorAgent system prompt |
| `src/config/tool_definitions.json` | 15 retail tools as OpenAI function schemas |
| `src/graders/grader.py` | F2 grader for RFT training (supports JSON + text fallback) |
| `src/evaluation/evaluators.py` | `evaluate_plan()` single source of truth for Recall/Precision/F2 metrics |
| `src/evaluation/agent_evaluators.py` | Azure AI Evaluation SDK wrappers (TaskAdherence, IntentResolution) |
| `src/multiagent/agents.py` | Agent creation (Planner, ExecutorAgent) |
| `src/multiagent/workflow.py` | ReAct workflow orchestration |
| `src/multiagent/models.py` | Pydantic models for structured output (`PlannerResponse`) |
| `src/cost/pricing.py` | Cost calculation and break-even analysis functions |

## Environment Variables

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=o4-mini              # Baseline planner
AZURE_OPENAI_DEPLOYMENT_VANILLA=gpt-5.2      # Comparison model
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
FINETUNED_DEPLOYMENT=retail-agent-ft         # After training
EXECUTOR_DEPLOYMENT=gpt-5.2                  # ExecutorAgent model
```

## Available Tools (15)

- **Account** (4): `find_user_id_by_email`, `find_user_id_by_name_zip`, `get_user_details`, `modify_user_address`
- **Order** (7): `get_order_details`, `cancel_pending_order`, `modify_pending_order_*`, `get_product_details`, `list_all_product_types`
- **Refund** (2): `return_delivered_order_items`, `exchange_delivered_order_items`
- **Utility** (2): `transfer_to_human_agents`, `calculate`

## Workflow

1. **Data Preparation** (`01_data_preparation.ipynb`): Load tau-bench (115 tasks), enrich with gpt-5.2
2. **RFT Training** (`02_training.ipynb`): Fine-tune with F1 grader
3. **Deployment** (`03_deployment.ipynb`): Deploy fine-tuned model to Azure
4. **Planner Evaluation** (`04_planner_evaluation.ipynb`): Compare 3 planner configurations (Baseline, gpt-5.2, Fine-tuned)
5. **Multi-Agent (full workflow)** (`05_multiagent_with_tool_calling.ipynb`): Complete workflow with tau-bench database and ReAct pattern

## Running Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env_example .env
# Edit .env with your Azure credentials

# Run notebooks in order
jupyter notebook notebooks/
```

## Metrics

- **Recall**: % of expected tools that were predicted
- **Precision**: % of predicted tools that were expected
- **F2 Score**: Recall-weighted F-score (β=2, recall 4x more important than precision)

## Azure AI Evaluation SDK

The `src/evaluation/agent_evaluators.py` module wraps Azure AI Evaluation SDK's agent-specific evaluators for qualitative assessment of workflow results.

### Evaluation Strategy

Our architecture separates evaluation concerns:
- **Planner**: predicts WHICH tools to call → measured by **F2/Recall/Precision**
- **ExecutorAgent**: extracts arguments and executes → measured by **TaskAdherence/IntentResolution/ToolCallAccuracy**

We use three evaluators from the SDK:
- **TaskAdherenceEvaluator**: Does the final response address the user's request? Returns binary pass/fail (1.0 or 0.0)
- **IntentResolutionEvaluator**: Was the user's intent correctly understood? Returns 1-5 score
- **ToolCallAccuracyEvaluator**: Were tool calls executed with correct arguments? Returns 1-5 score

### Usage

```python
from src.evaluation.agent_evaluators import evaluate_workflow_result, print_evaluation_results

# Evaluate a workflow result
results = evaluate_workflow_result(
    user_request="I want to return my hiking boots",
    execution_trace=workflow_results["execution_trace"],
    final_response=workflow_results["final_response"]
)

print_evaluation_results(results)
# Output:
# Task Adherence: PASS [PASS]
# Intent Resolution: 5/5 [PASS]
# Summary: Passed 2/2
```

### Message Format

The SDK requires a specific agent message format. The `convert_workflow_trace_to_agent_messages()` function handles this conversion:

```python
# Tool calls must be nested in this structure:
{
    "role": "assistant",
    "content": [{
        "type": "tool_call",
        "tool_call": {
            "id": "call_tool_0",
            "function": {
                "name": "find_user_id_by_email",
                "arguments": {"email": "user@example.com"}
            }
        }
    }]
}
```

### Content Filter Handling

Azure's content filter may trigger on some tau-bench prompts (jailbreak detection false positives). The evaluators catch these errors and return `"skipped"` status instead of failing.

## Structured Output

The planner can produce structured JSON for reliable parsing. This is supported in both RFT training and inference.

### Schema
`src/config/planner_schema.json` defines the expected output format:
```json
{
  "tools": ["find_user_id_by_email", "get_order_details", ...]
}
```

### Loading the Schema
```python
from src.settings import load_planner_schema

schema = load_planner_schema()
# Returns: {"type": "json_schema", "json_schema": {...}}
```

### Usage in RFT Training
```python
from src.settings import load_planner_schema

job = client.fine_tuning.jobs.create(
    model="o4-mini-2025-04-16",
    training_file=train_file.id,
    method={
        "type": "reinforcement",
        "reinforcement": {
            "grader": GRADER_CONFIG,
            "response_format": load_planner_schema()  # Structured output
        }
    }
)
```

### Usage in Responses API (Inference/Evaluation)

**Important:** The Fine-tuning API and Responses API use different formats for structured output:

| API | Parameter | Format |
|-----|-----------|--------|
| **Fine-tuning API** | `response_format` | `{"type": "json_schema", "json_schema": {...}}` |
| **Responses API** | `text.format` | `{"type": "json_schema", "name": "...", "schema": {...}}` |

When using the schema with the Responses API, extract `json_schema`:
```python
from src.settings import load_planner_schema

schema = load_planner_schema()

# For Responses API: use text.format with json_schema content
response = await client.responses.create(
    model=model_name,
    input=messages,
    text={"format": schema["json_schema"]}  # Extract json_schema
)
```

### Backward Compatibility
All components (grader, evaluators, workflow) support both:
1. **JSON output**: Parsed from `{"tools": [...]}`
2. **Text output**: Tool names extracted via substring matching (fallback)

## Cost Analysis

The `src/cost/` module provides utilities for ROI analysis of fine-tuning investments.

### Azure OpenAI Pricing (December 2025)

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|----------------|-----------------|-------|
| o4-mini | $1.10 | $4.40 | Reasoning tokens billed at output rate |
| gpt-5.2 | $1.75 | $14.00 | Reasoning tokens billed at output rate |
| Fine-tuned | $1.10 | $4.40 | + $100/hr training, $1.70/hr hosting |

### Usage

```python
from src.cost import calculate_model_cost, calculate_breakeven

# Calculate costs for a model
cost = calculate_model_cost(
    usage={"input_tokens": 15000, "output_tokens": 5000, "reasoning_tokens": 10000},
    input_price=1.10,      # USD per 1M tokens
    output_price=4.40,     # USD per 1M tokens
    num_samples=30,        # For extrapolation to per-1k
    training_runs=3,       # Number of training experiments
    training_hours_per_run=1.0,
    training_rate=100.0,   # USD/hour
    hosting_hours_monthly=720,  # 24/7 hosting
    hosting_rate=1.70,     # USD/hour
    amortization_months=6  # Spread training cost
)

# Calculate break-even point
breakeven = calculate_breakeven(
    finetuned_cost_per_1k=cost["inference_cost_per_1k"],
    alternative_cost_per_1k=7.50,  # gpt-5.2 cost
    training_cost_total=cost["training_cost_total"],
    hosting_cost_monthly=cost["hosting_cost_monthly"],
    amortization_months=6
)
```

### Return Values

**calculate_model_cost()** returns:
- `inference_cost_eval`: Cost for the evaluation run
- `inference_cost_per_1k`: Cost per 1,000 requests
- `training_cost_total`: Total training cost (all runs)
- `hosting_cost_monthly`: Monthly hosting cost
- `fixed_cost_amortized_monthly`: Training cost amortized per month
- `total_fixed_monthly`: Total fixed costs per month

**calculate_breakeven()** returns:
- `savings_per_1k`: Savings per 1,000 requests
- `breakeven_month1`: Requests needed in month 1 (training + hosting)
- `breakeven_monthly`: Requests needed per month (hosting only)
- `breakeven_daily`: Daily request threshold
- `is_viable`: True if fine-tuning saves money per request

## Data Format

Training samples in JSONL:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Customer request..."}
  ],
  "tools": [{"function": {...}}],
  "expected_tools": ["tool1", "tool2"]
}
```

