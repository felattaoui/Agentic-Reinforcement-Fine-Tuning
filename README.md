# Reinforcement Fine-Tuning for Retail Agent Planning

A complete tutorial and implementation for **Reinforcement Fine-Tuning (RFT)** using Azure OpenAI's o4-mini model, trained on the tau-bench retail dataset for intelligent tool selection in customer service scenarios.

## Project Overview

This repository demonstrates how to use **Reinforcement Fine-Tuning** to train a language model that acts as a **planner** in a multi-agent retail customer service system. The planner learns to select the right tools and determine their execution order for handling customer requests.

### Why RFT Instead of Supervised Fine-Tuning?

| Approach | How It Learns | Best For |
|----------|---------------|----------|
| **SFT** | "Here's the correct answer, learn it" | Tasks with single correct outputs |
| **RFT** | "Here's a score, improve yourself" | Tasks with multiple valid paths |

For tool selection tasks, there's rarely one "correct" answer—the order might vary, multiple tools might work, and context matters. RFT lets the model **explore strategies** and learn from rewards rather than memorizing fixed patterns.

## Architecture

```
User Request
    ↓
┌─────────────────────────┐
│        PLANNER          │
│  (Fine-tuned o4-mini)   │
│  "Which tools needed?"  │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│    EXECUTOR AGENT       │
│      (gpt-5.2)          │
│    ReAct Pattern        │
│  15 tools available     │
│  Iterates until done    │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│    RETAIL DATABASE      │
│   (tau-bench JSON)      │
│ 500 users, 1000 orders  │
└─────────────────────────┘
```

The **ExecutorAgent** uses the ReAct (Reasoning + Acting) pattern:
1. Receives the plan from the Planner
2. Executes tools one at a time via `agent.run()`
3. Observes results and decides next action
4. Iterates naturally (e.g., checks multiple orders to find the pending one)
5. Returns final response when task is complete

## Project Structure

```
RFT_agent_retail/
├── README.md
├── requirements.txt
├── .env                                # Azure credentials (create from template)
│
├── notebooks/
│   ├── 00_optional_RFT_graders_tutorial.ipynb    # Deep-dive into grader design
│   ├── 01_data_preparation.ipynb                 # Data exploration & preparation
│   ├── 02_training.ipynb                         # RFT training on Azure
│   ├── 03_deployment.ipynb                       # Deploy fine-tuned model
│   ├── 04_planner_evaluation.ipynb               # Planner-only evaluation
│   ├── 05_multiagent_with_tool_calling.ipynb     # Full workflow with ReAct agent
│   └── 06_debug_tca_analysis.ipynb               # Tool Call Accuracy debugging
│
├── data/
│   ├── datasets/                       # RFT training datasets
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── tau_bench/                      # Real database (copied from tau-bench)
│       ├── users.json                  # 500+ users
│       ├── orders.json                 # 1000+ orders
│       └── products.json               # 50+ products
│
├── src/
│   ├── settings.py                     # Centralized configuration
│   ├── data_utils.py                   # Data loading helpers
│   ├── config/
│   │   ├── planner_schema.json         # JSON schema for structured output
│   │   ├── executor_prompt.txt         # ExecutorAgent system prompt
│   │   └── tool_definitions.json       # 15 retail tool definitions
│   ├── graders/
│   │   └── grader.py                   # Python grader for RFT (F1 score)
│   ├── evaluation/                     # Evaluation module
│   │   ├── evaluators.py               # Recall, Precision, F2 metrics
│   │   └── agent_evaluators.py         # Azure AI Eval SDK (TaskAdherence, TCA)
│   └── multiagent/                     # Multi-agent system
│       ├── agents.py                   # Agent creation (Planner, ExecutorAgent)
│       ├── workflow.py                 # ReAct workflow orchestration
│       ├── models.py                   # Pydantic models for structured output
│       ├── database/                   # Real tau-bench data layer
│       │   ├── store.py                # RetailDatabase with snapshot
│       │   └── loader.py               # JSON loading
│       └── tools/                      # 15 real retail tools
│           ├── account.py              # 4 tools
│           ├── order.py                # 7 tools
│           ├── refund.py               # 2 tools
│           └── utility.py              # 2 tools
│
└── article/
    ├── article_1.md                    # RFT Fundamentals
    └── article_2.md                    # Multi-Agent Orchestration
```

## Getting Started

### Prerequisites

- Python 3.10+
- Azure OpenAI resource with access to o4-mini
- Azure subscription for fine-tuning jobs

### Installation

```bash
# Clone the repository
git clone https://github.com/felattaoui/RFT_agent_retail.git
cd RFT_agent_retail

# Create conda environment
conda create -n rft-retail python=3.10 -y
conda activate rft-retail

# Install dependencies
pip install -r requirements.txt
```

**Windows users:** If you encounter encoding errors during installation:
```bash
set PYTHONUTF8=1
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=o4-mini
AZURE_OPENAI_DEPLOYMENT_VANILLA=gpt-5.2
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
FINETUNED_DEPLOYMENT=retail-agent-ft
EXECUTOR_DEPLOYMENT=gpt-5.2
```

## Notebook Workflow

### Step 0: Understanding Graders (Optional)
**`00_optional_RFT_graders_tutorial.ipynb`**

A comprehensive tutorial on designing graders for RFT:
- Reinforcement Learning fundamentals and RLHF history
- All Azure OpenAI grader types: string_check, text_similarity, model, Python
- Hands-on examples of grader anti-patterns and best practices

### Step 1: Data Preparation
**`01_data_preparation.ipynb`**

Explore and prepare the tau-bench retail dataset:
- Load and analyze 115 customer service tasks
- Understand the 15 available retail tools
- Transform data to RFT format with train/val/test splits

### Step 2: Training
**`02_training.ipynb`**

Launch and monitor RFT training on Azure OpenAI:
- Configure the training job with custom Python grader
- Monitor reward progression and training metrics
- Save the trained model for deployment

### Step 3: Deployment
**`03_deployment.ipynb`**

Deploy the fine-tuned model to Azure OpenAI:
- Create deployment from fine-tuned model
- Configure RAI policies for content filtering

### Step 4: Planner Evaluation
**`04_planner_evaluation.ipynb`**

Compare 3 planner configurations on held-out test data:
- **Baseline** (o4-mini)
- **gpt-5.2** (vanilla)
- **Fine-tuned** (o4-mini RFT)

Metrics: Recall, Precision, F2 Score (recall-weighted). This notebook evaluates the **Planner only**.

### Step 5: Multi-Agent Integration
**`05_multiagent_with_tool_calling.ipynb`**

Run the complete multi-agent workflow with real database:
- Planner predicts required tools
- ExecutorAgent (ReAct pattern) executes tools iteratively
- Real function calling on tau-bench JSON data
- Azure AI Evaluation SDK metrics (TaskAdherence, IntentResolution, ToolCallAccuracy)

### Step 6: Debug & Analysis
**`06_debug_tca_analysis.ipynb`**

Debug Tool Call Accuracy issues:
- Analyze execution traces
- Compare planned vs executed tools
- Identify argument extraction errors

## The Grader

The grader is the "brain" of RFT—it defines what "good" means for your task. Our grader scores model outputs using the **F2 score**:

| Metric | Description |
|--------|-------------|
| **Recall** | Are the expected tools mentioned? |
| **Precision** | Ratio of correct tools to predicted tools |
| **F2 Score** | Recall-weighted F-score (recall 4x more important) |

```python
F2 = 5 * (precision * recall) / (4 * precision + recall)
```

F2 prioritizes recall because missing a required tool is worse than predicting an extra tool.

## Database Isolation

Each workflow execution operates on a **snapshot** of the database. The original JSON files are never modified, making the tutorial safe to run multiple times without data corruption.

```python
# Workflow automatically creates a snapshot
db_snapshot = database.snapshot()  # Deep copy
set_active_database(db_snapshot)
# Tools modify the snapshot, not the original
```

## Available Tools (15 total)

| Category | Tools |
|----------|-------|
| **Account (4)** | `find_user_id_by_email`, `find_user_id_by_name_zip`, `get_user_details`, `modify_user_address` |
| **Order (7)** | `get_order_details`, `cancel_pending_order`, `modify_pending_order_address`, `modify_pending_order_items`, `modify_pending_order_payment`, `get_product_details`, `list_all_product_types` |
| **Refund (2)** | `return_delivered_order_items`, `exchange_delivered_order_items` |
| **Utility (2)** | `transfer_to_human_agents`, `calculate` |

## Key Concepts

### Reinforcement Fine-Tuning (RFT)
Unlike supervised fine-tuning where you provide correct answers, RFT lets the model explore and learn from a reward signal. The model generates multiple responses, each is scored by the grader, and the training algorithm adjusts weights to favor high-scoring outputs.

### The Grader is Everything
Your model can only become as good as your grader can measure. If your grader has blind spots, the model will exploit them.

### Planning vs Execution
This project trains a **planner** that predicts which tools are needed, not an executor that calls them. The planner output feeds into an **ExecutorAgent** that uses the ReAct pattern to iteratively execute tools until the task is complete.

## Technologies

| Technology | Purpose |
|------------|---------|
| **Azure OpenAI** | o4-mini base model, RFT fine-tuning infrastructure |
| **Microsoft Agent Framework** | ExecutorAgent with ReAct pattern |
| **Azure AI Evaluation SDK** | TaskAdherence, IntentResolution, ToolCallAccuracy metrics |
| **tau-bench** | Benchmark dataset and database for retail agent evaluation |
| **Python** | Custom graders, data processing, evaluation |

## References

- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning)
- [Reinforcement Fine-Tuning Guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/reinforcement-fine-tuning)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [tau-bench: Benchmarking LLM Agents](https://github.com/sierra-research/tau-bench)
- [GRPO Paper (DeepSeek)](https://arxiv.org/abs/2402.03300)

## License

This project is provided for educational purposes. The tau-bench dataset is subject to its own license terms.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

*Built with Azure OpenAI and Microsoft Agent Framework*
