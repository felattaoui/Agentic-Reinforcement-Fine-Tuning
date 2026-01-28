# Reinforcement Fine-Tuning for LLMs: A Practical Guide
## Part 2: Multi-Agent Orchestration

Part 1 of this guide established the fundamentals of Reinforcement Fine-Tuning: how it differs from supervised approaches, the types of graders available, and the workflow for effective training. This second part applies those concepts to a specific challenge that has become central to enterprise AI deployments: orchestrating multi-agent systems.

Modern AI applications increasingly rely on specialized agents working together. A customer service system might have separate agents for order management, technical support, billing inquiries, and escalation handling. A legal research platform might coordinate agents for document retrieval, citation checking, precedent analysis, and summary generation. The value of these systems depends critically on correct orchestration, routing each request to the right agents in the right sequence.

This orchestration problem is where RFT demonstrates its clearest advantage over supervised fine-tuning. The combinatorial nature of routing decisions, where the optimal path depends on subtle contextual factors, makes it impractical to enumerate all correct examples. RFT allows the model to explore different routing strategies and learn from outcomes, developing a coherent decision-making policy rather than memorizing specific patterns.

## Why RFT for Agent Orchestration

Part 1 covered RFT fundamentals. For orchestration specifically, three properties make RFT particularly effective:

**Combinatorial decision space.** With 15 tools and typical sequences of 3-7 calls, the space of valid plans is vast. Many sequences are functionally equivalent (order doesn't matter for independent tools), while others have strict dependencies. SFT memorizes specific sequences; RFT learns which factors actually matter for routing decisions.

**Partial credit.** A plan that identifies 4 of 5 required tools is far more useful than one that identifies none. SFT treats both as equally "wrong." RFT with an appropriate grader rewards near-misses, providing gradient signal that pulls the model toward better solutions incrementally.

**Tool evolution.** Agent systems accumulate tools over time—legacy versions, specialized variants, overlapping functionality. SFT on historical examples perpetuates old patterns. RFT, evaluating outcomes rather than mimicking examples, can learn to prefer better tools if they lead to higher scores.

## Data Format Determines Architecture

Before preparing training data for an agent orchestration task, you must decide what architecture you're training for. This decision constrains what data format makes sense, and getting it wrong means your training data won't support your intended use case.

The two dominant patterns for agent orchestration are what we might call the Planner pattern and the ReAct pattern. In the Planner pattern, the model receives a request and outputs a complete plan, the full sequence of tools to call, before any execution begins. The system then executes that plan, possibly with a separate mechanism to handle failures. In the ReAct pattern (Reasoning and Acting), the model interleaves thinking and action: it decides on one tool call, observes the result, reasons about what to do next, makes another call, and continues until the task is complete.

These patterns require fundamentally different training data. For a Planner, your training examples contain a request and a reference tool sequence. The model learns to predict sequences from requests. For ReAct, your training examples contain traces of reasoning and action interleaved with observations. The model learns when to act, when to reason, and how to incorporate feedback.

The Planner pattern is simpler to train and evaluate. You can score the predicted sequence against the reference sequence using straightforward metrics. Inference is also simpler and cheaper, one model call produces the full plan. The limitation is that the plan is fixed; if execution reveals unexpected information, the system needs separate logic to adapt.

The ReAct pattern is more flexible but harder to train. Your grader must evaluate traces of variable length where the same outcome might be reached through different reasoning paths. Inference requires multiple model calls in sequence, increasing latency and cost. The advantage is adaptability, the model can adjust its approach based on what it learns during execution.

For the case study in this article, we use the Planner pattern. This choice was driven partly by the available data (which provided tool sequences rather than full execution traces) and partly by the production context (where predictable latency mattered more than mid-execution adaptation). The principles we demonstrate, grader design, multi-metric scoring, testing before training, apply equally to ReAct architectures, though the specific metrics would differ.

## Case Study: A Retail Customer Service Planner

To demonstrate RFT for agent orchestration concretely, we developed a planner for retail customer service. The task: given a customer request, predict which tools the system will need to call and in what order.

We used tau-bench as a source of realistic customer service scenarios. Tau-bench is a benchmark designed to evaluate agent execution in simulated environments, with tasks covering order management, returns, exchanges, and account modifications. We transformed this data into a planning format, using the task descriptions as inputs and the required tool sequences as reference outputs. This is not the intended use of tau-bench, but it provided realistic queries and ground-truth tool sequences that would otherwise require extensive manual labeling.

The transformed dataset contains 115 tasks with 15 available tools spanning account management, order operations, and utility functions. (Note: tau-bench defines 16 tools, but we excluded the `think` tool—an internal reasoning mechanism that never appears in ground-truth action sequences.) We split the data into 85 training examples, 20 for validation, and 10 for final testing. Each example follows this structure:

```json
{
  "messages": [
    {
      "role": "developer",
      "content": "You are a retail customer service planner. Given a customer request, identify which tools are needed and in what order.\n\nAvailable tools:\n- calculate: Calculate the result of a mathematical expression.\n- cancel_pending_order: Cancel a pending order.\n- exchange_delivered_order_items: Exchange items in a delivered order...\n[remaining tools listed]\n\nOutput only the list of tools needed, one per line, in execution order."
    },
    {
      "role": "user",
      "content": "I want to check how much I paid for my most recent order. I'm not sure when I placed it."
    }
  ],
  "reference_answer": "{\"expected_tools\": [\"find_user_id_by_name_zip\", \"get_user_details\", \"get_order_details\"], \"num_actions\": 3}"
}
```

Note that we use the `developer` role for task instructions rather than `system`, as RFT for reasoning models requires this role. The `reference_answer` encodes both the expected tool sequence and its length as a JSON string. We chose this format because our Python grader needs to parse structured data, and encoding it as JSON within a single field proved more reliable than using multiple top-level fields.

The system prompt lists all available tools with brief descriptions. This gives the model the vocabulary it needs and establishes the expected output format. The user message presents the customer request exactly as it would appear in production.

---

> ### Understanding the Scope: Planners vs. Executing Agents
>
> This tutorial trains a **Planner**—a model that predicts which tools will be needed—not an **executing agent** that calls tools, observes results, and adapts. This distinction has important implications.
>
> **What tau-bench actually is:** tau-bench was designed as a *simulated environment* for evaluating executing agents. It maintains internal state (user accounts, order statuses, inventory) and executes tool calls against that state. The reward signal comes from whether the task was *functionally completed*—did the order actually get cancelled? Was the refund applied to the correct payment method?
>
> **What we're doing instead:** We extract the *expected tool sequences* from tau-bench tasks and train a model to predict those sequences from the task description alone, without execution. This is planning, not execution. The model never sees execution results during training or inference.
>
> **Why this matters:** A Planner can fail even if it predicts the "right" tools, because execution might reveal edge cases (order already cancelled, user not found, etc.). Conversely, an executing agent might succeed despite an imperfect initial plan by adapting mid-execution. Our evaluation metrics measure planning quality, not task completion. This is a deliberate scope limitation that makes the problem tractable for a tutorial while still demonstrating RFT's value.

---

## Designing the Grader: Recall, Precision, and F2

The grader is where domain expertise becomes code. For a planning task, we care about two primary qualities: whether the model identifies all necessary tools (recall), and whether it avoids predicting unnecessary tools (precision). Each metric captures a different failure mode.

Recall measures completeness. If the reference sequence is [A, B, C, D] and the model predicts [A, B, C], recall is 0.75, three of four required tools were identified. A model that consistently misses tools will produce plans that fail during execution, requiring fallback handling or human intervention.

Precision measures efficiency. If the model predicts [A, B, C, D, E] when only [A, B, C, D] are needed, precision is 0.8, four of five predicted tools were actually necessary. Low precision means wasted computation and latency as the system calls tools that don't contribute to the outcome. It can also indicate confusion about tool functionality.

We combine these using the **F2 score**, a weighted harmonic mean that prioritizes recall:

```python
F2 = 5 * (precision * recall) / (4 * precision + recall)
```

Why F2 instead of F1? In a planning context, missing a required tool is more costly than predicting an extra one. A missing tool causes the workflow to fail or produce incomplete results. An extra tool wastes some computation but doesn't break the system. The F2 score weights recall four times more heavily than precision, reflecting this asymmetry.

To further discourage over-prediction without harming recall, we add a small penalty per extra tool predicted (we used 3%, though this is a tunable hyperparameter). This prevents the model from hedging by listing every plausible tool—a strategy that would inflate recall at the cost of practical utility.

Note on ordering: We initially considered an order metric to penalize wrong tool sequences. However, analysis of our enriched training data revealed ~21% of samples had inconsistent ordering due to GPT enrichment variability. More importantly, our architecture uses an intelligent executor that can reorder tool calls based on dependencies. For planners feeding into smart executors, identifying the right tools matters more than their sequence.

The full grader implementation handles edge cases that would otherwise corrupt training signal. Critically, it supports both JSON structured output (when using `response_format`) and text fallback for backward compatibility:

```python
def grade(sample, item):
    import json

    output_text = sample.get("output_text", "") or ""
    ref_raw = item.get("reference_answer", {})
    reference = json.loads(ref_raw) if isinstance(ref_raw, str) else ref_raw
    expected_tools = reference.get("expected_tools", [])

    if not expected_tools:
        return 1.0 if not output_text.strip() else 0.0

    if not output_text:
        return 0.0

    valid_tools = [
        "calculate", "cancel_pending_order", "exchange_delivered_order_items",
        "find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details",
        "get_product_details", "get_user_details", "list_all_product_types",
        "modify_pending_order_address", "modify_pending_order_items",
        "modify_pending_order_payment", "modify_user_address",
        "return_delivered_order_items", "transfer_to_human_agents"
    ]
    valid_lower = [t.lower() for t in valid_tools]

    # Try JSON parsing first (structured output)
    pred_lower = None
    try:
        parsed = json.loads(output_text)
        if isinstance(parsed, dict) and "tools" in parsed:
            tools = parsed["tools"]
            if isinstance(tools, list):
                pred_lower = [t.lower() for t in tools if t.lower() in valid_lower]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback to text parsing if JSON fails
    if pred_lower is None:
        response = output_text.lower().replace("-", "_")
        pred_tools = []
        for tool in valid_tools:
            if tool in response:
                pos = response.find(tool)
                pred_tools.append((pos, tool))
        pred_tools.sort(key=lambda x: x[0])
        pred_lower = [t[1] for t in pred_tools]

    exp_lower = [t.lower() for t in expected_tools]

    # Recall: % of expected tools found
    found = sum(1 for t in exp_lower if t in pred_lower)
    recall = found / len(exp_lower)

    # Precision: % of predicted tools that are correct
    if len(pred_lower) == 0:
        precision = 0.0
    else:
        correct = sum(1 for t in pred_lower if t in exp_lower)
        precision = correct / len(pred_lower)

    # F2: recall-weighted harmonic mean (beta=2)
    if precision + recall == 0:
        return 0.0

    f2 = 5 * (precision * recall) / (4 * precision + recall)

    # Penalty: 3% per extra tool to discourage over-prediction
    extra_tools = sum(1 for t in pred_lower if t not in exp_lower)
    penalty = extra_tools * 0.03

    return min(max(f2 - penalty, 0.0), 1.0)
```

The grader first attempts to parse JSON structured output—the format produced when training with `response_format`. If JSON parsing fails (malformed output or text response from baseline models), it falls back to substring matching. This dual-mode approach ensures the grader works correctly for both structured and unstructured responses, enabling fair comparison between models trained with different configurations.

## Testing the Grader Before Training

A grader that seems reasonable can have subtle bugs that corrupt training. Testing is not optional, it's the highest-leverage activity in an RFT project.

We test with synthetic cases that cover the score range and probe potential failure modes:

```python
test_cases = [
    {
        "name": "Perfect match",
        "output": "find_user_id_by_name_zip\nget_order_details\ncancel_pending_order",
        "expected": ["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"],
        "expected_score": 1.0  # F2 = 1.0
    },
    {
        "name": "Missing one tool (3/4)",
        "output": "find_user_id_by_name_zip\nget_order_details\nexchange_delivered_order_items",
        "expected": ["find_user_id_by_name_zip", "get_order_details", "get_product_details", "exchange_delivered_order_items"],
        "expected_score": 0.789  # recall=0.75, precision=1.0, F2=0.789
    },
    {
        "name": "One extra tool",
        "output": "find_user_id_by_name_zip\nget_user_details\nget_order_details\ncancel_pending_order",
        "expected": ["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"],
        "expected_score": 0.908  # recall=1.0, precision=0.75, F2=0.938, penalty=0.03
    },
    {
        "name": "Empty output",
        "output": "",
        "expected": ["find_user_id_by_name_zip", "get_order_details"],
        "expected_score": 0.0
    },
    {
        "name": "Half correct (2/4)",
        "output": "find_user_id_by_name_zip\nexchange_delivered_order_items",
        "expected": ["find_user_id_by_name_zip", "get_order_details", "get_product_details", "exchange_delivered_order_items"],
        "expected_score": 0.556  # recall=0.5, precision=1.0, F2=0.556
    }
]
```

Running these tests before training revealed bugs in our initial grader implementation, including issues with tool detection and duplicate counting. These would have silently corrupted training signal. Ten minutes of synthetic testing saves hours of debugging failed training runs.

## Training Configuration

With data prepared and grader tested, training configuration is straightforward. The key parameters for RFT on Azure OpenAI:

```python
training_config = {
    "model": "o4-mini-2025-04-16",
    "training_file": train_file_id,
    "validation_file": val_file_id,
    "method": {
        "type": "reinforcement",
        "reinforcement": {
            "grader": {
                "type": "python",
                "name": "planner_grader",
                "source": grader_code
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "planner_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered list of tools to execute"
                            }
                        },
                        "required": ["tools"],
                        "additionalProperties": False
                    }
                }
            }
        }
    },
    "hyperparameters": {
        "n_epochs": 4,
        "reasoning_effort": "medium",
        "compute_multiplier": 1.5
    },
    "suffix": "planner-retail"
}
```

**Note on structured output during training:** Part 1 noted that training can be more stable without enforcing structured output, allowing the model to explore more freely. The `response_format` parameter shown above is optional, you can omit it and let the model produce text responses. Our grader handles both formats (JSON parsing with text fallback), so you can experiment with either approach. The schema becomes essential at inference time for reliable parsing.

The `response_format` parameter tells Azure OpenAI to constrain the model's output to valid JSON matching our schema during training. This produces structured responses that our grader can parse reliably. The grader handles both JSON and text fallback for backward compatibility.

We chose `n_epochs=4` based on preliminary experiments showing continued improvement through epoch 3 with diminishing returns after. The `reasoning_effort` of "medium" balances response quality against training cost. A `compute_multiplier` of 1.5 provides moderate exploration, appropriate for a task with multiple valid approaches.

Training completed in approximately 2 hours on our 85-example dataset. The Azure portal provides real-time monitoring of training and validation rewards, which helps detect reward hacking early if validation reward diverges from training reward.

## Results

We conducted two levels of evaluation: the planner component in isolation, and the complete multi-agent system end-to-end.

> **Note:** Results vary between training runs due to the stochastic nature of reinforcement learning and depend on hyperparameters (epochs, reasoning effort, compute multiplier) and dataset size. Run the evaluation notebooks to observe your own results.

### Planner Evaluation

First, we evaluated the planner's ability to predict correct tool sequences, comparing baseline o4-mini against the fine-tuned model on held-out examples.

In our experiments, the fine-tuned planner showed consistent improvements across all metrics. Recall improved meaningfully—the model identifies more of the required tools. Precision remained high for both models, suggesting neither hallucinates unnecessary tools. The F2 score, which prioritizes recall while still penalizing over-prediction, showed substantial gains. This is exactly what we optimized for.

### End-to-End Multi-Agent Evaluation

The more meaningful test is whether improved planning translates to better system-level outcomes. We integrated both planners into a complete workflow using the **ReAct pattern** (Reasoning + Acting), where a single ExecutorAgent iterates through tool calls until the task is complete.

The architecture is deliberately simple:
1. **Planner** (fine-tuned model) predicts which tools are needed
2. **ExecutorAgent** (gpt-5.2 with ReAct) executes tools iteratively, observing results and deciding next actions
3. The agent continues until the task is complete or max iterations reached

This replaces our previous architecture of 4 specialized sub-agents (Account, Order, Refund, Utility) with a Coordinator routing between them. The ReAct pattern proved more effective because it can naturally iterate through entities—if a user has 3 orders and only one is "pending", the ExecutorAgent will check each until it finds the right one, rather than failing on the first mismatch.

In our experiments, the difference between baseline and fine-tuned planners was substantial. The fine-tuned planner showed transformative improvements across all metrics.

The recall improvement was particularly dramatic in the end-to-end context. The baseline planner in isolation may look reasonable, but when its outputs flow through the full system, errors compound. A plan missing one tool causes incomplete context for subsequent operations. The fine-tuned planner's higher recall provides the robustness needed for reliable end-to-end execution.

### Interpreting F2 Score

The F2 score is our primary metric: a recall-weighted harmonic mean of precision and recall. It prioritizes finding all required tools while still penalizing over-prediction. A high F2 indicates the planner reliably identifies needed tools without excessive false positives.

Analysis of failure cases typically reveals patterns around semantically similar tools: `return_delivered_order_items` versus `exchange_delivered_order_items`, or `cancel_pending_order` versus `return_delivered_order_items`. The model understands the intent but occasionally picks the wrong specific action.

These edge cases don't diminish the results—they contextualize them. In production, a plan with high F2 would likely succeed, and the F2 score accurately reflects practical utility.

### Qualitative Evaluation with Azure AI Evaluation SDK

Recall and precision tell you whether the right tools were selected, but they say nothing about whether the final response actually helped the customer. Azure AI Evaluation SDK addresses this gap with agent-specific evaluators that use an LLM as a judge to assess workflow quality.

Our architecture separates evaluation concerns: the **Planner** predicts *which* tools to call (measured by F2/Recall/Precision), while the **ExecutorAgent** extracts arguments and executes tools (measured by ToolCallAccuracy). TaskAdherence and IntentResolution evaluate the end result—the quality of the final response.

We use three evaluators from the SDK:
- **TaskAdherenceEvaluator**: Does the final response address the user's request? Returns binary pass/fail (1.0 or 0.0).
- **IntentResolutionEvaluator**: Was the user's intent correctly understood? Returns a 1-5 score.
- **ToolCallAccuracyEvaluator**: Were tools called with correct arguments? Returns a 1-5 score. This measures the quality of argument extraction by the ExecutorAgent.

When aggregating results, treat TaskAdherence as a pass rate (percentage of cases that passed) rather than averaging it with the 1-5 scores from other evaluators.

Integrating with the SDK requires converting your workflow traces to a specific message format. The SDK expects tool calls in a nested structure where the function name and arguments are wrapped inside a `tool_call` object. Getting this format wrong is a common source of "the evaluator returns unexpected scores" issues. The [Azure AI Evaluation SDK documentation](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/agent-evaluate-sdk) details the expected schema and provides examples for different agent frameworks.

These qualitative metrics complement rather than replace the deterministic recall/precision calculations. A workflow might achieve perfect recall—all required tools were called—yet still fail task adherence because the synthesized response was confusing or incomplete. Conversely, a workflow with lower recall might still pass task adherence if the missing tool wasn't critical to answering the user's actual question. Using both quantitative and qualitative evaluation provides a more complete picture of system performance.

## Cost Analysis and ROI

Performance improvements only matter if they justify the investment. Fine-tuning incurs upfront costs (training compute, experimentation iterations) and ongoing costs (dedicated model hosting). When does the investment pay off?

### Understanding the Cost Structure

Azure OpenAI pricing for fine-tuned models differs fundamentally from pay-per-use models:

| Cost Component | Baseline (o4-mini) | Fine-tuned |
|----------------|-------------------|------------|
| Input tokens | $1.10/1M | $1.10/1M |
| Output tokens | $4.40/1M | $4.40/1M |
| Training | — | $100/hour |
| Hosting | — | $1.70/hour (Standard tier) |

The key insight: fine-tuned models have the same per-token inference cost as the base model, but add fixed monthly hosting costs. This creates a break-even calculation.

### Reasoning Tokens: The Hidden Cost

Models like o4-mini and gpt-5.x with reasoning capabilities consume "reasoning tokens"—internal deliberation that doesn't appear in the output but is billed at the output rate. Our evaluation captured these separately:

```
Model             | Input   | Output  | Reasoning | Total
------------------|---------|---------|-----------|--------
Baseline (o4-mini)| 14,876  | 29,908  | 28,800    | 73,584
gpt-5.2 (none)    | 14,876  | 1,226   | 0         | 16,102
gpt-5.2 (low)     | 14,876  | 5,919   | 4,547     | 25,342
gpt-5.2 (medium)  | 14,876  | 10,547  | 9,148     | 34,571
gpt-5.2 (high)    | 14,876  | 15,524  | 14,139    | 44,539
Fine-tuned        | 14,876  | 16,675  | 15,296    | 46,847
```

The baseline o4-mini uses extensive reasoning (28,800 tokens) to achieve its F2 score. The fine-tuned model uses moderate reasoning (15,296 tokens) but achieves higher F2—it has internalized the task-specific knowledge through training.

### Break-Even Analysis

For a concrete example, consider comparing the fine-tuned model against gpt-5.2 with high reasoning (which achieved closest F2 in our evaluation):

**Fixed costs (monthly):**
- Training: 5 runs × 3 hours × $100 = $1,500 (amortized over 6 months = $250/month)
- Hosting: 720 hours × $1.70 = $1,224/month
- Total fixed: $1,474/month

**Variable costs (per 1,000 requests):**
- Fine-tuned: ~$5.00
- gpt-5.2 (high reasoning): ~$13.00

**Savings per 1,000 requests:** $8.00

**Break-even point:** $1,224 / $8.00 × 1,000 = ~153,000 requests/month (~5,100/day)

Below this volume, the hosting costs dominate and pay-per-use is cheaper. Above it, the fine-tuned model becomes increasingly economical.

### When Fine-Tuning Makes Economic Sense

The decision framework is straightforward:

1. **High volume, stable task**: If you process >10K requests/day on a well-defined task, fine-tuning likely pays for itself within months.

2. **Quality-critical applications**: When improved F2 translates to fewer escalations, reduced support costs, or higher customer satisfaction, the ROI calculation includes those downstream savings.

3. **Latency-sensitive systems**: Fine-tuned models often require less reasoning to achieve the same quality, reducing both cost and latency.

4. **Experimentation budget**: The 3 training runs in our example aren't wasted—each iteration teaches you about the task and grader design. Treat this as R&D investment.

The `src/cost/` module provides utilities for calculating your specific break-even point based on your pricing tier, expected volume, and training investment.

## Multi-Agent Integration

The architecture that produced these results separates concerns deliberately. The **Planner** (the fine-tuned model) receives customer requests and outputs structured JSON with reasoning and tool sequences. The **ExecutorAgent** then uses the ReAct pattern to execute those tools iteratively, observing results and adapting as needed.

```
User Request
    ↓
Planner (Fine-tuned o4-mini)
    ↓ Predicts: ["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"]
    ↓
ExecutorAgent (gpt-5.2, ReAct pattern, 15 tools)
    ↓
    ├── [1] find_user_id_by_name_zip → "emma_smith_8564"
    ├── [2] get_user_details → {orders: ["#W2417020", "#W5605613"]}
    ├── [3] get_order_details("#W2417020") → {status: "pending"}
    ├── [4] cancel_pending_order("#W2417020") → "Cancelled"
    ↓
Database (tau-bench JSON) → Real user/order/product data
    ↓
Final Response → Generated by ExecutorAgent after task completion
```

This architecture replaced an earlier design with 4 specialized sub-agents (Account, Order, Refund, Utility) and a Coordinator routing between them. **Why ReAct over specialized sub-agents?** The previous architecture had a fundamental limitation: each sub-agent executed one-shot without observing results. If a user had 3 orders and only one was "pending", the sub-agent would check the first order, find it "delivered", and stop—unable to iterate. The ReAct pattern naturally handles this: the ExecutorAgent observes each result and decides whether to continue, retry with different parameters, or move to the next tool.

### Structured Output

Structured output is used throughout the system—both during RFT training and at inference time. During training, the `response_format` parameter constrains the model to produce valid JSON. At inference, we use the Responses API with `text.format`.

**Important:** The Fine-tuning API and Responses API use different parameter formats:
- **Fine-tuning API**: `response_format={"type": "json_schema", "json_schema": {...}}`
- **Responses API**: `text={"format": schema["json_schema"]}` (extract the inner `json_schema` content)

This means you can define your schema once and convert it for each API. The Planner uses structured output via a Pydantic model:

```python
class PlannerResponse(BaseModel):
    tools: List[str]  # Ordered tool sequence
```

This design eliminates text parsing and enables programmatic control flow. The workflow receives typed `PlannerResponse` objects directly from the Planner, then passes the tool list to the ExecutorAgent which handles execution via its native ReAct loop.

The ExecutorAgent doesn't need structured output—it uses Agent Framework's native tool calling, which automatically handles argument extraction and function dispatch. This is a recurring principle: use structured output where you need predictable parsing (Planner output), use native capabilities where the framework already provides them (tool execution).

### Dynamic Tool Filtering and Intelligent Few-Shot

A key optimization in our architecture is **dynamic tool filtering**. Instead of giving the ExecutorAgent access to all 15 tools on every request, the Planner's prediction determines which subset of tools the executor receives. This provides three benefits:

1. **Reduced cognitive load**: With 6 tools instead of 15, the model focuses on relevant options
2. **Token savings**: Fewer tool descriptions in the prompt (~60% reduction in tool tokens)
3. **Faster execution**: On average 23% fewer iterations per task

But filtering alone wasn't enough. Initial evaluations showed the ExecutorAgent often stopped before completing final actions—it would look up user details and order status but fail to execute the actual cancel, return, or exchange. The model understood *what* to do but didn't follow through.

We addressed this with **intelligent few-shot selection**. Rather than including all examples or none, we dynamically select ONE example based on the action type predicted by the Planner:

```python
ACTION_TOOLS = {
    "cancel_pending_order": "cancel",
    "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange",
    "modify_pending_order_items": "modify",
    "modify_pending_order_address": "modify",
    "modify_pending_order_payment": "modify",
}

def select_fewshot_example(tools_predicted: list) -> dict | None:
    """Select ONE example matching the action type in tools_predicted."""
    for tool in tools_predicted:
        if tool in ACTION_TOOLS:
            action_type = ACTION_TOOLS[tool]
            return find_example_by_action(action_type)
    return None
```

The selected example shows a complete execution trace for that action type, demonstrating the full workflow from user lookup through final action execution. This teaches the model to complete the action rather than stopping at information gathering.

**A/B test results** confirmed the value of this approach:

| Configuration | TCA Pass Rate | All 3 Evaluators Passed |
|---------------|---------------|-------------------------|
| Without few-shot | 65.5% | 66% |
| With intelligent few-shot | **89.7%** | **90%** |

The improvement is substantial: +24% on Tool Call Accuracy. The few-shot example acts as a behavioral template, showing the model that the workflow should end with an ACTION tool, not just information retrieval.

**Token economics**: Comparing intelligent few-shot against a brute-force approach (all tools + all examples) shows ~80% token savings per request:

| Approach | Tokens/Request | Cost/1k Requests |
|----------|----------------|------------------|
| Brute force (15 tools + 13 examples) | ~10,500 | $18.38 |
| Intelligent (6 tools + 1 example) | ~2,150 | $3.76 |
| **Savings** | **~8,350 (79.6%)** | **$14.62** |

This is a double win: the intelligent approach is both cheaper AND more effective. Surgical context selection outperforms flooding the model with information.

The implementation lives in `src/settings.py` (`load_executor_prompt_dynamic`, `select_fewshot_example`) with examples stored in `src/config/fewshot_examples.json`.

## Real Database: From Stubs to tau-bench JSON

Early versions of this tutorial used stub tools that returned hardcoded responses. While convenient for prototyping, this created a gap between the tutorial and production reality. Real tools fail, return unexpected data, and modify state in ways that affect subsequent operations.

We bridged this gap by integrating the actual tau-bench database—500+ users, 1000+ orders, and 50+ products stored as JSON files. This introduces realistic complexity:

- **Data dependencies**: `get_order_details` requires a valid order_id from the database
- **State mutations**: `cancel_pending_order` modifies the order's status
- **Validation failures**: Attempting to cancel an already-delivered order returns an error
- **Cross-references**: Orders reference user_ids and product_ids that must exist

### Snapshot Isolation

A concern with real data is pollution—running the workflow multiple times would corrupt the database. We solve this with snapshot isolation:

```python
# Before each workflow execution
db_snapshot = database.snapshot()  # Deep copy
set_active_database(db_snapshot)

# Tools modify the snapshot, not the original
cancel_pending_order("W0001234", "customer_request")

# Original database remains unchanged
# Next workflow gets a fresh snapshot
```

This pattern ensures:
1. Each workflow operates on isolated state
2. The tutorial can be run repeatedly without side effects
3. Mutations are tracked for debugging (`db_snapshot.get_mutations()`)

### Tool Implementation Pattern

Each of the 15 tools follows a consistent pattern:

```python
def cancel_pending_order(order_id: str, reason: str) -> str:
    db = get_active_database()  # Get current snapshot

    order = db.orders.get(order_id)
    if not order:
        return f"Error: Order {order_id} not found"

    if order["status"] != "pending":
        return f"Error: Order {order_id} is {order['status']}, cannot cancel"

    # Modify state
    order["status"] = "cancelled"
    db.record_mutation("cancel_order", {"order_id": order_id, "reason": reason})

    return f"Order {order_id} has been cancelled. Reason: {reason}"
```

The `get_active_database()` function returns the current snapshot, set at workflow start. This global-reference pattern is a pragmatic choice—agent frameworks typically register tools as standalone functions, making dependency injection impractical.

## Lessons Learned

Part 1 covered general RFT lessons (grader investment, testing, partial credit, reward hacking prevention). Here are insights specific to multi-agent orchestration:

**Data format is an architectural commitment.** We built a Planner because our data provided tool sequences. ReAct-style adaptive execution would require traces with observations interleaved. Decide what architecture you want, then format data to support it—not the other way around.

**Benchmarks may not match your use case.** tau-bench was designed for executing agents, not planning agents. We transformed it to fit our needs, which worked, but we were always aware our task wasn't what the benchmark was designed for. Adapt resources to your needs, but be honest about what you're measuring.

**Real data exposes real problems.** Stub tools hide production complexity. Integrating real database operations revealed edge cases (orders in unexpected states, users with incomplete profiles) that simpler architectures couldn't handle.

**Thread isolation is essential.** Agent Framework stores all conversation state in `AgentThread` objects—agents themselves are stateless. Without explicit thread management (`agent.get_new_thread()` for each request), agents accumulate conversation history across requests, causing tools to execute for wrong users. This manifests as `tools_executed >> tools_planned` in evaluation metrics.

**Content filtering may need adjustment post-RFT.** The fine-tuned model became more sensitive to Azure's jailbreak detection, with some tau-bench prompts triggering false positives. The solution: create a custom RAI policy that disables jailbreak detection while keeping other safety filters enabled. This is specific to evaluation on benchmarks; production deployments should maintain appropriate safety filters.

---

*The complete implementation, including data preparation, grader code, training configuration, and evaluation scripts, is available in the accompanying notebooks. Part 1 of this guide covers the fundamentals of Reinforcement Fine-Tuning for readers seeking additional background on grader types, reward hacking, and the technical evolution from RLHF to RFT.*
