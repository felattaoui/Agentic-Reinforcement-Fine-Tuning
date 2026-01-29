# Reinforcement Fine-Tuning for LLMs: A Practical Guide for Azure OpenAI models - part 1
## Fundamentals  
The way we train language models has changed fundamentally over the past two years. What started as a straightforward supervised learning problem (show the model examples, let it learn patterns) has evolved into something far more nuanced. We're no longer just teaching models to understand and respond. We're teaching them to reason, to make decisions, and most critically, to orchestrate complex workflows across multiple specialized agents.  
This shift matters because the applications being built today look nothing like the chatbots of 2022. Enterprises are deploying multi-agent systems that need to triage medical cases, analyze legal documents with multiple cross-references, or manage intricate financial workflows where a single wrong routing decision cascades into hours of wasted work. In these scenarios, the old question of "does the model understand my input?" has been replaced by a far more challenging one: "can it consistently make the optimal decision in my specific domain, under my constraints, thousands of times per day?"  
Traditional supervised fine-tuning struggles here for a fundamental reason. When you're training a model to route a customer query to the right specialist agent, there might be five possible agents and a dozen valid sequences depending on urgency, context, and current system state. You can't possibly label every combination. Even if you could, you'd be teaching the model to memorize rather than to develop a coherent decision-making strategy. This is where reinforcement fine-tuning enters the picture.

## The Technical Evolution: From Human Feedback to Programmatic Rewards  
To understand where reinforcement fine-tuning fits, it helps to trace the path that got us here.

The breakthrough with models like InstructGPT and ChatGPT came from RLHF: Reinforcement Learning from Human Feedback. The approach was clever: train a separate reward model on human preferences, then use that model to guide the language model toward responses people actually wanted. The algorithm doing the optimization was PPO (Proximal Policy Optimization), introduced by Schulman et al. in 2017, which carefully nudged the model toward higher rewards while ensuring it didn't drift too far from its original behavior and lose coherence.

RLHF worked remarkably well for general-purpose alignment. It taught models to be helpful, harmless, and honest (at least most of the time). But it had real limitations when you wanted to specialize for a particular domain. Training the reward model required thousands of human comparisons. The reward model itself could be unstable, sometimes learning spurious correlations from the preference data. And the entire pipeline was complex enough that iterating on it, (trying different reward formulations, testing variations) became a multi-week engineering project rather than something you could experiment with rapidly.

PPO itself, while powerful, came with significant overhead. As documented in recent research, PPO requires keeping multiple copies of the model in memory: the policy, 
the reference policy, and the critic—plus optimizer states for each. This memory overhead, combined with its sensitivity to hyperparameters, made it challenging for teams without extensive compute resources.

Direct Preference Optimization, or DPO, simplified this considerably. Instead of training a separate reward model and then using it to guide the language model, DPO directly optimizes the language model using preference pairs. If humans preferred response A over response B, DPO directly increases the probability of A-style responses and decreases B-style responses. This made the training more stable and computationally cheaper, and it opened the door to more rapid iteration. But you still needed those preference pairs, which meant human labeling at scale.

The breakthrough came with GRPO (Group Relative Policy Optimization), introduced by DeepSeek in their February 2024 work on mathematical reasoning (DeepSeekMath paper, arXiv:2402.03300). GRPO eliminated the need for a separate value model by leveraging the one-step Markov Decision Process nature of LLM fine-tuning. Instead of maintaining a critic network, GRPO samples multiple outputs for each prompt and uses their relative quality as the learning signal. This yields a significant reduction in memory requirements compared to PPO, eliminating the critic entirely while maintaining or improving performance.

Reinforcement fine-tuning represents a different approach entirely, one that's particularly well-suited to the new generation of reasoning models. Instead of learning what humans prefer, you define what you want programmatically. Write a grader: a function that scores model outputs based on your domain logic. Maybe it checks whether the model selected the right specialist agents for a medical case. Maybe it validates that a financial analysis included all required regulatory checks. Maybe it measures how efficiently the model solved a mathematical problem. The key insight is that for many enterprise applications, you can encode your success criteria directly rather than trying to infer them from human feedback.

This shift from human feedback to programmatic grading changes what becomes practical. You no longer need annotators to compare thousands of response pairs. You need domain expertise to define what "good" means in your context, translated into code that can evaluate outputs reliably.

## How RFT Works  

The RFT training loop follows a straightforward pattern. For each prompt in your training data, the platform generates multiple candidate responses. Each response passes through your grader, which returns a score between 0 and 1. The training algorithm then updates model weights to make high-scoring responses more likely and low-scoring responses less likely. This cycle repeats across your dataset for the specified number of epochs.

The key differences from Supervised Fine-Tuning become apparent in the data format. In SFT, you provide input-output pairs: here is the prompt, here is the correct response, learn to produce this output. In RFT, you provide prompts and reference information for grading, but no target response. The model explores the space of possible responses, and your grader provides the signal for what works.  
This exploration is both the strength and the challenge of RFT. The model can discover strategies you might not have anticipated or explicitly encoded in training examples. But exploration requires that the model already has some capability at the task—if it never produces a good response, there is nothing to reinforce. OpenAI's documentation states this clearly: if your baseline model has a 0% success rate at a task, RFT cannot bootstrap to higher performance.

The grader serves as the sole source of truth for what constitutes quality. Every design choice in your grader: (what you measure, how you weight different factors, what edge cases you handle) directly shapes what the model learns. A poorly designed grader teaches the wrong lessons, regardless of how much compute you invest in training.

## Model Support  
Reinforcement fine-tuning is supported on reasoning models. As of January 2026:  
- `o4-mini` (version `2025-04-16`) — generally available  
- `gpt-5` (version `2025-08-07`) — in private preview, availability may vary by subscription  
Check the [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning) for the current list of supported models.

## Grader Types on Azure OpenAI
Azure OpenAI provides several grader types, each suited to different evaluation needs. Understanding when to use each type is essential for effective RFT.  
**String check graders** apply simple string operations and return binary scores. They work well for classification tasks, exact-match extraction, or any scenario with definitive correct answers.  
```json{  "type": "string_check",  "name": "exact_match",  "input": "{{sample.output_text}}",  "reference": "{{item.expected_answer}}",  "operation": "eq"}```  
The supported operations include `eq` (exact match), `ne` (not equal), `like` (contains, case-sensitive), and `ilike` (contains, case-insensitive). These graders are fast and deterministic but offer no partial credit (outputs either match or they don't).
**Text similarity graders** compute scores based on established NLP metrics. They suit tasks where approximate matches matter, such as summarization or translation.  
```json{  "type": "text_similarity",  "name": "summary_quality",  "input": "{{sample.output_text}}",  "reference": "{{item.reference_summary}}",  "evaluation_metric": "rouge-l"}```  
Available metrics include `fuzzy_match` (using RapidFuzz), `bleu`, `gleu`, `meteor`, and various ROUGE variants. These provide continuous scores but measure surface-level similarity rather than semantic correctness.  
**Model graders** use another language model to evaluate outputs. They handle nuanced judgments that resist simple rules (assessing reasoning quality, checking factual accuracy against source documents, or evaluating style and tone).  
As of January 2026, available grader models on Azure OpenAI are `gpt-4o-2024-08-06` and `o3-mini-2025-01-31`. These do not require separate deployments in Microsoft Foundry. Refer to [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning) for the current list, as additional models may become available.  
```json{  "type": "score_model",  "name": "reasoning_quality",  "model": "gpt-4o-2024-08-06",  "input": [    {      "role": "user",      "content": "Evaluate the reasoning in this response: {{sample.output_text}}\n\nReference answer: {{item.reference_answer}}\n\nScore from 0 to 1 based on logical coherence and correctness."    }  ],  "range": [0, 1]}```  
Model graders are flexible but introduce their own failure modes. The grading model can be inconsistent, can fail to follow your rubric, or can develop blind spots to certain error types. When using model graders, test them thoroughly on representative examples before training.  
**Python graders** execute custom code in a sandboxed environment. They provide maximum flexibility for implementing domain-specific evaluation logic.  
```json{  "type": "python",  "name": "custom_scorer",  "source": "def grade(sample, item):\n    # Your scoring logic here\n    return score"}```  
The Python code executes in a sandboxed environment with no network access and a 2-minute runtime limit. Common data science libraries (numpy, pandas, scikit-learn, etc.) are available. Refer to [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning) for the full list of constraints and available libraries.  
Python graders are deterministic and can implement arbitrary logic, but require careful error handling, if your code raises exceptions too frequently, the training job fails.  
**Endpoint graders** (private preview as of January 2026) call a remote HTTP endpoint to score responses. They suit scenarios requiring external ground truth access or grading logic in languages other than Python. The API specification remains unpublished during preview.  
**Multi-graders** combine scores from multiple graders using arithmetic expressions. They enable multi-objective optimization where you care about several quality dimensions simultaneously.  
```json{  "type": "multi",  "name": "combined_score",  "graders": {    "correctness": { "type": "string_check", "...": "..." },    "fluency": { "type": "text_similarity", "...": "..." }  },  "calculate_output": "0.7 * correctness + 0.3 * fluency"}```  
The expression supports standard arithmetic operators and functions like `min`, `max`, `abs`, and `sqrt`. Note that Azure currently does not support model graders as sub-graders within a multi-grader configuration.

## Structured Output  
Azure OpenAI RFT supports structured output through the `response_format` parameter, which constrains the model to produce valid JSON matching a schema you define.  
In our experiments, we found training more stable without enforcing structured output, the model explores more freely in text mode. However, we enable structured output at inference time to guarantee parseable JSON responses. This works because reasoning models like o4-mini can produce structured JSON on demand, regardless of how they were trained.  
Design your grader to handle both formats: parse JSON first, fall back to text extraction if needed. This gives you flexibility to experiment with either approach.

## Reward Hacking: The Central Challenge  
Reward hacking occurs when a model finds ways to achieve high scores without genuinely solving the intended task. The model exploits gaps or ambiguities in your grader rather than developing the capabilities you want. This is not a theoretical concern but it is the most common failure mode in RFT projects.  
The symptoms are distinctive: training reward climbs steadily while validation reward plateaus or declines. The model produces outputs that technically satisfy your grader but feel wrong when you read them. In some cases, the model discovers "magic phrases" or formatting patterns that score well regardless of content.  
Consider a grader that measures recall—whether the model mentions required elements. A model optimizing purely for recall might list every possible element regardless of relevance, achieving perfect recall while providing useless responses. Or consider a grader checking output length as a proxy for thoroughness; the model learns to pad responses with filler rather than adding substantive content.  
The challenge is insidious because reward hacking can be subtle. A model might learn to use specific phrasings that correlate with high scores in your training data without actually understanding why those phrasings were appropriate. It might learn to hedge every answer in ways that technically satisfy safety criteria while being unhelpful. It might produce verbose explanations that score well on "thoroughness" metrics while obscuring the actual answer.  
Prevention starts with grader design. Continuous scores provide better learning signals than binary pass/fail. If a response can be partially correct, your grader should reflect that with partial credit. Multi-dimensional scoring helps ensure that optimizing one metric doesn't come at catastrophic cost to others. If you care about both accuracy and conciseness, measure both and combine them: a model that games one dimension will be penalized on the other.  
Testing your grader before training is essential. This sounds obvious but is frequently skipped. Before launching an expensive training run, evaluate your grader on synthetic examples:  
- Clearly correct responses should score high  
- Clearly incorrect responses should score low  
- Partial successes should score in between  
- Gaming attempts (padding, repetition, format tricks) should not score well  
If your grader can be fooled by simple adversarial examples you construct manually, the model will find those exploits during training.  
The Azure documentation recommends using the same grader for evaluation before training. Run the baseline model against your validation data with your intended grader. If the scores don't align with your intuitive assessment of quality, revise the grader before proceeding.

## Data Format for RFT  
RFT training data uses the chat completions message format with specific constraints. Each example contains a `messages` array with the conversation context, but unlike SFT, you do not include an assistant response. Instead, you include fields that your grader will use for evaluation.  
```json{  "messages": [    {"role": "user", "content": "User query requiring model response..."}  ],  "reference_answer": "The expected output or key for grading",  "additional_field": "Any other data your grader needs"}```  
The final message in the array must have the `user` role: this is the prompt the model will respond to during training. An important distinction for RFT: system messages are not supported. For reasoning models like o4-mini, use the `developer` role if you need to provide instructions that frame the task:  
```json{  "messages": [    {"role": "developer", "content": "Task instructions..."},    {"role": "user", "content": "User query..."}  ],  "reference_answer": "..."}```  
The `developer` role replaces the `system` role for reasoning models and takes precedence over user instructions, but remember that for RFT specifically, the traditional `system` role is not accepted.  
Your grader accesses this data through template substitution. The `{{sample.output_text}}` variable contains the model's generated response. Variables under the `{{item.*}}` namespace contain fields from your training data. If you include `"expected_tools": ["a", "b", "c"]` in your training example, your grader can reference it as `{{item.expected_tools}}`.  
For complex reference data, you have two options. You can add multiple top-level fields to your training examples, each accessible via `{{item.field_name}}`. Alternatively, you can encode structured data as a JSON string in a single field and parse it within a Python grader:  
```pythondef grade(sample, item):    import json    reference = json.loads(item.get("reference_answer", "{}"))    expected = reference.get("expected_values", [])    # Continue with scoring logic```  
Both training and validation datasets follow the same format. Azure requires both for RFT jobs because the validation set enables monitoring of generalization during training.

## Hyperparameters for RFT on Azure
RFT introduces hyperparameters beyond those used in supervised fine-tuning: `reasoning_effort`, `compute_multiplier`, `eval_samples`, and `eval_interval`. All default to `auto` (except `reasoning_effort` which defaults to `medium`), with the service determining optimal values based on your training data. Refer to [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning) for detailed descriptions and valid ranges.  
The metrics to monitor during training include mean reward (should increase over time), validation reward (should track with training reward: divergence suggests overfitting or reward hacking), and parse error rate (frequent parse errors indicate format mismatches between model outputs and grader expectations).  
Azure automatically saves checkpoints during training. The final checkpoint is not always the best (earlier checkpoints sometimes generalize better). Evaluate multiple checkpoints on held-out test data before selecting one for deployment.  
A note on costs: the fine-tuning service includes safeguards to prevent runaway costs during experimentation. Training jobs are automatically paused after reaching $5,000 in total training costs (training + grading), at which point you can deploy the most recent checkpoint or resume training. If resumed, billing continues with no further cost-based limits. Check the [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning) for current policies.

## A Practical Workflow
Effective RFT follows a pattern that balances iteration speed against compute costs.  
**Establish a baseline.** Before any fine-tuning, measure the base model's performance on your task using the grader you plan to use for training. This gives you a target to beat and reveals whether RFT is likely to help. If the base model already achieves near-perfect scores, RFT has limited room to improve. If the base model scores near zero, RFT cannot help meaning the model needs some success cases to reinforce.  
**Iterate on grader design.** Use a small subset of your data (10 to 20 examples) and run quick training jobs with 1 epoch (maximum 2). These complete in under an hour and cost relatively little. Examine the results: 
Are rewards increasing? 
Do high-scoring outputs actually look good when you read them? 
Can you find easy exploits in your grader? 
Refine the grader based on what you observe. This iteration is cheap and saves expensive mistakes in full training runs.  
**Run full training.** Once your grader is working as intended, train on your complete dataset for more epochs. Monitor training and validation curves. If validation reward stops improving while training reward continues to climb, you may be overfitting or reward hacking (consider stopping and using an earlier checkpoint).  
**Evaluate multiple checkpoints.** Don't assume the final checkpoint is optimal. Test 2-3 checkpoints from different points in training against a held-out test set that wasn't used for training or validation. Select the checkpoint with the best test performance.  
**Deploy incrementally.** Start with a small fraction of production traffic routed to the fine-tuned model. Monitor for unexpected behaviors meaning cases where the fine-tuned model performs worse than baseline, or produces outputs that users react negatively to. Scale up gradually as confidence builds.  
This workflow treats fine-tuning as an iterative process rather than a one-time project. As your application evolves and you encounter new failure modes, you can collect additional training data and run new fine-tuning jobs. The infrastructure Azure provides makes this iteration practical.  
---  
*Part 2 of this guide applies these fundamentals to multi-agent orchestration, demonstrating how RFT can improve a planner component within a larger agent system.*  