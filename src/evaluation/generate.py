"""
Async response generation for model evaluation.

Generates responses from Azure OpenAI models in parallel using asyncio.
"""

import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
from openai import AsyncOpenAI

from src.settings import AZURE_ENDPOINT, AZURE_TOKEN_PROVIDER, PLANNER_RESPONSES_DIR
from src.evaluation.content_filter import get_policy_header


def get_async_client() -> AsyncOpenAI:
    """Get an async OpenAI client configured for Azure (Entra ID auth)."""
    return AsyncOpenAI(
        api_key=AZURE_TOKEN_PROVIDER(),
        base_url=f"{AZURE_ENDPOINT}/openai/v1/"
    )


async def generate_response_async(
    client: AsyncOpenAI,
    model_name: str,
    sample: Dict,
    idx: int,
    semaphore: asyncio.Semaphore,
    reasoning_effort: str = None,
    response_format: dict = None
) -> Dict:
    """
    Generate a single response using the Responses API (async).

    Args:
        client: Async OpenAI client
        model_name: Model deployment name
        sample: Sample dict with messages and reference_answer
        idx: Sample index (for error reporting)
        semaphore: Concurrency limiter
        reasoning_effort: Optional reasoning effort level
        response_format: Optional JSON schema for structured output (Fine-tuning API format)

    Returns:
        Dict with response, expected_tools, and query
    """
    # Parse reference answer
    ref = sample["reference_answer"]
    if isinstance(ref, str):
        ref = json.loads(ref)
    expected_tools = ref.get("expected_tools", [])
    
    # Build input from messages
    messages = sample["messages"]
    input_content = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
    
    async with semaphore:
        try:
            kwargs = {
                "model": model_name,
                "input": input_content,
                "extra_headers": get_policy_header()
            }
            # Set reasoning effort if specified, or default to "none" for gpt-5.x
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            elif model_name.startswith("gpt-5"):
                kwargs["reasoning"] = {"effort": "none"}

            # Add structured output format if specified
            # Fine-tuning API format: {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}
            # Responses API format:   text={"format": {"type": "json_schema", "name": ..., "schema": ...}}
            if response_format is not None:
                # Extract json_schema and add type field for Responses API
                format_content = response_format["json_schema"].copy()
                format_content["type"] = "json_schema"
                kwargs["text"] = {"format": format_content}

            # Measure per-request latency
            request_start = time.time()
            response = await client.responses.create(**kwargs)
            latency_ms = (time.time() - request_start) * 1000

            text = response.output_text or ""

            # Capture token usage
            # reasoning_tokens is in output_tokens_details, not directly in usage
            output_details = getattr(response.usage, 'output_tokens_details', None)
            reasoning_tokens = 0
            if output_details:
                reasoning_tokens = getattr(output_details, 'reasoning_tokens', 0) or 0

            usage = {
                "input_tokens": getattr(response.usage, 'input_tokens', 0) or 0,
                "output_tokens": getattr(response.usage, 'output_tokens', 0) or 0,
                "reasoning_tokens": reasoning_tokens,
                "latency_ms": round(latency_ms, 0)
            }
        except Exception as e:
            text = ""
            usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "latency_ms": 0}
            if "content_filter" in str(e):
                text = "__CONTENT_FILTER_SKIPPED__"
                print(f"Skipped sample {idx} (content filter false positive, sample excluded from metrics)")
            else:
                print(f"Error sample {idx}: {e}")

    return {
        "response": text,
        "expected_tools": json.dumps(expected_tools),
        "query": messages[-1]["content"][:100],
        "usage": usage
    }


async def _generate_responses_async_impl(
    model_name: str,
    samples: List[Dict],
    max_concurrent: int = 5,
    reasoning_effort: str = None,
    output_suffix: str = None,
    response_format: dict = None
) -> Tuple[str, Dict]:
    """
    Internal implementation for async response generation.

    Args:
        model_name: Model deployment name
        samples: List of evaluation samples
        max_concurrent: Maximum concurrent requests
        reasoning_effort: Optional reasoning effort level ("none", "low", "medium", "high")
        output_suffix: Optional suffix for output filename
        response_format: Optional JSON schema for structured output (Fine-tuning API format)

    Returns:
        Tuple of (filepath, total_usage) where:
            - filepath: Path to the generated JSONL file
            - total_usage: Dict with aggregated token counts and sample count
    """
    client = get_async_client()
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_response_async(client, model_name, sample, i, semaphore, reasoning_effort, response_format)
        for i, sample in enumerate(samples)
    ]

    start_time = time.time()
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Generating {model_name}"
    ):
        result = await coro
        results.append(result)
    elapsed_time = time.time() - start_time
    
    # Aggregate token usage and timing
    latencies = [r["usage"]["latency_ms"] for r in results if r["usage"].get("latency_ms", 0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    total_usage = {
        "input_tokens": sum(r["usage"]["input_tokens"] for r in results),
        "output_tokens": sum(r["usage"]["output_tokens"] for r in results),
        "reasoning_tokens": sum(r["usage"]["reasoning_tokens"] for r in results),
        "total_tokens": sum(
            r["usage"]["input_tokens"] + r["usage"]["output_tokens"] + r["usage"]["reasoning_tokens"]
            for r in results
        ),
        "samples": len(results),
        "elapsed_seconds": round(elapsed_time, 2),
        "avg_latency_ms": round(avg_latency, 0)
    }

    # Save to JSONL
    safe_name = model_name.replace("-", "_").replace(".", "_")
    if output_suffix:
        safe_name = f"{safe_name}_{output_suffix}"
    filepath = PLANNER_RESPONSES_DIR / f"eval_{safe_name}.jsonl"

    with open(filepath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return str(filepath), total_usage


def generate_responses_async(
    model_name: str,
    samples: List[Dict],
    max_concurrent: int = 5
) -> str:
    """
    Generate responses from a model in parallel using asyncio.
    
    This function handles the event loop creation for Jupyter notebooks.
    
    Args:
        model_name: Model deployment name (e.g., "o4-mini" or "planner-1210-2015")
        samples: List of evaluation samples
        max_concurrent: Maximum concurrent requests (default: 5)
    
    Returns:
        Path to the generated JSONL file with responses
    
    Example:
        >>> baseline_file = generate_responses_async("o4-mini", samples)
        >>> finetuned_file = generate_responses_async("planner-1210-2015", samples)
    """
    # Check if we're in an existing event loop (Jupyter)
    try:
        loop = asyncio.get_running_loop()
        # We're in Jupyter, need to use nest_asyncio or await directly
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(_generate_responses_async_impl(
            model_name, samples, max_concurrent
        ))
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(_generate_responses_async_impl(
            model_name, samples, max_concurrent
        ))


async def generate_responses_for_notebook(
    model_name: str,
    samples: List[Dict],
    max_concurrent: int = 5,
    reasoning_effort: str = None,
    output_suffix: str = None,
    response_format: dict = None
) -> Tuple[str, Dict]:
    """
    Async version for use with 'await' in Jupyter notebooks.

    Use this when you can use 'await' directly:
        filepath, usage = await generate_responses_for_notebook("o4-mini", samples)
        filepath, usage = await generate_responses_for_notebook("gpt-5.2", samples, reasoning_effort="low", output_suffix="low")

    For structured output, pass the schema from load_planner_schema():
        from src.settings import load_planner_schema
        schema = load_planner_schema()
        filepath, usage = await generate_responses_for_notebook("o4-mini", samples, response_format=schema)

    Args:
        model_name: Model deployment name
        samples: List of evaluation samples
        max_concurrent: Maximum concurrent requests
        reasoning_effort: Optional reasoning effort ("none", "low", "medium", "high")
        output_suffix: Optional suffix for output filename (e.g., "low" -> eval_gpt_5_2_low.jsonl)
        response_format: Optional JSON schema for structured output (use load_planner_schema())

    Returns:
        Tuple of (filepath, total_usage) where:
            - filepath: Path to the generated JSONL file
            - total_usage: Dict with aggregated token counts and timing:
                - input_tokens: Total input tokens across all samples
                - output_tokens: Total output tokens across all samples
                - reasoning_tokens: Total reasoning tokens (for o-series, gpt-5.x)
                - total_tokens: Sum of all token types
                - samples: Number of samples processed
                - elapsed_seconds: Total wall-clock time for all requests
                - avg_latency_ms: Average latency per request in milliseconds
    """
    return await _generate_responses_async_impl(
        model_name, samples, max_concurrent, reasoning_effort, output_suffix, response_format
    )
