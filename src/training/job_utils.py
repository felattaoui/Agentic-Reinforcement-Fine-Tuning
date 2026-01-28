"""
Utilities for managing RFT training jobs on Azure OpenAI.
"""

import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from openai import OpenAI

from src.settings import OUTPUTS_DIR, ROOT_DIR


def wait_for_file(client: OpenAI, file_id: str, timeout: int = 300) -> bool:
    """
    Wait for a file to be processed by Azure OpenAI.
    
    Args:
        client: OpenAI client instance
        file_id: ID of the uploaded file
        timeout: Maximum seconds to wait
        
    Returns:
        bool: True if file processed successfully, False otherwise
    """
    start = time.time()
    while time.time() - start < timeout:
        f = client.files.retrieve(file_id)
        print(f"  {file_id}: {f.status}")
        
        if f.status == "processed":
            return True
        if f.status == "error":
            print(f"  Error: {f.status_details}")
            return False
        
        time.sleep(5)
    
    print(f"  Timeout waiting for {file_id}")
    return False


def monitor_job(client: OpenAI, job_id: str, interval: int = 60) -> Optional[object]:
    """
    Monitor an RFT training job until completion.
    
    Args:
        client: OpenAI client instance
        job_id: Fine-tuning job ID
        interval: Seconds between status checks
        
    Returns:
        The completed job object, or None if interrupted
    """
    print(f"📊 Monitoring {job_id}")
    print("   Press Ctrl+C to stop (job continues on Azure)")
    
    start = time.time()
    last_status = None
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            
            if job.status != last_status:
                elapsed = int(time.time() - start) // 60
                print(f"\n[{elapsed}m] {job.status}")
                last_status = job.status
            else:
                print(".", end="", flush=True)
            
            if job.status in ["succeeded", "failed", "cancelled"]:
                print(f"\n🏁 {job.status}")
                if job.status == "succeeded":
                    print(f"   Model: {job.fine_tuned_model}")
                return job
                
        except Exception as e:
            print(f"\n⚠️ Connection error: {e}")
        
        time.sleep(interval)


def save_job_history(
    job_id: str,
    suffix: str,
    train_file_id: str,
    val_file_id: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    hyperparameters: dict,
    grader_name: str = "planner_grader"
) -> Path:
    """
    Save job information to history file.
    
    Args:
        job_id: Fine-tuning job ID
        suffix: Model suffix (e.g., "planner-0602-1430")
        train_file_id: Training file ID
        val_file_id: Validation file ID
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        hyperparameters: Training hyperparameters
        grader_name: Name of the grader used
        
    Returns:
        Path to the saved job history file
    """
    job_info = {
        "job_id": job_id,
        "created": datetime.now().isoformat(),
        "suffix": suffix,
        "train_file": train_file_id,
        "val_file": val_file_id,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "grader": grader_name,
        "hyperparameters": hyperparameters
    }
    
    history_path = OUTPUTS_DIR / "job_history.json"
    with open(history_path, "w") as f:
        json.dump(job_info, f, indent=2)
    
    print(f"💾 Saved to {history_path}")
    return history_path


def load_job_history() -> Optional[dict]:
    """
    Load job history from file.
    
    Returns:
        Job history dict, or None if not found
    """
    history_path = OUTPUTS_DIR / "job_history.json"
    
    if not history_path.exists():
        print(f"⚠️ No job history found at {history_path}")
        return None
    
    with open(history_path) as f:
        return json.load(f)


def update_job_history(updates: dict) -> None:
    """
    Update the job history file with new information.
    
    Args:
        updates: Dictionary of fields to update
    """
    history = load_job_history()
    if history is None:
        print("⚠️ Cannot update - no job history found")
        return
    
    history.update(updates)
    
    history_path = OUTPUTS_DIR / "job_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"💾 Updated {history_path}")


def update_env_file(deployment_name: str) -> None:
    """
    Update the .env file with the new fine-tuned deployment name.
    
    Args:
        deployment_name: Name of the fine-tuned deployment
    """
    env_path = ROOT_DIR / ".env"
    
    if not env_path.exists():
        print(f"⚠️ .env file not found at {env_path}")
        return
    
    env_content = env_path.read_text()
    
    # Update or add FINETUNED_DEPLOYMENT
    if "FINETUNED_DEPLOYMENT=" in env_content:
        env_content = re.sub(
            r'FINETUNED_DEPLOYMENT=.*',
            f'FINETUNED_DEPLOYMENT={deployment_name}',
            env_content
        )
    else:
        env_content += f'\nFINETUNED_DEPLOYMENT={deployment_name}\n'
    
    env_path.write_text(env_content)
    print(f"✅ Updated .env: FINETUNED_DEPLOYMENT={deployment_name}")


def finalize_successful_job(client: OpenAI, job_id: str) -> None:
    """
    Finalize a successful training job: update history and .env.

    Args:
        client: OpenAI client instance
        job_id: Fine-tuning job ID
    """
    job = client.fine_tuning.jobs.retrieve(job_id)

    if job.status != "succeeded":
        print(f"⚠️ Job status is {job.status}, not succeeded")
        return

    # Load current history to get suffix
    history = load_job_history()
    if history is None:
        return

    # Update history
    update_job_history({
        "status": "succeeded",
        "fine_tuned_model": job.fine_tuned_model,
        "completed": datetime.now().isoformat()
    })

    # Update .env
    update_env_file(history["suffix"])

    print(f"✅ Model: {job.fine_tuned_model}")
    print(f"\n🎉 Run 03_deployment.ipynb to deploy!")


def list_checkpoints(client: OpenAI, job_id: str) -> list:
    """
    List all checkpoints for a fine-tuning job.

    Each checkpoint is a fully functional model that can be deployed.
    Azure keeps the 3 most recent checkpoints (final + 2 previous epochs).

    Metrics are enriched from the job's result file (results.csv) when available,
    as the checkpoint API doesn't always return all metrics (especially for the final step).

    Args:
        client: OpenAI client instance
        job_id: Fine-tuning job ID

    Returns:
        List of checkpoint dicts with model name, step, metrics, and created_at
    """
    response = client.fine_tuning.jobs.checkpoints.list(job_id)

    checkpoints = []
    for cp in response.data:
        checkpoints.append({
            "id": cp.id,
            "model": cp.fine_tuned_model_checkpoint,
            "step": cp.step_number,
            "created_at": cp.created_at,
            "metrics": dict(cp.metrics) if cp.metrics else {}
        })

    # Sort by step descending (most recent first)
    checkpoints.sort(key=lambda x: x["step"], reverse=True)

    # Enrich metrics from result file (checkpoint API doesn't always have all metrics)
    checkpoints = _enrich_checkpoints_from_results(client, job_id, checkpoints)

    return checkpoints


def _enrich_checkpoints_from_results(client: OpenAI, job_id: str, checkpoints: list) -> list:
    """
    Enrich checkpoint metrics from the job's result file (results.csv).

    The checkpoint API sometimes lacks metrics (especially valid_reward for final step).
    The result file contains complete metrics for all steps.

    If a checkpoint lacks valid metrics, we use the last known value from a previous step.

    Args:
        client: OpenAI client instance
        job_id: Fine-tuning job ID
        checkpoints: List of checkpoint dicts to enrich

    Returns:
        Enriched checkpoints list
    """
    import csv
    from io import StringIO

    try:
        # Get job to find result file
        job = client.fine_tuning.jobs.retrieve(job_id)
        if not job.result_files:
            return checkpoints

        # Download result file
        result_file_id = job.result_files[0]
        content = client.files.content(result_file_id)
        csv_text = content.text

        # Parse CSV and track last known valid metrics
        reader = csv.DictReader(StringIO(csv_text))
        results_by_step = {}
        last_valid_reward = None
        last_valid_tokens = None

        for row in reader:
            step = int(float(row.get("step", 0)))
            results_by_step[step] = row

            # Track last step with valid reward
            valid_reward = row.get("full_valid_mean_reward")
            if valid_reward:
                last_valid_reward = float(valid_reward)
            valid_tokens = row.get("usage/samples/valid_reasoning_tokens_mean")
            if valid_tokens:
                last_valid_tokens = float(valid_tokens)

        # Enrich checkpoints with result file metrics
        for cp in checkpoints:
            step = cp["step"]
            if step in results_by_step:
                row = results_by_step[step]

                # Update train metrics if missing
                if not cp["metrics"].get("train_mean_reward"):
                    val = row.get("train_mean_reward")
                    if val:
                        cp["metrics"]["train_mean_reward"] = float(val)

                if not cp["metrics"].get("usages/samples/train_reasoning_tokens_mean"):
                    val = row.get("usage/samples/train_reasoning_tokens_mean")
                    if val:
                        cp["metrics"]["usages/samples/train_reasoning_tokens_mean"] = float(val)

                # Update valid metrics: use actual value if present, else last known
                if not cp["metrics"].get("full_valid_mean_reward"):
                    val = row.get("full_valid_mean_reward")
                    if val:
                        cp["metrics"]["full_valid_mean_reward"] = float(val)
                    elif last_valid_reward:
                        cp["metrics"]["full_valid_mean_reward"] = last_valid_reward
                        cp["metrics"]["_valid_reward_estimated"] = True

                if not cp["metrics"].get("usages/samples/valid_reasoning_tokens_mean"):
                    val = row.get("usage/samples/valid_reasoning_tokens_mean")
                    if val:
                        cp["metrics"]["usages/samples/valid_reasoning_tokens_mean"] = float(val)
                    elif last_valid_tokens:
                        cp["metrics"]["usages/samples/valid_reasoning_tokens_mean"] = last_valid_tokens
                        cp["metrics"]["_valid_tokens_estimated"] = True

    except Exception as e:
        # Silently fail - metrics from checkpoint API are still available
        pass

    return checkpoints


def print_checkpoints(checkpoints: list) -> None:
    """
    Display checkpoints in a formatted table with RFT metrics.

    Shows reward (train/valid) and reasoning tokens (train/valid) for each checkpoint,
    plus indicators for best valid reward, lowest train/valid gap, and lowest token usage.

    Args:
        checkpoints: List of checkpoint dicts from list_checkpoints()
    """
    if not checkpoints:
        print("⚠️ No checkpoints available")
        return

    print("\n📋 Available checkpoints:")
    print("─" * 90)
    print(f"  {'Idx':<5} {'Step':<6} {'Created':<18} {'Reward (train/valid)':<24} {'Reasoning tokens (train/valid)'}")
    print("─" * 90)

    # Track best values for indicators (lists for ties)
    best_valid_reward = {"indices": [], "value": -1}
    best_gap = {"indices": [], "value": float("inf")}
    lowest_tokens = {"indices": [], "value": float("inf")}

    for i, cp in enumerate(checkpoints):
        created = datetime.fromtimestamp(cp["created_at"]).strftime("%Y-%m-%d %H:%M")
        metrics = cp["metrics"]

        # Extract RFT metrics (Azure uses different key names)
        train_reward = metrics.get("train_mean_reward", 0)
        valid_reward = metrics.get("full_valid_mean_reward", 0)
        train_tokens = metrics.get("usages/samples/train_reasoning_tokens_mean", 0)
        valid_tokens = metrics.get("usages/samples/valid_reasoning_tokens_mean", 0)

        # Check if values are estimated (from previous step)
        valid_reward_estimated = metrics.get("_valid_reward_estimated", False)
        valid_tokens_estimated = metrics.get("_valid_tokens_estimated", False)

        # Format reward string (show — for missing, ~ prefix for estimated)
        train_r_str = f"{train_reward:.3f}" if train_reward else "—"
        if valid_reward:
            valid_r_str = f"~{valid_reward:.3f}" if valid_reward_estimated else f"{valid_reward:.3f}"
        else:
            valid_r_str = "—"
        reward_str = f"{train_r_str} / {valid_r_str}"

        # Format tokens string (show — for missing, ~ prefix for estimated)
        train_t_str = f"{int(train_tokens)}" if train_tokens else "—"
        if valid_tokens:
            valid_t_str = f"~{int(valid_tokens)}" if valid_tokens_estimated else f"{int(valid_tokens)}"
        else:
            valid_t_str = "—"
        tokens_str = f"{train_t_str} / {valid_t_str}"

        print(f"  [{i}]   {cp['step']:<6} {created:<18} {reward_str:<24} {tokens_str}")

        # Track indicators (handle ties)
        if valid_reward > best_valid_reward["value"]:
            best_valid_reward = {"indices": [i], "value": valid_reward}
        elif valid_reward == best_valid_reward["value"] and valid_reward > 0:
            best_valid_reward["indices"].append(i)

        if train_reward > 0 and valid_reward > 0:
            gap = abs(train_reward - valid_reward) / train_reward * 100
            if gap < best_gap["value"]:
                best_gap = {"indices": [i], "value": gap}
            elif gap == best_gap["value"]:
                best_gap["indices"].append(i)

        if valid_tokens > 0:
            if valid_tokens < lowest_tokens["value"]:
                lowest_tokens = {"indices": [i], "value": valid_tokens}
            elif valid_tokens == lowest_tokens["value"]:
                lowest_tokens["indices"].append(i)

    print("─" * 90)

    # Check if any values were estimated
    has_estimated = any(
        cp.get("metrics", {}).get("_valid_reward_estimated") or
        cp.get("metrics", {}).get("_valid_tokens_estimated")
        for cp in checkpoints
    )
    if has_estimated:
        print("   ~ = estimated from previous eval step")

    # Print indicators (showing ties)
    print("\n📊 Indicators:")
    if best_valid_reward["value"] > 0:
        indices = best_valid_reward["indices"]
        steps_str = ", ".join(f"[{i}] step {checkpoints[i]['step']}" for i in indices)
        print(f"   • Highest valid reward:       {steps_str} ({best_valid_reward['value']:.3f})")

    if best_gap["value"] < float("inf"):
        indices = best_gap["indices"]
        steps_str = ", ".join(f"[{i}] step {checkpoints[i]['step']}" for i in indices)
        print(f"   • Lowest train/valid gap:     {steps_str} ({best_gap['value']:.1f}%)")

    if lowest_tokens["value"] < float("inf"):
        indices = lowest_tokens["indices"]
        steps_str = ", ".join(f"[{i}] step {checkpoints[i]['step']}" for i in indices)
        print(f"   • Lowest reasoning tokens:    {steps_str} ({int(lowest_tokens['value'])} valid)")

    print()


def select_checkpoint(checkpoints: list, index: int) -> Optional[str]:
    """
    Select a checkpoint model to deploy.

    Args:
        checkpoints: List of checkpoint dicts from list_checkpoints()
        index: Index of the checkpoint to select (0 = first/most recent)

    Returns:
        Model name to deploy, or None if invalid selection
    """
    if not checkpoints:
        print("❌ No checkpoints available")
        return None

    if index < 0 or index >= len(checkpoints):
        print(f"❌ Invalid index. Valid range: 0-{len(checkpoints) - 1}")
        return None

    selected = checkpoints[index]
    print(f"✅ Selected checkpoint step {selected['step']}: {selected['model']}")
    return selected["model"]
