"""
Azure OpenAI deployment management utilities.

Provides functions for listing, deploying, and deleting fine-tuned model deployments.
"""

import warnings
import logging
from typing import List, Dict, Optional, Tuple

import requests
from azure.identity import AzureCliCredential

from src.settings import SUBSCRIPTION_ID, RESOURCE_GROUP, RESOURCE_NAME


# Suppress Azure credential warnings
logging.getLogger("azure.identity").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_azure_credentials() -> Tuple[dict, str]:
    """
    Get Azure credentials and base URL for management API.

    Returns:
        Tuple of (headers dict, base_url string)
    """
    credential = AzureCliCredential(process_timeout=30)
    token = credential.get_token("https://management.azure.com/.default").token
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    base_url = (
        f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
        f"/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.CognitiveServices/accounts/{RESOURCE_NAME}"
    )
    
    return headers, base_url


def list_finetuned_deployments() -> List[Dict]:
    """
    List all fine-tuned model deployments.
    
    Returns:
        List of deployment dicts with name, model, and capacity
    """
    headers, base_url = get_azure_credentials()
    
    r = requests.get(
        f"{base_url}/deployments?api-version=2023-05-01",
        headers=headers
    )
    
    deployments = r.json().get("value", [])
    
    # Filter to only fine-tuned models
    finetuned = [
        d for d in deployments 
        if ".ft-" in d["properties"]["model"]["name"]
    ]
    
    return finetuned


def print_deployments(deployments: List[Dict]) -> None:
    """Print deployment list in a formatted way."""
    print(f"📋 Current fine-tuned deployments ({len(deployments)}):")
    
    for i, d in enumerate(deployments):
        name = d["name"]
        model = d["properties"]["model"]["name"]
        capacity = d["sku"]["capacity"]
        print(f"   {i+1}. {name} → {model} ({capacity}K TPM)")
    
    total_tpm = sum(d["sku"]["capacity"] for d in deployments)
    print(f"\n📊 Quota: {total_tpm}K / 500K TPM")


def check_if_deployed(target_model: str, deployments: List[Dict] = None) -> Optional[str]:
    """
    Check if a model is already deployed.
    
    Args:
        target_model: The fine-tuned model name to check
        deployments: Optional list of deployments (fetched if not provided)
    
    Returns:
        Deployment name if deployed, None otherwise
    """
    if deployments is None:
        deployments = list_finetuned_deployments()
    
    for d in deployments:
        if d["properties"]["model"]["name"] == target_model:
            return d["name"]
    
    return None


def deploy_model(
    model_name: str,
    deployment_name: str,
    capacity: int = None
) -> bool:
    """
    Deploy a fine-tuned model.

    Args:
        model_name: The fine-tuned model name (e.g., "o4-mini-2025-04-16.ft-...")
        deployment_name: Name for the deployment (e.g., "planner-1210-2015")
        capacity: TPM capacity in thousands. If None, uses all available quota.

    Returns:
        True if deployment successful, False otherwise
    """
    headers, base_url = get_azure_credentials()

    # Use available quota if not specified
    if capacity is None:
        capacity = get_available_quota()
        if capacity <= 0:
            print(f"No quota available! Resize or delete existing deployments first.")
            return False

    print(f"Deploying {model_name} as '{deployment_name}' with {capacity}K TPM...")

    r = requests.put(
        f"{base_url}/deployments/{deployment_name}?api-version=2023-05-01",
        headers=headers,
        json={
            "sku": {"name": "Standard", "capacity": capacity},
            "properties": {
                "model": {
                    "format": "OpenAI",
                    "name": model_name,
                    "version": "1"
                }
            }
        }
    )

    if r.status_code in [200, 201]:
        print(f"Deployed successfully!")
        return True
    else:
        print(f"Failed: {r.text}")
        return False


def delete_deployment(deployment_name: str) -> bool:
    """
    Delete a deployment.
    
    Args:
        deployment_name: Name of the deployment to delete
    
    Returns:
        True if deletion successful, False otherwise
    """
    headers, base_url = get_azure_credentials()
    
    print(f"🗑️ Deleting: {deployment_name}")
    
    r = requests.delete(
        f"{base_url}/deployments/{deployment_name}?api-version=2023-05-01",
        headers=headers
    )
    
    if r.status_code in [200, 202, 204]:
        print(f"✅ Deleted! Wait 60s before deploying a new model.")
        return True
    else:
        print(f"❌ Failed: {r.text}")
        return False


def get_deployment_status(deployments: List[Dict], target_model: str) -> str:
    """
    Get the deployment status for a target model.

    Args:
        deployments: List of current deployments
        target_model: The model to check

    Returns:
        Status string: "deployed", "quota_available", or "quota_full"
    """
    existing = check_if_deployed(target_model, deployments)

    if existing:
        return "deployed"

    current_tpm = sum(d["sku"]["capacity"] for d in deployments)

    if current_tpm < 500:
        return "quota_available"
    else:
        return "quota_full"


# Maximum TPM quota for fine-tuned models
MAX_QUOTA = 500


def get_available_quota(deployments: List[Dict] = None) -> int:
    """
    Get available TPM quota for new deployments.

    Args:
        deployments: Optional list of deployments (fetched if not provided)

    Returns:
        Available quota in K TPM (e.g., 250 means 250K TPM available)
    """
    if deployments is None:
        deployments = list_finetuned_deployments()

    used = sum(d["sku"]["capacity"] for d in deployments)
    return MAX_QUOTA - used


def update_deployment_capacity(deployment_name: str, new_capacity: int) -> bool:
    """
    Update the TPM capacity of an existing deployment.

    Use this to resize a deployment and free up quota for other models.

    Args:
        deployment_name: Name of the deployment to resize
        new_capacity: New capacity in K TPM (e.g., 250 for 250K TPM)

    Returns:
        True if update successful, False otherwise
    """
    headers, base_url = get_azure_credentials()

    print(f"Resizing '{deployment_name}' to {new_capacity}K TPM...")

    # Use PATCH to update only the capacity
    r = requests.patch(
        f"{base_url}/deployments/{deployment_name}?api-version=2023-05-01",
        headers=headers,
        json={
            "sku": {"name": "Standard", "capacity": new_capacity}
        }
    )

    if r.status_code in [200, 201, 202]:
        print(f"Resized successfully!")
        return True
    else:
        print(f"Failed: {r.text}")
        return False
