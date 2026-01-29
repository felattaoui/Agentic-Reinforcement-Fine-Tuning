"""
Azure OpenAI content filter management.

Creates a custom RAI policy that disables jailbreak detection for evaluation.
This is needed because tau-bench prompts can trigger false positive jailbreak detections.
"""

import re
import warnings
import logging

from azure.identity import AzureCliCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

from src.settings import AZURE_ENDPOINT, SUBSCRIPTION_ID, RESOURCE_GROUP


# Suppress Azure warnings
logging.getLogger("azure.identity").setLevel(logging.ERROR)
logging.getLogger("azure.mgmt").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Default policy name
RAI_POLICY_NAME = "no-jailbreak-filter"


def get_account_name() -> str:
    """Extract account name from Azure endpoint."""
    match = re.search(
        r"https://(.+?)\.(openai\.azure\.com|cognitiveservices\.azure\.com)",
        AZURE_ENDPOINT
    )
    if not match:
        raise ValueError(f"Unable to extract account name from: {AZURE_ENDPOINT}")
    return match.group(1)


def create_no_jailbreak_filter(policy_name: str = RAI_POLICY_NAME) -> bool:
    """
    Create a content filter that disables jailbreak detection.
    
    This filter:
    - ✅ Keeps all standard safety filters (Hate, Sexual, Violence, Self-harm)
    - ❌ Disables only jailbreak detection
    
    Args:
        policy_name: Name for the RAI policy (default: "no-jailbreak-filter")
    
    Returns:
        True if creation successful, False otherwise
    
    Note:
        This configuration is for benchmark purposes only.
        Production deployments should keep safety filters enabled.
    """
    account_name = get_account_name()
    
    print(f"Configuration:")
    print(f"   Account Name: {account_name}")
    print(f"   Resource Group: {RESOURCE_GROUP}")
    
    # Create management client
    client = CognitiveServicesManagementClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION_ID,
    )
    
    # Define content filters
    content_filters = [
        # Standard safety filters - ENABLED
        {"name": "Hate", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Prompt"},
        {"name": "Hate", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Completion"},
        {"name": "Sexual", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Prompt"},
        {"name": "Sexual", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Completion"},
        {"name": "Selfharm", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Prompt"},
        {"name": "Selfharm", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Completion"},
        {"name": "Violence", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Prompt"},
        {"name": "Violence", "enabled": True, "blocking": True, "severityThreshold": "Medium", "source": "Completion"},
        
        # Jailbreak detection - DISABLED (causes false positives with tau-bench)
        {"name": "Jailbreak", "enabled": False, "blocking": False, "source": "Prompt"},
        
        # Protected material filters - ENABLED
        {"name": "Protected Material Text", "enabled": True, "blocking": True, "source": "Completion"},
        {"name": "Protected Material Code", "enabled": True, "blocking": True, "source": "Completion"},
    ]
    
    try:
        response = client.rai_policies.create_or_update(
            resource_group_name=RESOURCE_GROUP,
            account_name=account_name,
            rai_policy_name=policy_name,
            rai_policy={
                "properties": {
                    "basePolicyName": "Microsoft.Default",
                    "mode": "Asynchronous_filter",
                    "contentFilters": content_filters,
                }
            },
        )
        
        print(f"\n✅ Content filter '{policy_name}' created successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to create content filter: {e}")
        return False


def get_policy_header(policy_name: str = RAI_POLICY_NAME) -> dict:
    """
    Get the HTTP header to use a custom RAI policy.

    Args:
        policy_name: Name of the RAI policy

    Returns:
        Dict with x-policy-id header

    Usage:
        response = client.responses.create(
            model=model_name,
            input=content,
            extra_headers=get_policy_header()
        )
    """
    return {"x-policy-id": policy_name}


def apply_rai_policy_to_deployment(deployment_name: str, policy_name: str = RAI_POLICY_NAME) -> bool:
    """
    Attach a RAI policy to an existing deployment.

    This is needed because Azure AI Evaluation SDK doesn't support extra_headers.
    The policy must be attached to the deployment itself.

    Args:
        deployment_name: Name of the deployment (e.g., "gpt-4.1-mini")
        policy_name: Name of the RAI policy (default: "no-jailbreak-filter")

    Returns:
        True if successful, False otherwise

    Note:
        The RAI policy must already exist (call create_no_jailbreak_filter() first).
    """
    import requests
    from src.evaluation.deployment import get_azure_credentials

    headers, base_url = get_azure_credentials()

    print(f"Attaching RAI policy '{policy_name}' to deployment '{deployment_name}'...")

    r = requests.patch(
        f"{base_url}/deployments/{deployment_name}?api-version=2024-10-01",
        headers=headers,
        json={
            "properties": {
                "raiPolicyName": policy_name
            }
        }
    )

    if r.status_code in [200, 201, 202]:
        print(f"✅ RAI policy '{policy_name}' attached to '{deployment_name}'")
        return True
    else:
        print(f"❌ Failed to attach RAI policy: {r.status_code} - {r.text}")
        return False
