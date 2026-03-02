"""
Azure OpenAI client initialization and utilities.

Provides a pre-configured OpenAI client for Azure OpenAI endpoints.
"""

from openai import OpenAI, AsyncOpenAI
from src.settings import AZURE_ENDPOINT, AZURE_TOKEN_PROVIDER, AZURE_DEPLOYMENT


def get_client() -> OpenAI:
    """
    Get a synchronous OpenAI client configured for Azure (Entra ID auth).

    Returns:
        OpenAI: Configured client instance
    """
    return OpenAI(
        api_key=AZURE_TOKEN_PROVIDER,
        base_url=f"{AZURE_ENDPOINT}/openai/v1/"
    )


def get_async_client() -> AsyncOpenAI:
    """
    Get an asynchronous OpenAI client configured for Azure (Entra ID auth).

    Returns:
        AsyncOpenAI: Configured async client instance
    """
    return AsyncOpenAI(
        api_key=AZURE_TOKEN_PROVIDER(),
        base_url=f"{AZURE_ENDPOINT}/openai/v1/"
    )


def test_connection(client: OpenAI = None, deployment: str = None) -> bool:
    """
    Test the connection to Azure OpenAI.
    
    Args:
        client: Optional client to use (creates one if not provided)
        deployment: Optional deployment name (uses default if not provided)
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    if client is None:
        client = get_client()
    if deployment is None:
        deployment = AZURE_DEPLOYMENT
    
    try:
        response = client.responses.create(
            model=deployment,
            input="Say OK"
        )
        print(f"✅ Connection OK - {deployment}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
