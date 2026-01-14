import pandas as pd
from typing import List


def get_openai_token_usage_from_response_metadata(response_metadata: dict) -> dict:
    """
    Extract token usage information from response metadata for OpenAI models.

    Args:
        response_metadata: A dictionary containing detailed response metadata.
    Returns:
        A dictionary with keys 'input_tokens', 'output_tokens', and 'total_tokens'.
    """
    usage = response_metadata["usage"]
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"]
    }


def get_litellm_token_usage_from_response_metadata(response_metadata: dict) -> dict:
    """
    Extract token usage information from response metadata for LiteLLM API.

    Args:
        response_metadata: A dictionary containing detailed response metadata.
    Returns:
        A dictionary with keys 'input_tokens', 'output_tokens', and 'total_tokens'.
    """
    usage = response_metadata["model_extra"]["usage"]
    return {
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"]
    }


def get_vllm_token_usage_from_response_metadata(response_metadata: dict) -> dict:
    """
    Extract token usage information from response metadata for VLLM models.

    Args:
        response_metadata: A dictionary containing detailed response metadata.
    Returns:
        A dictionary with keys 'input_tokens', 'output_tokens', and 'total_tokens'.
    """
    input_tokens = len(response_metadata['prompt_token_ids'])
    output_tokens = len(response_metadata['outputs'][0]["token_ids"])
    total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


def get_token_usage_from_response_metadata(response_metadata: dict, model_type: str) -> dict:
    """
    Extract token usage information from response metadata based on model type.

    Args:
        response_metadata: A dictionary containing detailed response metadata.
        model_type: Type of the model ('openai', 'litellm', 'vllm').
    Returns:
        A dictionary with keys 'input_tokens', 'output_tokens', and 'total_tokens'.
    """
    if model_type == "openai":
        return get_openai_token_usage_from_response_metadata(response_metadata)
    elif model_type == "litellm":
        return get_litellm_token_usage_from_response_metadata(response_metadata)
    
    try:
        return get_vllm_token_usage_from_response_metadata(response_metadata)
    except Exception as e:
        raise ValueError(f"Unsupported model type or invalid response metadata: {model_type}") from e


def append_or_load_json_output_fp_with_token_usage(json_fp: str) -> pd.DataFrame:
    """
    Append token usage information to a DataFrame based on response metadata.

    Args:
        json_fp: File path to the JSON file containing response metadata.
    Returns:
        DataFrame with additional columns for token usage.
    """    
    fn = json_fp.split("/")[-1]
    if "gpt-" in fn and "gpt-oss" not in fn:
        model_type = "openai"
    elif "deepseek-reasoner" in fn or "deepseek-chat" in fn:
        model_type = "litellm"
    else:
        model_type = "vllm"

    df = pd.read_json(json_fp)

    if "response_detailed" not in df.columns:
        print(f"No 'response_detailed' column in {json_fp}. Skipping...")
        return

    if "input_tokens" in df.columns and df["input_tokens"].notnull().all():
        print(f"Token usage columns already exist in {json_fp}. Skipping...")
        return df

    token_usages = df["response_detailed"].apply(
        lambda rd: get_token_usage_from_response_metadata(rd, model_type)
    )
    df["input_tokens"] = token_usages.apply(lambda x: x["input_tokens"])
    df["output_tokens"] = token_usages.apply(lambda x: x["output_tokens"])
    df["total_tokens"] = token_usages.apply(lambda x: x["total_tokens"])

    df.to_json(json_fp, orient="records", lines=False, indent=4)

    return df


def get_queries_from_response_detailed(response_detailed: dict) -> List[str]:
    queries = []
    for item in response_detailed["output"]:
        if item["type"] == "web_search_call":
            queries.append(item["action"]["query"])
    return queries


def get_cited_urls_from_response_detailed(response_detailed: dict) -> List[str]:
    urls = []
    
    # 1. Iterate through the items in the output array
    for item in response_detailed.get("output", []):
        
        # 2. Look specifically for the model's message
        if item.get("type") == "message":
            # 3. Check the content parts (usually there is only one)
            for content in item.get("content", []):
                # 4. Check for the annotations list
                annotations = content.get("annotations", [])
                
                for annotation in annotations:
                    # 5. Only extract 'url_citation' types
                    if annotation.get("type") == "url_citation":
                        url = annotation.get("url")
                        if url and url not in urls:
                            urls.append(url)
                            
    return urls