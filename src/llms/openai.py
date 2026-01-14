from openai import OpenAI
from openai.types.responses.response import Response as OpenAIResponseObject

from typing import Tuple

client = OpenAI()


def get_response(prompt: str,
                 model_name: str = "gpt-5-nano-2025-08-07", 
                 experiment_condition: str = "no_tool_use", 
                 **kwargs) -> Tuple[OpenAIResponseObject, str]:    

    tool_map = {
        "no_tool_use": [],
        "with_web_search": [{"type": "web_search"}],
        "with_code_interpreter": [
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "memory_limit": "1g"}
            }
        ],
        "with_web_search_and_code_interpreter": [
            {"type": "web_search"},
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "memory_limit": "1g"}
            }
        ],
    }

    tools = tool_map.get(experiment_condition)

    instructions = None
    if "code_interpreter" in experiment_condition:
        instructions = (
            "You have access to a code interpreter tool that can execute Python code. "
            "Use this tool to perform calculations, data analysis, or any other tasks "
            "that require code execution to help you answer the user's query."
        )
    
    if tools is None:
        print(f"Unrecognized experiment_condition: {experiment_condition}. "
              "Defaulting to no tool use.")
        tools = []

    tool_choice = "none" if not tools else "required"

    response = client.responses.create(
        model=model_name,
        tools=tools,
        input=prompt,
        instructions=instructions,
        tool_choice=tool_choice,
        **kwargs
    )
    output_text = getattr(response, "output_text", None)
    return response, output_text