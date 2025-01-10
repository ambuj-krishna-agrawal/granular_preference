from router.gemini_router import GeminiRouter
from router.ollama import OllamaRouter
from router.open_ai_router import OpenAIRouter
from router.transformers_router import TransformerRouter
from router.vllm_router import VLLMRouter
from router.open_ai_compatible_router_vllm import VLLMRouter
import os
import string
import json


def choose_router(router_name):
    if router_name == 'openai':
        return OpenAIRouter()
    elif router_name == 'gemini':
        return GeminiRouter()
    elif router_name == "vllm":
        return VLLMRouter()
    elif router_name == "huggingface":
        return TransformerRouter()
    return OllamaRouter()


def save_final_config(args, configs, output_dir, base_filename=None):
    final_config = {**configs, **vars(args)}  # Merge configs with args (args take precedence)

    if not base_filename:
        base_filename = "config"
    file_extension = ".json"
    file_index = 0
    while True:
        if file_index == 0:
            filename = f"{base_filename}{file_extension}"
        else:
            filename = f"{base_filename}_{file_index}{file_extension}"

        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):
            break
        file_index += 1

    # Save the final configuration to the unique file
    with open(output_path, "w") as f:
        json.dump(final_config, f, indent=4)

    print(f"Final configuration saved to {output_path}")


def get_template_fields(template):
    """
    Extracts field names from the given template.

    Args:
        template (str): The template string.

    Returns:
        set: A set of field names present in the template.
    """
    formatter = string.Formatter()
    return {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}
