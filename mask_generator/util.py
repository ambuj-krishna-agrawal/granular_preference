import string

from router.gemini_router import GeminiRouter
from router.ollama import OllamaRouter
from router.open_ai_router import OpenAIRouter
from router.transformers_router import TransformerRouter
from router.vllm_router import VLLMRouter
import numpy as np


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


def read(path):
    with open(path) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split("\t") for x in data]
    return data

def strip_punctuations(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def softmax(x):
    """Compute the softmax of a list of numbers."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

