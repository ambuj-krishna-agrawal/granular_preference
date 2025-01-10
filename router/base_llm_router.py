from abc import ABC, abstractmethod

import requests


class BaseLLMRouter(ABC):
    def __init__(self):
        pass

    def get_sentence_probability(self, model: str, base_url: str, prompt: str) -> float:
        pass

    def inference_call(self, model: str, base_url: str, prompt: str) -> str:
        pass

    def generate_candidates(self, validator, model_name: str, base_url: str, prompt: str, top_k: int, image: str = None):
        pass