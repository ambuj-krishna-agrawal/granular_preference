class LLM:
    def __init__(self):
        pass

    def call(self, router, model_name, base_url, prompt):
        return router.inference_call(model_name, base_url, prompt)

    def generate_candidates(self, validator, router, model_name, base_url, prompt, top_k):
        return router.generate_candidates(validator, model_name, base_url, prompt, top_k)

    def get_sentence_prob(self, router, model_name, base_url, prompt):
        return router.get_sentence_probability(model_name, base_url, prompt)
