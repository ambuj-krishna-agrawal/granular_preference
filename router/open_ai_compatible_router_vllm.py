import requests
from transformers import AutoTokenizer
import numpy as np
import copy

from router.base_llm_router import BaseLLMRouter


class VLLMRouter(BaseLLMRouter):
    '''
    Class responsible for interaction with LLM models via vLLM
    '''

    def __init__(self):
        pass

    def get_sentence_probability(self, model, base_url, prompt):
        payload = self.get_payload(model_name=model, max_tokens=0, logprobs=1, prompt=prompt, temperature=0,
                                   is_chat_tuned=False, echo=True)
        try:
            response = requests.post(base_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
                total_logprob = sum(logprobs[1:])
                return round(total_logprob, 3)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                raise ("error happened, check")
        except Exception as e:
            print(e)
            raise (e)

    def inference_call(self, model, base_url, prompt):

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1000,
            "echo": False
        }
        response = requests.post(f"{base_url}", json=payload)
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['text']
            return generated_text
        return None

    def get_logprobs_content(self, is_chat_tuned, payload, base_url):
        response = requests.post(f"{base_url}", json=payload)
        result = response.json()

        if not is_chat_tuned:
            logprobs = result['choices'][0]['logprobs']['top_logprobs'][0]
            content = result['choices'][0]['text']
            logprobs = sorted(logprobs.items(), key=lambda val: -val[1])
            return logprobs, content
        else:
            logprobs = result['choices'][0]['logprobs']['content'][0]['top_logprobs']
            logprobs = sorted(logprobs, key=lambda val: -val['logprob'])
            content = result['choices'][0]['message']['content']
            return logprobs, content

    def get_payload(self, model_name, max_tokens, logprobs, prompt, temperature, is_chat_tuned=False, echo=False):
        if not is_chat_tuned:
            return {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "logprobs": logprobs,
                "echo": echo,
                "temperature": temperature
            }
        else:
            chat_template = '''{% if messages[0].role == 'system' %}{{messages[0].content}}{% endif %}
{% for message in messages %}
    {% if message.role == 'user' %}
        {% for content in message.content %}
            {% if content.type == 'text' %}
                {{content.text}}
            {% elif content.type == 'image_url' and content.image_url %}
                <image>{{content.image_url.url}}</image>
            {% endif %}
        {% endfor %}
    {% elif message.role == 'assistant' %}
        {{message.content}}
    {% endif %}
{% endfor %}'''
            return {
                "model": model_name,
                "messages": prompt,
                "max_tokens": max_tokens,
                "logprobs": True,
                "echo": echo,
                "top_logprobs": logprobs,
                "temperature": temperature,
                'chat_template': chat_template
            }

    def get_token_logprob(self, i, logprobs, is_chat_tuned):

        if is_chat_tuned:
            return logprobs[i]['token'], logprobs[i]['logprob']
        else:
            return logprobs[i][0], logprobs[i][1]

    def concatenate(self, prompt, token, is_chat_tuned):
        if is_chat_tuned:
            for message in prompt:
                if message.get('role') == 'user' and 'content' in message:
                    content_list = message['content']
                    for content_item in content_list:
                        if content_item.get('type') == 'text':
                            content_item['text'] += token
        else:
            prompt += token
        return prompt

    def generate_candidates(self, validator, model_name, base_url, prompt, top_k):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        final_word_predictions = {}
        # print(model_name)
        is_chat_tuned = isinstance(prompt, list)
        try:
            candidates = []
            payload = self.get_payload(model_name=model_name, max_tokens=5, logprobs=top_k + 1, prompt=prompt,
                                       temperature=0, is_chat_tuned=is_chat_tuned)
            logprobs, content = self.get_logprobs_content(is_chat_tuned=is_chat_tuned, payload=payload,
                                                          base_url=base_url)
            print("First")
            print(content)
            for i in range(top_k):
                candidates.append(self.get_token_logprob(i=i, logprobs=logprobs, is_chat_tuned=is_chat_tuned))
        except Exception as e:
            print(e)
            raise (e)

        i = 0
        for candidate in candidates:
            step_in_future = 1
            token, logprob = candidate
            current_prompt = copy.deepcopy(prompt)
            current_prompt = self.concatenate(current_prompt, token, is_chat_tuned)
            predicted_tokens = [token]
            predicted_logprobs = [logprob]
            while True:
                payload = self.get_payload(model_name=model_name, max_tokens=5, logprobs=3, prompt=current_prompt,
                                           temperature=0, is_chat_tuned=is_chat_tuned)
                logprobs, content = self.get_logprobs_content(is_chat_tuned=is_chat_tuned, payload=payload,
                                                              base_url=base_url)
                print("Continuation")
                print(content)
                next_token, next_logprob = self.get_token_logprob(i=0, logprobs=logprobs, is_chat_tuned=is_chat_tuned)
                if (next_token.startswith(" ") and step_in_future > 0) or step_in_future >= 10:
                    break
                step_in_future += 1
                predicted_tokens.append(next_token)
                predicted_logprobs.append(next_logprob)
                current_prompt = self.concatenate(current_prompt, next_token, is_chat_tuned)
            detokenized_output = tokenizer.convert_tokens_to_string(predicted_tokens).strip()
            predicted_word = detokenized_output.replace("\t", "")
            predicted_word = detokenized_output.replace("\n", "")
            word_prob = np.sum(predicted_logprobs) / len(predicted_logprobs)
            prob = np.round(np.exp(word_prob), 3)
            final_word_predictions[i] = (predicted_word, prob)
            i += 1

        preds = {k: v for (k, v) in sorted(final_word_predictions.items(), key=lambda val: val[0])}
        # print("predictions from vllm: ", final_word_predictions, " for model: ", model_name)
        return preds