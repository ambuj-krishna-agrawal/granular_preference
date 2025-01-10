import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.tokenize.treebank import TreebankWordDetokenizer

import string
import os
import copy
from router.llm import LLM
from util import choose_router, get_template_fields


class WordValidator:

    def __init__(self, word_list, stop_words):
        self.word_list = word_list
        self.stop_words = stop_words

    def validate_input(self, word):
        is_stop_word = word.lower() in self.stop_words
        is_punctuation = word in string.punctuation
        is_valid = self.is_valid_word(word)
        return is_valid and not is_stop_word and not is_punctuation

    def validate_output(self, word):
        is_punctuation = word in string.punctuation
        is_valid = self.is_valid_word(word)
        return is_valid and not is_punctuation

    def is_valid_word(self, word):
        if word != "" and word is not None:
            expanded = contractions.fix(word)
            if expanded == word:
                return word in self.word_list or word.istitle()
            else:
                for exp_word in expanded.split():
                    exp_word = exp_word.strip()
                    if exp_word == "" or exp_word not in self.word_list and not exp_word.istitle():
                        return False
                return True
        return False


class MaskGenerator:
    def __init__(self, MODEL_MAP):
        self.MODEL_MAP = MODEL_MAP
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('words')

        self.exclude_pos_tags = ['DT', 'IN', 'CC']
        # noun_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        self.word_list = set(words.words())
        self.stop_words = set(stopwords.words('english'))
        self.validator = WordValidator(self.word_list, self.stop_words)
        self.detokenizer = TreebankWordDetokenizer()
        self.llm = LLM()

    def write_statics(self, prefix_candidates, input, label, masked_candidates, data_path):
        with open(f"{data_path}/candidates_prefix.tsv", "a") as f:
            for prefix_candidate in prefix_candidates:
                f.write(prefix_candidate)
                f.write("\n")
        with open(f"{data_path}/candidates_label.tsv", "a") as f:
            for _ in range(len(prefix_candidates)):
                f.write(label)
                f.write("\n")
        with open(f"{data_path}/candidates_input.tsv", "a") as f:
            for _ in range(len(prefix_candidates)):
                f.write(input)
                f.write("\n")
        with open(f"{data_path}/candidates_masked.tsv", "a") as f:
            for masked_candidate in masked_candidates:
                f.write(masked_candidate)
                f.write("\n")

    def write_file(self, word_sentence_model_1, data_path, model_directory_name):
        model_path = f"{data_path}/{model_directory_name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(f"{model_path}/candidates.tsv", "a") as f:
            final_sentence = "\t".join(
                [f"{key}:{word}:{prob}" for key, (word, prob) in word_sentence_model_1.items()]) + "\n"
            f.write(final_sentence)

    def get_candidates(self, use_existing_candidates, label, input, data_path):
        prefix_candidates = []
        masked_candidates = []
        if not use_existing_candidates:
            tokens = word_tokenize(label)
            pos_tags = nltk.pos_tag(tokens)
            candidate_indexes = []

            for i, (word, pos) in enumerate(pos_tags):
                if i < 5:
                    continue
                if self.validator.validate_input(word) and pos not in self.exclude_pos_tags:
                    candidate_indexes.append(i)

            for ind, i in enumerate(candidate_indexes):
                masked_sentence = tokens[:i] + ["___"] + tokens[i + 1:]
                new_prefix = tokens[:i]
                new_prefix = self.detokenizer.detokenize(new_prefix)
                masked_candidate = self.detokenizer.detokenize(masked_sentence)
                prefix_candidates.append(new_prefix)
                masked_candidates.append(masked_candidate)

            self.write_statics(prefix_candidates, input, label, masked_candidates, data_path)
        else:
            with open(f"{data_path}/candidates_prefix.tsv", "r") as f:
                for line in f:
                    prefix_candidates.append(line)

            with open(f"{data_path}/candidates_masked.tsv", "r") as f:
                for line in f:
                    masked_candidates.append(line)
        return prefix_candidates, masked_candidates

    def get_final_prompt(self, template, is_chat_tuned, input_text, prefix_candidate, masked_candidate):
        """
        Constructs the final prompt or messages array based on whether chat tuning is used.

        Args:
            template (dict): The prompt templates containing 'not_chat_tuned' and 'chat_tuned'.
            is_chat_tuned (bool): Flag indicating whether to use chat-tuned templates.
            input_text (str): The input image URL to be summarized.
            prefix_candidate (str): The prefix of the summary for token prediction.

        Returns:
            str or list: Formatted prompt string for completions or messages list for chat completions.
        """
        if not is_chat_tuned:
            not_chat_template = template['not_chat_tuned']
            template_fields = get_template_fields(not_chat_template)
            format_kwargs = {'prefix_candidate': prefix_candidate}
            if 'input' in template_fields:
                format_kwargs['input'] = input_text
            if 'masked_candidate' in template_fields:
                format_kwargs['masked_candidate'] = masked_candidate
            formatted_prompt = not_chat_template.format(**format_kwargs)
            return formatted_prompt
        else:
            chat_tuned_template = copy.deepcopy(template['chat_tuned'])
            for message in chat_tuned_template:
                if message['role'] == 'user':
                    content_list = message['content']
                    for content_item in content_list:
                        if content_item['type'] == 'text':
                            format_kwargs = {'prefix_candidate': prefix_candidate}
                            if '{input}' in content_item['text']:
                                format_kwargs['input'] = input_text
                            if '{masked_candidate}' in content_item['text']:
                                format_kwargs['masked_candidate'] = masked_candidate
                            content_item['text'] = content_item['text'].format(**format_kwargs)
                        elif content_item['type'] == 'image_url':
                            if 'url' in content_item['image_url']:
                                content_item['image_url']['url'] = content_item['image_url']['url'].format(
                                    input=input_text)
            return chat_tuned_template

    def generate(self, input, prefix_candidates, masked_candidates, template, data_path, is_multimodal):
        print("Processing: ", len(masked_candidates))
        for prefix_candidate, masked_candidate in zip(prefix_candidates, masked_candidates):
            word_model_dict_list = []
            try:
                for model_name, model_info in self.MODEL_MAP.items():
                    huggingface_name = model_info.get("huggingface_name")
                    is_chat_tuned = model_info.get("chat_tuned")
                    base_url = model_info.get("base_url")
                    directory_name = model_info.get("directory_name")
                    mode = model_info.get("mode")

                    if not is_multimodal:
                        base_url = model_info.get("completion_base_url")
                        is_chat_tuned = False

                    # print(huggingface_name, base_url, directory_name)
                    prompt = self.get_final_prompt(template, is_chat_tuned, input, prefix_candidate, masked_candidate)
                    word_prob_dict = self.llm.generate_candidates(self.validator, choose_router(mode), huggingface_name,
                                                                  base_url, prompt, 10)
                    if not word_prob_dict:
                        raise ("word_prob_dict is None")
                    word_model_dict_list.append(word_prob_dict)
                    self.write_file(word_prob_dict, data_path, directory_name)
            except Exception as e:
                print(e)


