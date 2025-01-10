import os
from tqdm import tqdm
from mask_generator import mask_generator
from reranker import candidates_reranker, samples_reranker
from sampler import sampler
import re
from typing import Any


class BaseSummarization:
    def __init__(self, source: str, candidates_reranker_generator: candidates_reranker.CandidatesReranker,
                 pair_sampler: sampler.Sampler, sample_reranker_generator: samples_reranker.SamplerReranker,
                 model_path: str, mask_generator: mask_generator.MaskGenerator, dataloader: Any, template: Any):
        self.source = source
        self.sample_reranker_generator = sample_reranker_generator
        self.candidates_reranker_generator = candidates_reranker_generator
        self.pair_sampler = pair_sampler
        self.model_path = model_path
        self.mask_generator = mask_generator
        self.dataloader = dataloader
        self.prompt_template = template

    def generate_candidates(self, start_group, end_group, is_multimodal=False):
        group_id = 0
        for inputs, labels in self.dataloader:
            group_id += 1
            if group_id < start_group:
                continue
            if group_id > end_group:
                break
            group_folder = f"{self.model_path}/group_{group_id}"
            if not os.path.exists(group_folder):
                os.makedirs(group_folder)

            processed_inputs = []
            processed_labels = []

            with open(f"{group_folder}/label_original_indexed.tsv", "w") as label_file, open(
                    f"{group_folder}/input_original_indexed.tsv", "w") as input_file:
                for input, label in zip(inputs, labels):
                    label = label.replace("\n", "")
                    input = input.replace("\n", "")
                    processed_labels.append(label)
                    processed_inputs.append(input)
                    label_file.write(label + "\n")
                    input_file.write(input + "\n")
            use_existing_candidates = False

            if not use_existing_candidates:
                files_to_delete = [
                    f"{group_folder}/candidates_prefix.tsv",
                    f"{group_folder}/candidates_input.tsv",
                    f"{group_folder}/candidates_label.tsv",
                    f"{group_folder}/candidates_masked.tsv",
                ]

                for file_path in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")
                        else:
                            print(f"File not found, skipping: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

            for input, label in tqdm(zip(processed_inputs, processed_labels), desc="processing masking"):
                prefix_candidates, masked_candidates = self.mask_generator.get_candidates(use_existing_candidates,
                                                                                          label, input, group_folder)
                self.mask_generator.generate(input=input, prefix_candidates=prefix_candidates,
                                             masked_candidates=masked_candidates, template=self.prompt_template,
                                             data_path=group_folder, is_multimodal=is_multimodal)

    def candidates_reranker(self, start_group, end_group):
        group_folder = f"{self.model_path}/"

        for group_path in os.listdir(group_folder):
            matches = re.findall(r"group_(\d+)", group_path)
            if matches:
                group_id = int(matches[0])
            else:
                continue
            if group_id < start_group:
                continue
            if group_id > end_group:
                break

            group_folder = f"{self.model_path}/{group_path}"
            self.candidates_reranker_generator.rerank(group_folder)

    def sampler(self, start_group, end_group, model_a, model_b):
        group_folder = f"{self.model_path}/"

        for group_path in os.listdir(group_folder):
            matches = re.findall(r"group_(\d+)", group_path)
            if matches:
                group_id = int(matches[0])
            else:
                continue
            if group_id < start_group:
                continue
            if group_id > end_group:
                break

            group_folder = f"{self.model_path}/{group_path}"
            self.pair_sampler.sample(model_a=model_a, model_b=model_b, group_path=group_folder)

    def sample_reranker(self, start_group, end_group, model_a, model_b):
        group_folder = f"{self.model_path}/"

        for group_path in os.listdir(group_folder):
            matches = re.findall(r"group_(\d+)", group_path)
            if matches:
                group_id = int(matches[0])
            else:
                continue
            if group_id < start_group:
                continue
            if group_id > end_group:
                break

            group_folder = f"{self.model_path}/{group_path}"
            self.sample_reranker_generator.rerank(model_a=model_a, model_b=model_b, group_path=group_folder)

