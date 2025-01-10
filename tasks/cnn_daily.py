from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm
from mask_generator import mask_generator
from reranker import candidates_reranker, samples_reranker
from sampler import sampler
from huggingface_hub import login
import json
import argparse
from tasks import base_summarizer
import util


class CNNDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        return input, label


class CNNDailySummarization(base_summarizer.BaseSummarization):
    # "abisee/cnn_dailymail"
    # "3.0.0"
    def __init__(self, dataset_name, version, batch_size, model_path, MODEL_MAP, re_ranker_config, sampler_config):
        hf_auth_token = "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"
        login(token=hf_auth_token)
        MODEL_MAP = {k: v for k, v in MODEL_MAP.items() if v.get("active")}
        dataset_loaded = load_dataset(dataset_name, version)
        data, summary = self.process_dataset(dataset_loaded)
        sentence_probability_template = self.get_sentence_template()
        dataset = CNNDataset(data, summary)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        generate_mask = mask_generator.MaskGenerator(MODEL_MAP=MODEL_MAP)
        candidates_reranker_generator = candidates_reranker.CandidatesReranker(sentence_probability_template,
                                                                               re_ranker_config=re_ranker_config,
                                                                               MODEL_MAP=MODEL_MAP)
        pair_sampler = sampler.Sampler(MODEL_MAP=MODEL_MAP, sampler_config=sampler_config)
        sample_reranker_generator = samples_reranker.SamplerReranker(re_ranker_config=re_ranker_config,
                                                                     MODEL_MAP=MODEL_MAP)

        prompt_template = self.get_template()

        super().__init__("cnn_daily", candidates_reranker_generator, pair_sampler, sample_reranker_generator,
                         model_path, generate_mask, dataloader, prompt_template)

    def get_sentence_template(self):
        return {
            "not_chat_tuned": "{sentence}",
            "chat_tuned": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "{sentence}"
                    }]
            }]
        }

    def get_template(self):
        return {
            "not_chat_tuned": (
                "You are an intelligent agent specialized in predicting the next token in a summary based on an original paragraph and a given prefix.\n\n"
                "### Original Paragraph:\n"
                "{input}\n\n"
                "### Instructions:\n"
                "Given the prefix of the summary, output **only** the next token to be predicted.\n"
                "- If the next token continues the last word of the summary, **do not** begin with a leading space.\n"
                "- If the next token starts a new word, **begin** with a leading space.\n\n"
                "### Examples:\n"
                "1. **Original Paragraph:** The quick brown fox jumps over the lazy dog.\n"
                "   **Summary Prefix:** The quick brown fox jumps over the lazy d\n"
                "\"og\"\n\n"
                "2. **Original Paragraph:** Climate change is causing severe weather patterns worldwide.\n"
                "   **Summary Prefix:** Climate change is causing severe weather patterns worldwide and\n"
                "\" has\"\n\n"
                "3. **Original Paragraph:** The advancements in artificial intelligence are remarkable.\n"
                "   **Summary Prefix:** The advancements in artificial intelligence are remarkable and\n"
                "\" will\"\n\n"
                "### Summary Prefix for Token Prediction:\n"
                "{prefix_candidate}"
            ),
            "chat_tuned": [
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent agent specialized in predicting the next token in a summary based on an original paragraph and a given prefix. "
                        "Please follow the instructions meticulously to ensure accurate predictions."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "### Original Paragraph:\n"
                                "{input}\n\n"
                                "### Instructions:\n"
                                "Given the prefix of the summary, output **only** the next token to be predicted.\n"
                                "- If the next token continues the last word of the summary, **do not** begin with a leading space.\n"
                                "- If the next token starts a new word, **begin** with a leading space.\n\n"
                                "### Examples:\n"
                                "1. **Original Paragraph:** The quick brown fox jumps over the lazy dog.\n"
                                "   **Summary Prefix:** The quick brown fox jumps over the lazy d\n"
                                "\"og\"\n\n"
                                "2. **Original Paragraph:** Climate change is causing severe weather patterns worldwide.\n"
                                "   **Summary Prefix:** Climate change is causing severe weather patterns worldwide and\n"
                                "\" has\"\n\n"
                                "3. **Original Paragraph:** The advancements in artificial intelligence are remarkable.\n"
                                "   **Summary Prefix:** The advancements in artificial intelligence are remarkable and\n"
                                "\" will\"\n\n"
                                "### Summary Prefix for Token Prediction:\n"
                                "{prefix_candidate}"
                            )
                        }
                    ]
                }
            ]
        }

    def process_dataset(self, ds):
        df_train = pd.DataFrame({
            'article': ds['train']['article'],  # Actual article (source text)
            'summary': ds['train']['highlights']  # Summarized version (highlights)
        })
        return df_train.get("article"), df_train.get("summary")


def parse_arguments(configs):
    """Parse command-line arguments and use defaults from the configuration file."""
    parser = argparse.ArgumentParser(description="Run CNN/DailyMail summarization pipeline.")

    # Define command-line arguments with defaults from configs.json
    parser.add_argument("--data_base_dir", default="/data/group_data/starlight/gpa/data",
                        help="Base directory for data.")
    parser.add_argument("--experiment_pipeline", default=configs.get("experiment_pipeline"),
                        help="Name of the experiment pipeline.")
    parser.add_argument("--batch_size", type=int, default=configs.get("batch_size"), help="Batch size for processing.")
    parser.add_argument("--start_group_id", type=int, default=configs.get("group_id_range", {}).get("start"),
                        help="Start group ID.")
    parser.add_argument("--end_group_id", type=int, default=configs.get("group_id_range", {}).get("end"),
                        help="End group ID.")
    parser.add_argument("--model_a", default=configs.get("model_a"), help="Path to Model A.")
    parser.add_argument("--model_b", default=configs.get("model_b"), help="Path to Model B.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    prefix = os.getcwd()

    # Load configs.json
    with open(f"{prefix}/granular_preferences/tasks/summarization/configs.json") as f:
        configs = json.load(f)

    # Parse arguments
    args = parse_arguments(configs)

    dataset = "cnn_daily"

    # Prepare directories
    if not os.path.exists(f"{args.data_base_dir}/{args.experiment_pipeline}"):
        os.makedirs(f"{args.data_base_dir}/{args.experiment_pipeline}")

    model_path = f"{args.data_base_dir}/{args.experiment_pipeline}/{dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Extract model-related configurations
    MODEL_MAP = configs.get("model_map")
    re_ranker_config = configs.get("re_ranker_config")
    sampler_config = configs.get("sampler_config")

    # Initialize and run the summarization pipeline
    cnn_summarizer = CNNDailySummarization(
        "abisee/cnn_dailymail", "3.0.0", args.batch_size, model_path, MODEL_MAP, re_ranker_config, sampler_config
    )

    cnn_summarizer.generate_candidates(start_group=args.start_group_id, end_group=args.end_group_id)
    cnn_summarizer.candidates_reranker(start_group=args.start_group_id, end_group=args.end_group_id)

    # Add sampling and reranking
    cnn_summarizer.sampler(start_group=args.start_group_id, end_group=args.end_group_id, model_a=args.model_a,
                           model_b=args.model_b)
    cnn_summarizer.sample_reranker(start_group=args.start_group_id, end_group=args.end_group_id, model_a=args.model_a,
                                   model_b=args.model_b)

    util.save_final_config(args, configs, model_path)
