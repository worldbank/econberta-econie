import json
import pandas as pd
import argparse
import numpy as np
import ast

from transformers import AutoTokenizer
from sklearn.metrics import classification_report

from src.zsl.llm_utils import convert_df_to_token_level, convert_response_from_llm

import logging

logging.basicConfig(level=logging.INFO)


def main(args):
    """
    This function is the main entry point for the script. It performs the following steps:
    - Loads the parameters and tokenizer from the settings file.
    - Loads the test set and converts it to token-level IOB format.
    - Loads the predictions.
    - Computes the metrics by comparing the predictions to the true labels.
    - Logs any problems with the format of the responses.
    - Flattens the true labels and predictions and computes the classification report.

    :param args: The command line arguments. Expected to contain the paths to the settings file, test set, and predictions.
    """
    logging.info("Loading parameters and tokenizer")

    with open(args.path_settings, "r") as f:
        setting_metrics = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(setting_metrics.get("tokenizer"))
    categories = setting_metrics.get("categories")

    logging.info("Loading test set")

    test_df = pd.read_csv(args.path_data_test)
    test_dict = test_df.fillna("[]").T.to_dict()
    test_dict = convert_df_to_token_level(test_dict, tokenizer, categories)

    logging.info("Loading predictions...")

    with open(args.path_data_preds, "r") as f:
        responses = [json.loads(line) for line in f]

    trues = []
    preds = []

    format_problems = []

    logging.info("Compute the metrics...")

    for i, response in enumerate(responses):
        try:
            text = response["text"]
            response = ast.literal_eval(response["response"])
            iob_tokens = convert_response_from_llm(
                text, response, tokenizer, categories
            )
            preds.append(iob_tokens)
            trues.append(test_dict[i]["IoB"])
        except Exception:
            format_problems.append(response)

    flattened_trues = np.array([item for sublist in trues for item in sublist]).reshape(
        -1, 2
    )
    flattened_preds = np.array([item for sublist in preds for item in sublist]).reshape(
        -1, 2
    )

    logging.info(
        classification_report(flattened_trues[:, 1], flattened_preds[:, 1], digits=5)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply LLM to some sentences")

    parser.add_argument("--path_data_test", type=str, help="Path to the test data")
    parser.add_argument("--path_data_preds", type=str, help="Path to the preds data")
    parser.add_argument("--path_settings", type=str, help="Path to the settings")

    args = parser.parse_args()

    main(args)
