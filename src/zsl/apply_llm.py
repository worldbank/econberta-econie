import openai
import json
import os
from os.path import join
import pandas as pd
import argparse

from src.zsl.llm_utils import (
    get_examples_from_annotation,
    load_prompts,
    log_prompts,
    process_sentences,
)

import logging

logging.basicConfig(level=logging.INFO)


def main(args):
    """
    This function is the main entry point for the script. It performs the following steps:
    - Initializes the OpenAI client with the API key.
    - Loads the examples from the annotation file.
    - Loads the instruction prompt, examples template, and inference prompt.
    - If verbose mode is enabled, logs the prompts.
    - Loads the data for inference.
    - Initializes the responses and tokens info.
    - If the output file exists, loads the existing responses.
    - Processes the sentences, generating responses and updating the tokens info.
    - Writes the responses to the output file.

    :param args: The command line arguments. Expected to contain the paths to the data, prompts, and output, as well as the model name and verbosity flag.
    """

    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    fsl = pd.read_csv(join(args.path_data, f"{args.example_csv_name}.csv"))
    EXAMPLES = get_examples_from_annotation(fsl, n=args.n_examples)

    INSTRUCTION_PROMPT, EXAMPLES_TEMPLATE, INFERENCE_PROMPT = load_prompts(
        args.path_prompt,
        args.instruction_prompt,
        args.examples_prompt,
        args.inference_prompt,
        load_examples=args.n_examples,
    )

    if args.verbose:
        log_prompts(INSTRUCTION_PROMPT, EXAMPLES_TEMPLATE, EXAMPLES, INFERENCE_PROMPT)

    df = pd.read_csv(join(args.path_data, f"{args.inference_csv_name}.csv"))

    responses = []
    tokens_info = {
        "total_token_prompts": 0,
        "total_token_completion": 0,
        "total_prompt_price": 0,
        "total_completion_price": 0,
    }
    try:
        with open(join(args.path_output), "r") as f:
            responses = [json.loads(line) for line in f]
    except FileNotFoundError:
        pass

    responses, tokens_info = process_sentences(
        df.iloc[:50, :],
        responses,
        tokens_info,
        client,
        INSTRUCTION_PROMPT,
        INFERENCE_PROMPT,
        EXAMPLES_TEMPLATE,
        EXAMPLES,
        model=args.model,
        verbose=args.verbose,
    )

    with open(join(args.path_output), "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply LLM to some sentences")

    parser.add_argument("--path_data", type=str, help="Path to the data")
    parser.add_argument(
        "--example_csv_name", type=str, help="Name of the example csv file"
    )
    parser.add_argument(
        "--n_examples", type=int, help="Number of examples to use", default=1
    )
    parser.add_argument("--path_prompt", type=str, help="Path to the prompt")
    parser.add_argument("--instruction_prompt", type=str, help="Instruction prompt")
    parser.add_argument("--examples_prompt", type=str, help="Examples prompt")
    parser.add_argument("--inference_prompt", type=str, help="Inference prompt")
    parser.add_argument("--inference_csv_name", type=str, help="Inference csv name")
    parser.add_argument(
        "--model", type=str, help="Model to use", default="gpt-3.5-turbo-0125"
    )
    parser.add_argument("--path_output", type=str, help="Path to the output")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    main(args)
