from src.zsl.conll_utils import load_sentences, conll_sentences_to_csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)


def main(path_data, args):
    """
    This function is the main entry point for the script. It performs the following steps:
    - Loads the sentences from the CoNLL-formatted data file.
    - Converts the sentences to CSV format and saves them to a new file.

    :param path_data: The path to the directory containing the data file.
    :param args: The command line arguments. Expected to contain the names of the CoNLL and CSV files.
    """
    logging.info("Loading the data")
    sentences = load_sentences(path_data, args.name_conll)
    logging.info("Converting to csv")
    conll_sentences_to_csv(sentences, path_data, args.name_csv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process CONLL to get csv")

    parser.add_argument("--path_data", type=str, help="Path to the data")
    parser.add_argument("--name_conll", type=str, help="Name of the conll file")
    parser.add_argument("--name_csv", type=str, help="Name of the csv file")

    args = parser.parse_args()

    main(args.path_data, args)
