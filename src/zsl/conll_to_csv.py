from src.zsl.conll_utils import load_sentences, conll_sentences_to_csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)


def main(path_data, args):
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