from typing import List, Union
from pathlib import Path
import pandas as pd
from os.path import join
import logging

logging.basicConfig(level=logging.INFO)

### CONLL TO LLM DATA ###


def load_sentences(
    path: Union[str, Path], name_conll: str, separator: str = " "
) -> List[List[List[str]]]:
    """
    Load sentences from a file. A line in the file must contain at least a word and its tag.
    Sentences in the file are separated by empty lines.

    :param path (Union[str, Path]): The path to the file to load the sentences from.
    :param name_conll (str): The name of the file to load the sentences from.
    :param separator (str): The separator used to split the words and tags in the file.
    ;return List[List[List[str]]]: A list of sentences where each sentence is a list of word-tag pairs.
    """
    sentences = []
    sentence = []

    path = join(path, f"{name_conll}.conll")

    # Open the file
    if isinstance(path, str):
        file = open(path, mode="r", encoding="utf8")
    elif isinstance(path, Path):
        file = path.open(mode="r", encoding="utf8")
    else:
        raise ValueError(f"Invalid argument type {type(path)}")

    with file:
        for line in file:
            line = line.rstrip()
            if not line:
                # End of sentence
                if sentence and "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
            else:
                # Continue sentence
                word = line.split(sep=separator)
                assert (
                    len(word) >= 2
                ), "Each line must contain at least a word and its tag."
                sentence.append(word)

    # Add the last sentence if it exists
    if sentence and "DOCSTART" not in sentence[0][0]:
        sentences.append(sentence)

    return sentences


def generate_entities(sentence):
    """
    Generate a dictionary of entities and the sentence from a list of words and tags.

    :param sentence: A list of lists where each inner list contains a word and its corresponding tag.
    :return entities: A dictionary where the keys are the entity types and the values are lists of entities of that type, and the sentence as a string.
    """
    # Initialize an empty dictionary to store the entities
    entities = {}

    # Initialize an empty list to store the current entity
    entity = []

    # Initialize a variable to store the current tag
    current_tag = None

    # Initialize an empty list to store the words of the sentence
    sentence_words = []

    # Iterate over each word and tag in the sentence
    for word, tag in sentence:
        # Add the word to the sentence
        sentence_words.append(word)

        # If the tag starts with 'B-', it indicates the beginning of an entity
        if tag.startswith("B-"):
            # If there is a current entity, add it to the dictionary
            if current_tag is not None and entity:
                entities.setdefault(current_tag[2:], []).append(" ".join(entity))
            # Start a new entity with the current word
            entity = [word]
            # Update the current tag
            current_tag = tag
        # If the tag starts with 'I-', it indicates the continuation of an entity
        elif tag.startswith("I-") and tag[2:] == current_tag[2:]:
            # Add the current word to the entity
            entity.append(word)

    # If there is a current entity at the end of the sentence, add it to the dictionary
    if current_tag is not None and entity:
        entities.setdefault(current_tag[2:], []).append(" ".join(entity))

    # Add the counts of entities to the dictionary
    _entities = list(entities.keys())
    for tag in _entities:
        entities[f"count_{tag}"] = len(entities[tag])

    # Join the words of the sentence to form a string
    entities["sentence"] = " ".join(sentence_words)

    return entities


def conll_sentences_to_csv(sentences, path_data, name_csv):
    """
    Convert a list of sentences into a CSV file where each row represents a sentence and its entities.

    :param sentences: A list of sentences where each sentence is a list of words and tags.
    :param path_data: The path where the CSV file will be saved.
    :param name_csv: The name of the CSV file.
    """

    data = []
    for sentence in sentences:
        entities = generate_entities(sentence)
        transformed_data = {
            **{"sentence": entities.get("sentence")},
            **{
                key: str(value)
                for key, value in entities.items()
                if key != "sentence" or key.startswith("count_")
            },
        }
        data.append(transformed_data)

    # Create a DataFrame from the transformed dictionary
    df = pd.DataFrame(data)

    for key in df.columns:
        if key.startswith("count_"):
            df[key] = df[key].fillna(0)

    df.to_csv(join(path_data, f"{name_csv}.csv"), index=False)
