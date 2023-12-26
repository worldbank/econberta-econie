from pathlib import Path
from typing import List, Union


def load_sentences(path: Union[str, Path], separator: str = ' ') -> List[List[List[str]]]:
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    print('')
    print('Loading sentences from %s' % path)
    print('')
    if type(path) == str:
        _path = open(path, mode='r', encoding='utf8')
    elif type(path) == Path:
        _path = path.open(mode='r', encoding='utf8')
    else:
        raise ValueError(f'Invalid argument type {type(path)}')
    for line in _path:
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split(sep=separator)
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences
