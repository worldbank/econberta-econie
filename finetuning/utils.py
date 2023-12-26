import html
import re

from typing import Tuple, Union, List
from datetime import datetime
from pathlib import Path

BOM = chr(ord('\ufeff'))
re1 = re.compile(r'  +')
__BLANK_SPACE__ = re.compile(r"^\s+$")
re_single_quotes = re.compile(r"(?u)[‘’′`']", re.UNICODE)

keep_codes = [8, 9, 10, 13, 730, 8208, 8209, 8210, 8211, 8212, 8213, 8214, 8216, 8217, 8218, 8219, 8220, 8221, 8222,
              8223, 8226, 8230]


def match(text: str, regex: re.Pattern):
    return len(regex.findall(text)) > 0


def is_blank_space(text: str) -> bool:
    return match(text=text, regex=__BLANK_SPACE__)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def skip_token(token: str) -> bool:
    return len(token.strip()) == 1 and ord(token.strip()) > 255 and ord(token.strip()) not in keep_codes


def normalize_single_quotes(text: Union[str, List[str]]) -> Union[str, List[str]]:
    _type = type(text)
    if _type == str:
        return re_single_quotes.sub("'", text)
    elif _type == list:
        return [normalize_single_quotes(token) for token in text]


def get_or_create_path(path: Union[str, Path]) -> Path:
    if type(path) == str:
        path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_or_remove_path(path: Union[str, Path]) -> Path:
    if type(path) == str:
        path = Path(path)
    if path.exists():
        path.unlink()
    return path


def log(message):
    print(datetime.now(), '-', message)


def abrev(string) -> str:
    return ''.join([substring[0] for substring in string.split('_')])


def coordinates_in_interval(ids: Tuple[int, int], start: int, end: int) -> bool:
    return (start <= ids[0] <= end) and (start <= ids[1] <= end)


def clean_non_ascii_characters(string: str) -> str:
    string = string.replace('ﬁ', 'fi')
    keep_string = ''.join([i if (ord(i) < 256 or ord(i) in keep_codes) else '' for i in string]).strip()
    # discard_chars = [i for i in string if (ord(i) > 256 and ord(i) not in keep_codes)]
    # if len(discard_chars) > 0:
    #     logger.info(f"Discarding: {discard_chars}")
    return keep_string


def clean_characters(string: str) -> str:
    return string.replace(BOM, '').replace('\u200b', '').replace('\uf0d8', '').replace('\uf0fc', '').replace('\uf0b7',
                                                                                                             '').strip()


def fixup(x):
    return clean_characters(re1.sub(' ', html.unescape(x)).replace('\xa0', ' ').replace('\xad', ''))


def clean_text(string: str) -> str:
    return clean_non_ascii_characters(string).replace('\n', '')


def get_splits_sizes(total: int, train_pct: float, dev_pct: float = None, test_pct: float = None):
    if train_pct is not None and (dev_pct is None or test_pct is None):
        dev_pct = (1. - train_pct) / 2
        test_pct = dev_pct
    train_sz, test_sz, dev_sz = int(total * train_pct), int(total * test_pct), int(total * dev_pct)
    train_sz = train_sz + (total - train_sz - test_sz - dev_sz)
    return train_sz, dev_sz, test_sz


def get_train_dev_splits_sizes(total: int, train_pct: float):
    train_sz = int(total * train_pct)
    dev_sz = total - train_sz
    return train_sz, dev_sz
