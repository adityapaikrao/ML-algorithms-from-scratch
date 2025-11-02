import unicodedata


from typing import List, Dict, Tuple
from collections import Counter

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s



### Functions for the BPETokenizer ###

def get_pairs(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Returns the counts of the consecutive pairs in the list of ints (token ids)
    """
    counts = Counter(zip(ids, ids[1:]))
    return dict(counts)

def merge(ids: List[int], max_pair: Tuple[int, int], new_idx: int):
    """
    Replaces all occurences of the max_pair in the given list of ints (token ids) and asssigns them the new_idx
    """
    new_ids = []
    i = 0 

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == max_pair[0] and ids[i+1] == max_pair[1]:
            new_ids.append(new_idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    
    return new_ids



