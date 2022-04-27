
import numpy as np

from typing import List


def locations_to_spans(location: List[str]) -> List[List[int]]:
    """
    Converts location dataframe to spans.

    Args:
        location (list[str]): Location. E.g., ['161 178', 161 169;179 183']

    Returns:
        list[list[int[2]]]: Spans. E.g., [[161, 178], [161, 169], [179, 183]]
    """
    spans = []
    for loc in location:
        if ";" in loc:
            loc = loc.split(';')
        else:
            loc = [loc]

        for l in loc:
            spans.append(list(np.array(l.split(' ')).astype(int)))

    return spans
