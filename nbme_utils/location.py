import numpy as np

from typing import List, Tuple


def locations_to_spans(locations: List[str]) -> List[Tuple[int]]:
    """
    Converts location dataframe to spans.

    Args:
        locations (list[str]): Location. E.g., ['161 178', '161 169;179 183']

    Returns:
        list[tuple[int[2]]]: Spans. E.g., [(161, 178), (161, 169), (179, 183)]
    """
    spans = []
    for loc in locations:
        if ';' in loc:
            loc = loc.split(';')
        else:
            loc = [loc]

        for l in loc:
            spans.append(tuple(np.array(l.split(' ')).astype(int)))

    return spans


def spans_to_locations(spans: List[Tuple[int]]) -> List[str]:
    """
    Converts location dataframe to spans.

    Args:
        spans (list[tuple[int[2]]]): Spans. E.g., [(161, 178), (161, 169), (179, 183)]

    Returns:
        list[str]: Location. E.g., ['161 178; 161 169; 179 183']
    """
    locations = ['; '.join(f'{start} {end}' for start, end in spans)]
    return locations


def generate_labels(location_spans, sequence_ids, offset_mapping):
    labels = [0] * len(sequence_ids)

    for idx, (seq_id, offsets) in enumerate(zip(sequence_ids, offset_mapping)):
        # None for special tokens added around or between sequences,
        # 0 for tokens corresponding to words in the first sequence,
        # 1 for tokens corresponding to words in the second sequence when a pair of sequences was jointly encoded.
        # Labels are generated from patient notes, which are encoded as the first sequence.
        if seq_id != 0:
            labels[idx] = -1
            continue

        token_start, token_end = offsets

        # offsets are inclusive but location is exclusive at the end
        if any([token_start >= location_start and token_end < location_end
                for location_start, location_end in location_spans]):
            labels[idx] = 1

    return labels
