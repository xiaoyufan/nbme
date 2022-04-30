
import numpy as np

from typing import List


def locations_to_spans(locations: List[str]) -> List[List[int]]:
    """
    Converts location dataframe to spans.

    Args:
        locations (list[str]): Location. E.g., ['161 178', '161 169;179 183']

    Returns:
        list[list[int[2]]]: Spans. E.g., [[161, 178], [161, 169], [179, 183]]
    """
    spans = []
    for loc in locations:
        if ";" in loc:
            loc = loc.split(';')
        else:
            loc = [loc]

        for l in loc:
            spans.append(list(np.array(l.split(' ')).astype(int)))

    return spans


def spans_to_locations(spans: List[List[int]]) -> List[str]:
    """
    Converts location dataframe to spans.

    Args:
        spans (list[list[int[2]]]): Spans. E.g., [[161, 178], [161, 169], [179, 183]]

    Returns:
        list[str]: Location. E.g., ['161 178; 161 169; 179 183']
    """
    locations = ['; '.join(map(str, span)) for span in spans]
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
