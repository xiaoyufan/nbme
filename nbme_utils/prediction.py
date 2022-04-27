TRUE_THRESHOLD = 0.5


def logits_to_spans(batch_logits, batch_encoded):
    batch_spans = []

    for seq_logits, seq_offsets, seq_ids in zip(batch_logits,
                                                batch_encoded['offset_mapping'],
                                                batch_encoded['sequence_ids']):
        seq_spans = []
        is_prev_true = False

        for token_logit, token_offsets, token_seq_id in zip(seq_logits, seq_offsets, seq_ids):
            if token_seq_id != 0 or token_logit < TRUE_THRESHOLD:
                is_prev_true = False
                continue

            token_start, token_end = token_offsets

            if is_prev_true:
                # Update end offset of the current span
                seq_spans[-1][1] = token_end
            else:
                seq_spans.append([token_start, token_end])

            is_prev_true = True

        batch_spans.append(seq_spans)

    return batch_spans