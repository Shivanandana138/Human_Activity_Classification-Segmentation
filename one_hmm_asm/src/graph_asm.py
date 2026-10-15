"""Explicit graph constraints and Viterbi decoding for activity sequences.

These project-specific edges are assumptions, not the paper's unavailable
edge list. Edit ``NON_SELF_TRANSITIONS`` to configure the application graph.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ACTIVITY_LABELS = (1, 2, 3, 4, 5, 6)
NON_SELF_TRANSITIONS = {
    1: (2, 3, 5),
    2: (1, 3),
    3: (1, 2),
    4: (5, 6),
    5: (1, 4, 6),
    6: (4, 5),
}


@dataclass(frozen=True)
class ActivityGraph:
    labels: tuple[int, ...]
    allowed_transitions: tuple[tuple[int, ...], ...]
    log_transition_probabilities: np.ndarray
    log_start_probabilities: np.ndarray


def _validated_lengths(lengths, sample_count):
    lengths = np.asarray(lengths, dtype=np.int64)
    if (
        lengths.ndim != 1
        or len(lengths) == 0
        or np.any(lengths <= 0)
        or int(lengths.sum()) != sample_count
    ):
        raise ValueError("Sequence lengths must be positive and sum to sample count.")
    return lengths


def build_activity_graph(
    activity_labels: Sequence[int] = ACTIVITY_LABELS,
    non_self_transitions: Mapping[int, Sequence[int]] = NON_SELF_TRANSITIONS,
):
    labels = tuple(int(label) for label in activity_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Activity labels must be non-empty and unique.")
    if set(non_self_transitions) != set(labels):
        raise ValueError("Graph must define non-self transitions for every activity.")
    edge_map = {label: {label} for label in labels}
    for label in labels:
        destinations = {int(value) for value in non_self_transitions[label]}
        if label in destinations or not destinations.issubset(labels):
            raise ValueError(f"Invalid non-self transitions for activity {label}.")
        edge_map[label].update(destinations)

    label_indices = {label: index for index, label in enumerate(labels)}
    log_transition_probabilities = np.full(
        (len(labels), len(labels)), -np.inf, dtype=np.float64
    )
    for source in labels:
        source_index = label_indices[source]
        destinations = edge_map[source]
        destination_indices = [label_indices[destination] for destination in destinations]
        log_transition_probabilities[source_index, destination_indices] = 0.0

    return ActivityGraph(
        labels=labels,
        allowed_transitions=tuple(
            tuple(label for label in labels if label in edge_map[source])
            for source in labels
        ),
        log_transition_probabilities=log_transition_probabilities,
        log_start_probabilities=np.zeros(len(labels), dtype=np.float64),
    )


def activity_log_evidence(model, observations, sequence_lengths, state_activity_mapping, graph):
    observations = np.asarray(observations)
    lengths = _validated_lengths(sequence_lengths, len(observations))
    state_posteriors = model.predict_proba(observations, lengths=lengths.tolist())
    if state_posteriors.shape != (len(observations), model.n_components):
        raise ValueError("HMM posterior output has an unexpected shape.")

    label_indices = {label: index for index, label in enumerate(graph.labels)}
    activity_probabilities = np.zeros(
        (len(observations), len(graph.labels)), dtype=np.float64
    )
    for state in range(model.n_components):
        activity = int(state_activity_mapping.get(state, -1))
        if activity in label_indices:
            activity_probabilities[:, label_indices[activity]] += state_posteriors[:, state]

    row_sums = activity_probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or not np.isfinite(activity_probabilities).all():
        raise ValueError("HMM did not provide finite evidence for mapped activities.")
    activity_probabilities /= row_sums
    activity_probabilities = np.maximum(activity_probabilities, 1e-12)
    activity_probabilities /= activity_probabilities.sum(axis=1, keepdims=True)
    return np.log(activity_probabilities)


def decode_activity_sequence(log_evidence, graph, sequence_lengths):
    log_evidence = np.asarray(log_evidence, dtype=np.float64)
    if log_evidence.ndim != 2 or log_evidence.shape[1] != len(graph.labels):
        raise ValueError("Evidence columns must correspond to graph activity labels.")
    if not np.isfinite(log_evidence).all():
        raise ValueError("Activity evidence must contain only finite log scores.")
    lengths = _validated_lengths(sequence_lengths, len(log_evidence))

    decoded = np.empty(len(log_evidence), dtype=np.int64)
    offset = 0
    for length_value in lengths:
        length = int(length_value)
        end = offset + length
        sequence_evidence = log_evidence[offset:end]
        scores = graph.log_start_probabilities + sequence_evidence[0]
        backpointers = np.empty((length, len(graph.labels)), dtype=np.int64)

        for time_index in range(1, length):
            candidates = scores[:, None] + graph.log_transition_probabilities
            best_previous = np.argmax(candidates, axis=0)
            scores = candidates[best_previous, np.arange(len(graph.labels))]
            scores += sequence_evidence[time_index]
            backpointers[time_index] = best_previous
            if not np.isfinite(scores).any():
                raise ValueError("No valid graph-constrained activity path exists.")

        state = int(np.argmax(scores))
        state_path = np.empty(length, dtype=np.int64)
        state_path[-1] = state
        for time_index in range(length - 1, 0, -1):
            state_path[time_index - 1] = backpointers[time_index, state_path[time_index]]
        decoded[offset:end] = np.asarray(graph.labels, dtype=np.int64)[state_path]
        offset = end

    return decoded
