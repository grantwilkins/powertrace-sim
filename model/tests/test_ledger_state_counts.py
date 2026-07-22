"""Ledger state counts must preserve the exact request interval semantics.

A wrong optimization could count arrivals on the right-open bin edge, retain a
request at its completion edge, or allocate a request-by-time matrix.
"""
from __future__ import annotations

import numpy as np

from model.timing.ledger import request_state_counts


def test_event_counts_match_interval_definition_at_edges() -> None:
    arrivals = np.asarray([0.0, 0.25, 0.25, 0.63])
    admitted = np.asarray([0.0, 0.25, 0.50, 0.75])
    completed = np.asarray([0.25, 0.75, 1.00, 1.25])
    edges = np.arange(0.25, 1.51, 0.25)

    active, running = request_state_counts(
        arrivals, admitted, completed, edges, dt=0.25
    )
    expected_active = np.asarray([
        np.count_nonzero((arrivals < edge) & (completed > edge)) for edge in edges
    ])
    expected_running = np.asarray([
        np.count_nonzero(
            (arrivals < edge) & (admitted <= edge) & (completed > edge)
        )
        for edge in edges
    ])

    np.testing.assert_array_equal(active, expected_active)
    np.testing.assert_array_equal(running, expected_running)
