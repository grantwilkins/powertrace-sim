#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class FacilityLayout:
    rows: int = 10
    racks_per_row: int = 6
    nodes_per_rack: int = 4

    @property
    def n_nodes(self) -> int:
        return int(self.rows) * int(self.racks_per_row) * int(self.nodes_per_rack)

    def node_id_to_coords(self, node_id: int) -> Tuple[int, int, int]:
        npr = int(self.nodes_per_rack)
        rpr = int(self.racks_per_row)
        per_row = rpr * npr
        row = int(node_id) // per_row
        rem = int(node_id) % per_row
        rack = rem // npr
        node = rem % npr
        return int(row), int(rack), int(node)

    def iter_node_ids(self) -> range:
        return range(self.n_nodes)

    def iter_nodes(self) -> Iterator[Tuple[int, int, int]]:
        for node_id in self.iter_node_ids():
            yield self.node_id_to_coords(int(node_id))


def downsample_mean(values: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    f = int(factor)
    if f <= 0:
        raise ValueError("downsample factor must be >= 1")
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if arr.size % f != 0:
        raise ValueError(f"Array length {arr.size} not divisible by factor {f}")
    return np.mean(arr.reshape(-1, f), axis=1).astype(np.float64)
