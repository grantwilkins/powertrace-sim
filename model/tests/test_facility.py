import numpy as np

from scripts.eval.facility import FacilityLayout, downsample_mean


def test_facility_layout_node_to_coords():
    layout = FacilityLayout(rows=10, racks_per_row=6, nodes_per_rack=4)
    row, rack, node = layout.node_id_to_coords(137)
    assert (row, rack, node) == (5, 4, 1)


def test_facility_layout_coords_to_node_roundtrip():
    layout = FacilityLayout(rows=10, racks_per_row=6, nodes_per_rack=4)
    row, rack, node = layout.node_id_to_coords(173)
    reconstructed = (
        row * layout.racks_per_row * layout.nodes_per_rack
        + rack * layout.nodes_per_rack
        + node
    )
    assert reconstructed == 173


def test_downsample_mean_basic():
    values = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    out = downsample_mean(values, factor=2)
    assert np.allclose(out, np.asarray([2.0, 6.0], dtype=np.float64))
