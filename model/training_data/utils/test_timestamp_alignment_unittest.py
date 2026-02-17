import unittest

from model.training_data.utils.request_timestamps import compute_aligned_request_timestamps


class TestTimestampAlignment(unittest.TestCase):
    def test_recorded_passthrough(self):
        out = compute_aligned_request_timestamps(
            power_timestamps_s=[1000.0, 1010.0],
            poisson_rate=10.0,
            recorded_request_timestamps_s=[2000.0, 2005.0, 2010.0],
            timestamp_source="recorded",
        )
        self.assertEqual(out, [2000.0, 2005.0, 2010.0])

    def test_recorded_scaled_maps_into_power_window(self):
        out = compute_aligned_request_timestamps(
            power_timestamps_s=[1000.0, 1010.0],
            poisson_rate=10.0,
            recorded_request_timestamps_s=[2000.0, 2005.0, 2010.0],
            timestamp_source="recorded_scaled_or_poisson",
        )
        self.assertEqual(out[0], 1000.0)
        self.assertEqual(out[-1], 1010.0)

    def test_poisson_uses_time_step_scaling_when_present(self):
        out = compute_aligned_request_timestamps(
            power_timestamps_s=[1000.0, 1010.0],
            poisson_rate=10.0,
            time_steps=[0.0, 5.0, 10.0],
            timestamp_source="poisson",
        )
        self.assertEqual(out, [1000.0, 1005.0, 1010.0])

    def test_recorded_requires_timestamps(self):
        with self.assertRaises(ValueError):
            compute_aligned_request_timestamps(
                power_timestamps_s=[1000.0, 1010.0],
                poisson_rate=10.0,
                recorded_request_timestamps_s=[0.0, 0.0],
                timestamp_source="recorded",
            )


if __name__ == "__main__":
    unittest.main()
