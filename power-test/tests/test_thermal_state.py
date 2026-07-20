"""
Claim:
The thermal coordinate is the exact discrete first-order response to dynamic
power, starts from ambient for every run, and uses a time constant in seconds.

Plausible wrong implementations:
- Use total power or an accumulated sum instead of the supplied dynamic power.
- Reuse the previous run's terminal heat state.
- Use tau/dt rather than exp(-dt/tau), or update one bin late.
- Cool with the wrong sign when dynamic power falls to zero.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_surface import thermal_state
from thermal_ablation import dynamic_driver, heat_driver


def test_first_order_heating_cooling_and_run_reset():
    rho = 0.5
    tau_s = -1.0 / np.log(rho)
    power = np.asarray([8.0, 8.0, 0.0, 8.0, 0.0])
    run_id = np.asarray([0, 0, 0, 1, 1])

    state = thermal_state(power, run_id, dt_s=1.0, tau_s=tau_s)

    np.testing.assert_allclose(state, [4.0, 6.0, 3.0, 4.0, 2.0])


def test_constant_power_converges_to_its_input_without_overshoot():
    state = thermal_state(np.full(100, 7.0), np.zeros(100), 1.0, 5.0)

    assert np.all(np.diff(state) > 0.0)
    assert np.all(state < 7.0)
    np.testing.assert_allclose(state[-1], 7.0, rtol=3e-9)


def test_heat_driver_excludes_static_power_and_tp8_gate_is_node_level():
    names = ["tp", "tp_link", "busy_tp"]
    design = np.asarray([[8.0, 8.0, 0.0], [8.0, 8.0, 4.0]])
    driver = dynamic_driver(design, names, [10.0, 2.0, 3.0])

    np.testing.assert_allclose(driver, [0.0, 12.0])
    np.testing.assert_allclose(
        heat_driver(driver, np.asarray([4, 8]), "tp8_chassis"),
        [0.0, 12.0],
    )
