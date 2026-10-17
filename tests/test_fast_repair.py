import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from openwam_recovery.fast_repair import FastErrorEstimator, GeometricCorrector


def test_clean_motion_does_not_become_a_fault():
    scale = np.asarray([0.011, 0.013, 0.012])
    estimator = FastErrorEstimator(scale)
    rng = np.random.default_rng(4)
    estimates = []
    for _ in range(80):
        command = rng.uniform(-0.8, 0.8, 3)
        expected = command * scale
        estimates.append(estimator.update(command, expected, expected, 0.05))

    assert not any(item.stable for item in estimates)


def test_persistent_gain_fault_is_detected_after_short_warmup():
    scale = np.asarray([0.011, 0.013, 0.012])
    estimator = FastErrorEstimator(scale)
    rng = np.random.default_rng(5)
    estimates = []
    for step in range(80):
        command = rng.uniform(-0.8, 0.8, 3)
        expected = command * scale
        observed = expected if step < 25 else expected * 0.72
        estimates.append(estimator.update(command, observed, expected, 0.05))

    fault_steps = [i for i, item in enumerate(estimates) if item.stable]
    assert fault_steps
    assert fault_steps[0] >= 25
    assert fault_steps[0] <= 45


def test_geometric_correction_is_bounded():
    scale = np.asarray([0.011, 0.013, 0.012])
    estimator = FastErrorEstimator(scale)
    for _ in range(40):
        estimate = estimator.update(
            np.asarray([0.5, -0.4, 0.3]),
            np.asarray([0.5, -0.4, 0.3]) * scale * 0.72,
            np.asarray([0.5, -0.4, 0.3]) * scale,
            0.05,
        )
    correction = GeometricCorrector(scale, max_norm=0.12).action_delta(
        np.asarray([0.5, -0.4, 0.3]),
        np.asarray([0.04, -0.02, 0.03]),
        estimate,
    )
    assert np.linalg.norm(correction) <= 0.12 + 1e-9
