import numpy as np

from scripts.analysis.run_rcc_drift_diagnostic import solve_redundant_shifts


def test_redundant_shift_solver_recovers_block_trajectory() -> None:
    expected_y = np.asarray([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    expected_x = np.asarray([0.0, -0.5, -1.0, -1.5], dtype=np.float64)
    pairs = []
    for i in range(expected_x.size):
        for j in range(i + 1, expected_x.size):
            pairs.append((i, j, expected_y[j] - expected_y[i], expected_x[j] - expected_x[i]))

    drift_y, drift_x = solve_redundant_shifts(expected_x.size, pairs)

    np.testing.assert_allclose(drift_y, expected_y, atol=1e-7)
    np.testing.assert_allclose(drift_x, expected_x, atol=1e-7)
