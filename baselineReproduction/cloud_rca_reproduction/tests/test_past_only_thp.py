import unittest
import numpy as np

from scripts.fit_past_only_thp import (
    past_effects,
    build_fixture,
    objective_and_grad,
)


class Tests(unittest.TestCase):

    def test_strictly_past(self):
        x = np.zeros((2, 3, 2))
        x[0, 0, 0] = 1

        h = past_effects(x, decay=np.log(2))

        np.testing.assert_allclose(
            h[0, :, 0], [0, 0.5, 0.25]
        )
        np.testing.assert_allclose(h[1], 0)

    def test_analytical_gradient(self):
        x = build_fixture(steps=10, devices=2)
        h = past_effects(x)

        params = np.array([
            0.3, 0.2, 0.1, 0.15, 0.04, 0.03
        ])

        _, analytical = objective_and_grad(params, x, h)

        for i in range(len(params)):
            plus = params.copy()
            minus = params.copy()

            plus[i] += 1e-6
            minus[i] -= 1e-6

            numerical = (
                objective_and_grad(plus, x, h)[0]
                - objective_and_grad(minus, x, h)[0]
            ) / (2e-6)

            self.assertAlmostEqual(
                analytical[i], numerical, places=5
            )


if __name__ == "__main__":
    unittest.main()
