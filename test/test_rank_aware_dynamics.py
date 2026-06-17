import unittest

import numpy as np

from franka_sysid_tools.dynamics import BaseRegressorModel, _random_feasible_samples


class FakeRegressorModel(BaseRegressorModel):
    def __init__(self, full_row: np.ndarray, base_columns: list[int] | None = None):
        full_row = np.asarray(full_row, dtype=np.float64)
        if base_columns is None:
            base_columns = list(range(full_row.shape[1]))
        super().__init__(
            model=None,
            data=None,
            joint_names=[f"joint_{idx}" for idx in range(full_row.shape[0])],
            base_columns=np.asarray(base_columns, dtype=np.int64),
            structural_rank=len(base_columns),
            full_parameter_count=full_row.shape[1],
        )
        self._full_row = full_row

    def full_regressor(self, q, dq, ddq):
        return self._full_row


class RankAwareDynamicsTests(unittest.TestCase):
    def test_random_feasible_samples_are_deterministic(self):
        kwargs = {
            "center": np.array([0.0, 1.0]),
            "amplitudes": np.array([0.5, 0.25]),
            "joint_limits": np.array([[-1.0, 1.0], [0.0, 2.0]]),
            "max_joint_velocity": 0.7,
            "max_joint_acceleration": 1.3,
            "sample_count": 8,
            "seed": 123,
        }
        first = _random_feasible_samples(**kwargs)
        second = _random_feasible_samples(**kwargs)
        for lhs, rhs in zip(first, second):
            np.testing.assert_allclose(lhs, rhs)

    def test_broad_sampling_can_recover_more_rank_than_poor_trajectory(self):
        center = np.array([0.0, 0.0])
        amplitudes = np.array([1.0, 1.0])
        poor_q = np.zeros((20, 2))
        poor_dq = np.zeros_like(poor_q)
        poor_ddq = np.zeros_like(poor_q)
        broad_q, broad_dq, broad_ddq = _random_feasible_samples(
            center=center,
            amplitudes=amplitudes,
            joint_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            max_joint_velocity=1.0,
            max_joint_acceleration=1.0,
            sample_count=20,
            seed=7,
        )

        def fake_matrix(q, dq, ddq):
            return np.column_stack(
                (
                    np.ones(q.shape[0]),
                    q[:, 0],
                    q[:, 1],
                    dq[:, 0],
                    ddq[:, 0],
                )
            )

        poor = BaseRegressorModel.diagnostics_for_matrix(fake_matrix(poor_q, poor_dq, poor_ddq), rank_tolerance=1e-10, ridge=0.0)
        broad = BaseRegressorModel.diagnostics_for_matrix(
            fake_matrix(broad_q, broad_dq, broad_ddq),
            rank_tolerance=1e-10,
            ridge=0.0,
        )
        self.assertLess(poor.rank, broad.rank)

    def test_diagnostics_report_known_rank_and_condition(self):
        matrix = np.diag([3.0, 2.0, 0.0])
        diag = BaseRegressorModel.diagnostics_for_matrix(matrix, rank_tolerance=1e-8, ridge=1e-3)
        self.assertEqual(diag.rank, 2)
        self.assertEqual(diag.row_count, 3)
        self.assertEqual(diag.column_count, 3)
        self.assertAlmostEqual(diag.max_singular_value, 3.0)
        self.assertAlmostEqual(diag.min_singular_value, 0.0)
        self.assertTrue(np.isinf(diag.condition_number))

    def test_friction_columns_have_expected_shape_and_values(self):
        model = FakeRegressorModel(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        q = np.array([[0.0, 0.0]])
        dq = np.array([[0.5, -0.25]])
        ddq = np.array([[0.0, 0.0]])
        stacked = model.stacked_base_regressor(q, dq, ddq, include_friction=True)
        self.assertEqual(stacked.shape, (2, 7))
        expected_friction = np.array([[1.0, 0.0, 0.5, 0.0], [0.0, -1.0, 0.0, -0.25]])
        np.testing.assert_allclose(stacked[:, -4:], expected_friction)

    def test_objective_scores_order_well_conditioned_matrix_higher(self):
        well_conditioned = FakeRegressorModel(np.eye(2))
        poorly_conditioned = FakeRegressorModel(np.diag([1.0, 0.1]))
        q = np.array([[0.0, 0.0]])
        dq = np.array([[0.0, 0.0]])
        ddq = np.array([[0.0, 0.0]])
        for objective in ["d_opt", "conditioned_d_opt", "e_opt", "condition"]:
            lhs = well_conditioned.objective_score(
                q,
                dq,
                ddq,
                ridge=1e-6,
                condition_penalty=0.05,
                objective=objective,
                include_friction=False,
            )
            rhs = poorly_conditioned.objective_score(
                q,
                dq,
                ddq,
                ridge=1e-6,
                condition_penalty=0.05,
                objective=objective,
                include_friction=False,
            )
            self.assertTrue(np.isfinite(lhs))
            self.assertTrue(np.isfinite(rhs))
            self.assertGreater(lhs, rhs)


if __name__ == "__main__":
    unittest.main()
