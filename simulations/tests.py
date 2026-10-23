"""Checks for the closed forms and Monte Carlo targets."""

from __future__ import annotations

import unittest

import numpy as np

from simulations.econometrics import (
    iv_versus_ols,
    ols_sampling,
    omitted_variable_bias,
    population_omitted_slope,
    rdd_local_linear,
)
from simulations.macro import (
    golden_rule_saving,
    nk_impulse,
    nk_residuals,
    solow_steady_state,
    solow_transition,
    steady_consumption,
)
from simulations.micro import (
    APPLICANTS,
    DEPARTMENTS,
    becker_crime_rate,
    becker_crime_rate_uniform,
    cournot_dynamics,
    cournot_nash,
    deferred_acceptance,
    fictitious_play_matching_pennies,
    is_stable,
    simulate_posted_price_revenue,
    uniform_optimal_price,
    uniform_seller_profit,
)


class MicroTests(unittest.TestCase):
    def test_uniform_monopoly_price_matches_the_lean_formula(self) -> None:
        self.assertAlmostEqual(uniform_optimal_price(0.0, 1.0, 0.0), 0.5)
        self.assertAlmostEqual(uniform_optimal_price(0.0, 1.0, 0.2), 0.6)
        self.assertAlmostEqual(uniform_optimal_price(0.0, 1.0, 3.0), 1.0)
        self.assertAlmostEqual(uniform_optimal_price(0.0, 1.0, -3.0), 0.0)
        self.assertAlmostEqual(uniform_seller_profit(0.5, 0.0, 1.0, 0.0), 0.25)

    def test_higher_continuation_value_raises_the_posted_price(self) -> None:
        low = uniform_optimal_price(0.0, 1.0, 0.0)
        high = uniform_optimal_price(0.0, 1.0, 0.4)
        self.assertGreater(high, low)

    def test_sampled_revenue_tracks_the_analytic_profit(self) -> None:
        analytic = uniform_seller_profit(0.5, 0.0, 1.0, 0.0)
        sampled = simulate_posted_price_revenue(0.5, 0.0, 1.0, 0.0, n=50_000, seed=4)
        self.assertAlmostEqual(sampled, analytic, delta=0.01)

    def test_becker_cutoff_lowers_crime_as_the_sanction_rises(self) -> None:
        gains = np.linspace(0.0, 1.0, 1001)
        soft = becker_crime_rate(gains, probability=0.2, fine=1.0)
        harsh = becker_crime_rate(gains, probability=0.2, fine=3.0)
        self.assertAlmostEqual(soft, becker_crime_rate_uniform(0.2, 1.0), places=3)
        self.assertGreater(soft, harsh)
        self.assertAlmostEqual(becker_crime_rate_uniform(1.0, 0.25), 0.75)

    def test_matching_pennies_frequencies_settle_near_one_half(self) -> None:
        frequency = fictitious_play_matching_pennies(3_000, seed=1)
        self.assertTrue(np.allclose(frequency[-500:].mean(axis=0), 0.5, atol=0.05))

    def test_cournot_dynamics_reach_the_nash_quantity(self) -> None:
        nash = cournot_nash(10.0, 1.0, 1.0)
        path = cournot_dynamics(np.array([0.2, 6.0]), 10.0, 1.0, 1.0, rounds=30)
        self.assertAlmostEqual(path[-1, 0], nash, places=4)
        self.assertAlmostEqual(path[-1, 1], nash, places=4)

    def test_who_proposes_changes_the_stable_match(self) -> None:
        applicant_optimal = deferred_acceptance(APPLICANTS, DEPARTMENTS)
        department_optimal = deferred_acceptance(DEPARTMENTS, APPLICANTS)
        department_optimal = {reviewer: dept for dept, reviewer in department_optimal.items()}
        self.assertEqual(applicant_optimal, {"Ada": "Theory", "Grace": "Metrics"})
        self.assertEqual(department_optimal, {"Ada": "Metrics", "Grace": "Theory"})
        self.assertTrue(is_stable(applicant_optimal, APPLICANTS, DEPARTMENTS))
        self.assertTrue(is_stable(department_optimal, APPLICANTS, DEPARTMENTS))
        self.assertNotEqual(applicant_optimal, department_optimal)


class MacroTests(unittest.TestCase):
    def test_solow_capital_converges_to_the_closed_form(self) -> None:
        steady = solow_steady_state(0.3, 0.08, 0.35)
        for start in (0.3 * steady, 2.0 * steady):
            path = solow_transition(start, 0.3, 0.08, 0.35, periods=200)
            self.assertAlmostEqual(path[-1], steady, delta=1e-3)

    def test_golden_rule_saving_rate_maximizes_steady_consumption(self) -> None:
        alpha, depreciation = 0.35, 0.08
        self.assertAlmostEqual(golden_rule_saving(alpha), alpha)
        peak = steady_consumption(alpha, depreciation, alpha)
        for saving in (0.1, 0.2, 0.5, 0.7):
            self.assertLess(steady_consumption(saving, depreciation, alpha), peak)

    def test_new_keynesian_loadings_solve_the_three_equations(self) -> None:
        params = dict(beta=0.99, sigma=1.0, kappa=0.15, phi_pi=1.5, phi_x=0.5, rho=0.6)
        for shock in ("monetary", "demand"):
            is_gap, pc_gap = nk_residuals(shock=shock, **params)
            self.assertAlmostEqual(is_gap, 0.0, places=9)
            self.assertAlmostEqual(pc_gap, 0.0, places=9)
        monetary = nk_impulse(shock="monetary", **params)
        demand = nk_impulse(shock="demand", **params)
        self.assertLess(monetary["output"][0], 0.0)
        self.assertLess(monetary["inflation"][0], 0.0)
        self.assertGreater(demand["output"][0], 0.0)
        self.assertGreater(demand["inflation"][0], 0.0)


class EconometricsTests(unittest.TestCase):
    def test_ols_is_centered_on_the_true_slope(self) -> None:
        slopes = ols_sampling(n=250, reps=600, seed=11)
        self.assertAlmostEqual(float(slopes.mean()), 2.0, delta=0.05)

    def test_omitted_variable_bias_has_the_population_sign(self) -> None:
        target = population_omitted_slope()
        self.assertGreater(target, 1.3)
        slopes = omitted_variable_bias(n=800, reps=300, seed=12)
        self.assertAlmostEqual(float(slopes.mean()), target, delta=0.08)

    def test_instrument_removes_the_endogeneity_bias(self) -> None:
        ols, iv = iv_versus_ols(n=700, reps=300, seed=13)
        self.assertLess(abs(float(iv.mean()) - 2.0), abs(float(ols.mean()) - 2.0))
        self.assertAlmostEqual(float(iv.mean()), 2.0, delta=0.12)
        self.assertGreater(float(ols.mean()), 2.4)

    def test_regression_discontinuity_recovers_the_jump(self) -> None:
        estimate = rdd_local_linear(n=4_000, effect=1.5, bandwidth=0.4, seed=14)
        self.assertAlmostEqual(estimate, 1.5, delta=0.2)


if __name__ == "__main__":
    unittest.main()
