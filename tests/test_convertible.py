"""Tests for convertible-bond-lab.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convertible.analytics import (  # noqa: E402
    breakeven_years,
    conversion_premium,
    greeks,
    investment_value,
    parity,
    risk_premium,
    straight_bond_ytm,
)
from convertible.cli import main  # noqa: E402
from convertible.instruments import CallWindow, ConvertibleBond, PutDate  # noqa: E402
from convertible.tree import price_convertible, price_refined  # noqa: E402

R = 0.042
Q = 0.005
VOL = 0.32


def sample(**kwargs) -> ConvertibleBond:
    base = dict(
        coupon_rate=0.02,
        frequency=2,
        maturity=5.0,
        conversion_ratio=4.0,
        credit_spread=0.025,
    )
    base.update(kwargs)
    return ConvertibleBond(**base)


class TestInstruments(unittest.TestCase):
    def test_coupon_times_step_back_from_maturity(self):
        b = sample(maturity=3.0, frequency=2)
        self.assertEqual(b.coupon_times(), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def test_zero_coupon_has_no_coupon_dates(self):
        b = sample(coupon_rate=0.0, frequency=0)
        self.assertEqual(b.coupon_times(), [])
        self.assertEqual(b.coupon_amount, 0.0)

    def test_cashflows_merge_final_coupon_into_redemption(self):
        b = sample(maturity=2.0, coupon_rate=0.04, frequency=2)
        flows = b.cashflows()
        self.assertEqual(len(flows), 4)
        self.assertAlmostEqual(flows[-1][0], 2.0)
        self.assertAlmostEqual(flows[-1][1], 102.0)
        self.assertAlmostEqual(sum(a for _, a in flows), 108.0)

    def test_conversion_price_is_face_over_ratio(self):
        self.assertAlmostEqual(sample(conversion_ratio=4.0).conversion_price, 25.0)
        self.assertEqual(sample(conversion_ratio=0.0).conversion_price, float("inf"))

    def test_cheapest_live_call_wins(self):
        b = sample(
            call_windows=[CallWindow(2, 5, 104), CallWindow(3, 5, 100)]
        )
        self.assertIsNone(b.call_price_at(1.0))
        self.assertAlmostEqual(b.call_price_at(2.5).price, 104.0)
        self.assertAlmostEqual(b.call_price_at(4.0).price, 100.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ConvertibleBond(maturity=0)
        with self.assertRaises(ValueError):
            ConvertibleBond(coupon_rate=0.05, frequency=0)
        with self.assertRaises(ValueError):
            ConvertibleBond(credit_spread=-0.01)
        with self.assertRaises(ValueError):
            ConvertibleBond(maturity=3, call_windows=[CallWindow(1, 4, 100)])
        with self.assertRaises(ValueError):
            CallWindow(4, 2, 100)


class TestStraightBondLimit(unittest.TestCase):
    """With no conversion right the tree must reproduce a discounted bond exactly."""

    def analytic(self, bond: ConvertibleBond, rate: float) -> float:
        return sum(a * math.exp(-rate * t) for t, a in bond.cashflows())

    def test_no_conversion_no_spread(self):
        b = sample(conversion_ratio=0.0, credit_spread=0.0, coupon_rate=0.05)
        res = price_convertible(b, 50.0, VOL, R, Q, steps=200)
        self.assertAlmostEqual(res.price, self.analytic(b, R), places=8)

    def test_no_conversion_with_spread_uses_risky_rate(self):
        b = sample(conversion_ratio=0.0, credit_spread=0.03, coupon_rate=0.05)
        res = price_convertible(b, 50.0, VOL, R, Q, steps=200)
        self.assertAlmostEqual(res.price, self.analytic(b, R + 0.03), places=8)
        self.assertAlmostEqual(res.price, res.cash_component, places=8)
        self.assertAlmostEqual(res.equity_component, 0.0, places=8)

    def test_investment_value_matches_tree_floor(self):
        b = sample(conversion_ratio=0.0)
        self.assertAlmostEqual(
            investment_value(b, R),
            price_convertible(b, 1.0, VOL, R, Q, steps=200).price,
            places=8,
        )

    def test_zero_coupon_discounts_to_redemption(self):
        b = sample(coupon_rate=0.0, frequency=0, conversion_ratio=0.0, credit_spread=0.02)
        res = price_convertible(b, 10.0, VOL, R, Q, steps=150)
        self.assertAlmostEqual(res.price, 100 * math.exp(-(R + 0.02) * 5.0), places=8)


class TestNoArbitrageBounds(unittest.TestCase):
    def test_price_never_below_parity_or_floor(self):
        b = sample()
        floor = investment_value(b, R)
        for spot in (1, 5, 10, 18, 25, 33, 60, 120):
            res = price_convertible(b, spot, VOL, R, Q, steps=250)
            self.assertGreaterEqual(res.price, res.parity - 1e-9, f"parity at {spot}")
            self.assertGreaterEqual(res.price, floor - 1e-6, f"floor at {spot}")

    def test_deep_in_the_money_converges_to_parity(self):
        b = sample(coupon_rate=0.0, frequency=0, credit_spread=0.0)
        res = price_convertible(b, 400.0, VOL, R, 0.0, steps=300)
        self.assertAlmostEqual(res.price, res.parity, delta=0.5)
        self.assertAlmostEqual(res.cash_component, 0.0, delta=0.5)

    def test_price_rises_with_share_price(self):
        b = sample()
        prices = [
            price_convertible(b, s, VOL, R, Q, steps=200).price
            for s in (5, 10, 15, 20, 25, 30, 40, 55)
        ]
        for lo, hi in zip(prices, prices[1:]):
            self.assertLess(lo, hi)

    def test_price_rises_with_volatility(self):
        b = sample()
        prices = [
            price_convertible(b, 22.0, v, R, Q, steps=250).price
            for v in (0.10, 0.20, 0.30, 0.45, 0.60)
        ]
        for lo, hi in zip(prices, prices[1:]):
            self.assertLess(lo, hi)

    def test_wider_credit_spread_lowers_price(self):
        prices = [
            price_convertible(sample(credit_spread=s), 20.0, VOL, R, Q, steps=250).price
            for s in (0.0, 0.01, 0.03, 0.06)
        ]
        for hi, lo in zip(prices, prices[1:]):
            self.assertGreater(hi, lo)

    def test_cash_component_shrinks_as_equity_takes_over(self):
        b = sample()
        cash = [
            price_convertible(b, s, VOL, R, Q, steps=250).cash_component
            for s in (8, 15, 25, 40, 80)
        ]
        for hi, lo in zip(cash, cash[1:]):
            self.assertGreater(hi, lo)

    def test_raw_tree_converges_upward_with_shrinking_steps(self):
        """Raw convergence is monotone and each doubling shrinks the increment."""
        b = sample()
        prices = [
            price_convertible(b, 25.0, VOL, R, Q, steps=n).price
            for n in (100, 200, 400, 800, 1600)
        ]
        diffs = [hi - lo for lo, hi in zip(prices, prices[1:])]
        for d in diffs:
            self.assertGreater(d, 0.0)
        for big, small in zip(diffs, diffs[1:]):
            self.assertLess(small, big)

    def test_extrapolation_is_stable_across_base_step_counts(self):
        b = sample()
        refined = [
            price_refined(b, 25.0, VOL, R, Q, steps=n).price for n in (100, 400, 1600)
        ]
        self.assertAlmostEqual(refined[0], refined[-1], delta=0.02)
        self.assertAlmostEqual(refined[1], refined[-1], delta=0.02)
        # and it beats the raw tree it is built from
        raw = price_convertible(b, 25.0, VOL, R, Q, steps=800).price
        limit = refined[-1]
        self.assertLess(abs(refined[1] - limit), abs(raw - limit))

    def test_extrapolation_leaves_a_straight_bond_alone(self):
        b = sample(conversion_ratio=0.0, credit_spread=0.03, coupon_rate=0.05)
        exact = sum(a * math.exp(-(R + 0.03) * t) for t, a in b.cashflows())
        self.assertAlmostEqual(price_refined(b, 50.0, VOL, R, Q, 100).price, exact, places=7)


class TestEmbeddedOptions(unittest.TestCase):
    def test_issuer_call_caps_the_price(self):
        free = sample()
        called = sample(call_windows=[CallWindow(0.0, 5.0, 100.0)])
        for spot in (20.0, 25.0, 30.0):
            self.assertLess(
                price_convertible(called, spot, VOL, R, Q, steps=250).price,
                price_convertible(free, spot, VOL, R, Q, steps=250).price,
            )

    def test_soft_call_trigger_is_worth_more_to_the_holder(self):
        hard = sample(call_windows=[CallWindow(2.0, 5.0, 100.0)])
        soft = sample(call_windows=[CallWindow(2.0, 5.0, 100.0, trigger=1.6)])
        self.assertGreater(
            price_convertible(soft, 25.0, VOL, R, Q, steps=250).price,
            price_convertible(hard, 25.0, VOL, R, Q, steps=250).price,
        )

    def test_holder_put_supports_a_busted_convert(self):
        plain = sample()
        putable = sample(put_dates=[PutDate(3.0, 100.0)])
        self.assertGreater(
            price_convertible(putable, 6.0, VOL, R, Q, steps=300).price,
            price_convertible(plain, 6.0, VOL, R, Q, steps=300).price,
        )

    def test_put_at_maturity_floors_at_the_put_price(self):
        b = sample(
            conversion_ratio=0.0,
            coupon_rate=0.0,
            frequency=0,
            credit_spread=0.0,
            put_dates=[PutDate(5.0, 130.0)],
        )
        res = price_convertible(b, 1.0, VOL, R, Q, steps=200)
        self.assertAlmostEqual(res.price, 130.0 * math.exp(-R * 5.0), places=6)

    def test_call_below_parity_forces_conversion_not_a_giveaway(self):
        b = sample(call_windows=[CallWindow(0.0, 5.0, 100.0)])
        res = price_convertible(b, 60.0, VOL, R, Q, steps=250)
        self.assertGreaterEqual(res.price, res.parity - 1e-9)


class TestGreeks(unittest.TestCase):
    def test_delta_sits_between_zero_and_the_conversion_ratio(self):
        b = sample()
        for spot in (5, 12, 25, 45, 100):
            g = greeks(b, spot, VOL, R, Q, steps=250)
            self.assertGreater(g.delta, 0.0)
            self.assertLessEqual(g.delta, b.conversion_ratio + 1e-6)
            self.assertLessEqual(g.equity_participation, 1.0 + 1e-6)

    def test_delta_increases_with_the_share_price(self):
        b = sample()
        deltas = [greeks(b, s, VOL, R, Q, steps=300).delta for s in (8, 15, 25, 40, 80)]
        for lo, hi in zip(deltas, deltas[1:]):
            self.assertLess(lo, hi)

    def test_gamma_is_positive_for_an_uncalled_convert(self):
        self.assertGreater(greeks(sample(), 25.0, VOL, R, Q, steps=300).gamma, 0.0)

    def test_vega_positive_rho_and_spread_negative(self):
        g = greeks(sample(), 20.0, VOL, R, Q, steps=300)
        self.assertGreater(g.vega, 0.0)
        self.assertLess(g.spread_sensitivity, 0.0)
        self.assertLess(g.rho, 0.0)

    def test_participation_is_zero_without_a_conversion_right(self):
        g = greeks(sample(conversion_ratio=0.0), 25.0, VOL, R, Q, steps=150)
        self.assertEqual(g.equity_participation, 0.0)


class TestDerivedMeasures(unittest.TestCase):
    def test_parity_and_premiums(self):
        b = sample()
        self.assertAlmostEqual(parity(b, 25.0), 100.0)
        self.assertAlmostEqual(conversion_premium(118.0, 100.0), 0.18)
        self.assertAlmostEqual(risk_premium(118.0, 80.0), 0.475)
        self.assertIsNone(conversion_premium(118.0, 0.0))
        self.assertIsNone(risk_premium(118.0, 0.0))

    def test_ytm_round_trip(self):
        b = sample(coupon_rate=0.05)
        for rate in (0.01, 0.045, 0.09, 0.15):
            price = sum(a * math.exp(-rate * t) for t, a in b.cashflows())
            self.assertAlmostEqual(straight_bond_ytm(b, price), rate, places=7)

    def test_ytm_rejects_impossible_prices(self):
        b = sample(coupon_rate=0.05)
        with self.assertRaises(ValueError):
            straight_bond_ytm(b, 0.0)
        with self.assertRaises(ValueError):
            straight_bond_ytm(b, 1e9)

    def test_breakeven(self):
        b = sample(coupon_rate=0.02)
        self.assertAlmostEqual(breakeven_years(b, 118.0, 25.0, 0.0), 9.0)
        self.assertEqual(breakeven_years(b, 95.0, 25.0, 0.0), 0.0)
        # A dividend yield richer than the coupon never pays the premium back.
        self.assertIsNone(breakeven_years(b, 118.0, 25.0, 0.05))
        self.assertIsNone(breakeven_years(sample(conversion_ratio=0.0), 118.0, 25.0))


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_cli_runs_and_reports(self):
        code, out, _ = self.run_cli(
            ["--spot", "25", "--steps", "120", "--grid", "18,25,32"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Fair value (extrapolated)", out)
        self.assertIn("Conversion premium", out)
        self.assertIn("share", out)

    def test_cli_parses_calls_and_puts(self):
        code, out, _ = self.run_cli(
            ["--steps", "80", "--call", "2:5:100:1.3", "--put", "3:100"]
        )
        self.assertEqual(code, 0)
        self.assertIn("callable 2y-5y at 100.00", out)
        self.assertIn("trigger 1.3x conv px", out)
        self.assertIn("putable at 3y at 100.00", out)

    def test_cli_raw_flag_switches_engine(self):
        code, out, _ = self.run_cli(["--steps", "80", "--raw"])
        self.assertEqual(code, 0)
        self.assertIn("Fair value (raw tree)", out)

    def test_cli_rejects_a_bad_bond(self):
        code, _, err = self.run_cli(["--maturity", "0", "--steps", "40"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()
