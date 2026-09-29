"""Command line interface for convertible-bond-lab."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .analytics import (
    breakeven_years,
    conversion_premium,
    greeks,
    investment_value,
    parity,
    risk_premium,
    straight_bond_ytm,
)
from .instruments import CallWindow, ConvertibleBond, PutDate
from .tree import price_convertible, price_refined


def _parse_call(text: str) -> CallWindow:
    """Parse start:end:price[:trigger], e.g. 3:5:100 or 3:5:100:1.3."""
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "call must be start:end:price[:trigger], e.g. 3:5:100:1.3"
        )
    trigger = float(parts[3]) if len(parts) == 4 else None
    return CallWindow(float(parts[0]), float(parts[1]), float(parts[2]), trigger)


def _parse_put(text: str) -> PutDate:
    """Parse time:price, e.g. 3:100."""
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("put must be time:price, e.g. 3:100")
    return PutDate(float(parts[0]), float(parts[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convertible",
        description="Price a convertible bond and show its desk analytics.",
    )
    p.add_argument("--spot", type=float, default=25.0, help="share price")
    p.add_argument("--vol", type=float, default=0.32, help="share volatility, decimal")
    p.add_argument("--rate", type=float, default=0.042, help="risk-free rate, decimal")
    p.add_argument("--dividend", type=float, default=0.005, help="dividend yield")
    p.add_argument("--coupon", type=float, default=0.02, help="annual coupon, decimal")
    p.add_argument("--frequency", type=int, default=2, help="coupons per year")
    p.add_argument("--maturity", type=float, default=5.0, help="years to maturity")
    p.add_argument("--ratio", type=float, default=4.0, help="shares per 100 face")
    p.add_argument("--spread", type=float, default=0.025, help="credit spread, decimal")
    p.add_argument("--face", type=float, default=100.0, help="face value")
    p.add_argument("--steps", type=int, default=400, help="binomial steps")
    p.add_argument(
        "--call", action="append", default=[], type=_parse_call, metavar="S:E:P[:T]",
        help="issuer call window, repeatable",
    )
    p.add_argument(
        "--put", action="append", default=[], type=_parse_put, metavar="T:P",
        help="holder put date, repeatable",
    )
    p.add_argument(
        "--raw", action="store_true",
        help="report the raw tree price instead of the extrapolated limit",
    )
    p.add_argument(
        "--grid", type=str, default="",
        help="comma-separated share prices for a sensitivity table",
    )
    return p


def _bond_from_args(args: argparse.Namespace) -> ConvertibleBond:
    return ConvertibleBond(
        face=args.face,
        coupon_rate=args.coupon,
        frequency=args.frequency,
        maturity=args.maturity,
        conversion_ratio=args.ratio,
        call_windows=list(args.call),
        put_dates=list(args.put),
        credit_spread=args.spread,
    )


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _report(bond: ConvertibleBond, args: argparse.Namespace) -> List[str]:
    engine = price_convertible if args.raw else price_refined
    res = engine(bond, args.spot, args.vol, args.rate, args.dividend, args.steps)
    floor = investment_value(bond, args.rate)
    par = parity(bond, args.spot)
    g = greeks(bond, args.spot, args.vol, args.rate, args.dividend, args.steps)
    be = breakeven_years(bond, res.price, args.spot, args.dividend)

    lines: List[str] = []
    lines.append("=" * 68)
    lines.append("CONVERTIBLE BOND  (values per 100 of face)")
    lines.append("=" * 68)
    lines.append(
        f"  {bond.coupon_rate * 100:.2f}% coupon, {bond.maturity:g}y, "
        f"ratio {bond.conversion_ratio:g} shares/bond"
    )
    lines.append(
        f"  conversion price {bond.conversion_price:.2f}   "
        f"share {args.spot:.2f}   vol {args.vol * 100:.1f}%"
    )
    lines.append(
        f"  risk-free {args.rate * 100:.2f}%   credit spread "
        f"{bond.credit_spread * 1e4:.0f}bp   dividend {args.dividend * 100:.2f}%"
    )
    if bond.call_windows:
        for w in bond.call_windows:
            trig = f", trigger {w.trigger:g}x conv px" if w.trigger else ""
            lines.append(
                f"  callable {w.start:g}y-{w.end:g}y at {w.price:.2f}{trig}"
            )
    for put in bond.put_dates:
        lines.append(f"  putable at {put.time:g}y at {put.price:.2f}")
    lines.append("-" * 68)
    label = "raw tree" if args.raw else "extrapolated"
    lines.append(f"  Fair value ({label})".ljust(28) + f"{res.price:10.4f}")
    lines.append(f"    cash-only component     {res.cash_component:10.4f}")
    lines.append(f"    equity component        {res.equity_component:10.4f}")
    lines.append(f"  Parity (conversion value) {par:10.4f}")
    lines.append(f"  Investment value (floor)  {floor:10.4f}")
    lines.append("-" * 68)
    lines.append(f"  Conversion premium        {_fmt_pct(conversion_premium(res.price, par)):>10}")
    lines.append(f"  Risk premium over floor   {_fmt_pct(risk_premium(res.price, floor)):>10}")
    lines.append(
        f"  Breakeven                 "
        f"{'n/a' if be is None else format(be, '.2f') + ' yrs':>10}"
    )
    lines.append(
        f"  Straight-bond YTM at floor{straight_bond_ytm(bond, floor) * 100:9.3f}%"
    )
    lines.append("-" * 68)
    lines.append(f"  Delta (shares per bond)   {g.delta:10.4f}")
    lines.append(f"  Equity participation      {g.equity_participation * 100:9.2f}%")
    lines.append(f"  Gamma                     {g.gamma:10.4f}")
    lines.append(f"  Vega  (per 1 vol point)   {g.vega:10.4f}")
    lines.append(f"  Rho   (per 1bp)           {g.rho:10.4f}")
    lines.append(f"  Spread dv (per 1bp)       {g.spread_sensitivity:10.4f}")

    grid = [float(x) for x in args.grid.split(",")] if args.grid else []
    if grid:
        lines.append("=" * 68)
        lines.append(
            f"  {'share':>8}{'price':>11}{'parity':>10}{'premium':>10}"
            f"{'cash':>10}{'delta':>9}"
        )
        lines.append("-" * 68)
        for s in grid:
            r = engine(bond, s, args.vol, args.rate, args.dividend, args.steps)
            gg = greeks(bond, s, args.vol, args.rate, args.dividend, args.steps)
            prem = conversion_premium(r.price, r.parity)
            lines.append(
                f"  {s:8.2f}{r.price:11.3f}{r.parity:10.2f}"
                f"{_fmt_pct(prem):>10}{r.cash_component:10.3f}{gg.delta:9.3f}"
            )
    lines.append("=" * 68)
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bond = _bond_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for line in _report(bond, args):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
