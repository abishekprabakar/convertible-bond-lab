"""Derived measures for convertible bonds: floors, premiums, breakeven, greeks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .instruments import ConvertibleBond
from .tree import price_convertible


def parity(bond: ConvertibleBond, spot: float) -> float:
    """Conversion value: what the shares are worth per 100 of face."""
    return bond.conversion_ratio * spot


def investment_value(bond: ConvertibleBond, risk_free: float) -> float:
    """The straight-bond floor: cashflows discounted at the credit-risky rate."""
    rate = risk_free + bond.credit_spread
    return sum(amt * math.exp(-rate * t) for t, amt in bond.cashflows())


def conversion_premium(price: float, par: float) -> Optional[float]:
    """How much more than the shares the convertible costs, as a fraction."""
    if par <= 0:
        return None
    return price / par - 1.0


def risk_premium(price: float, floor: float) -> Optional[float]:
    """How much more than the bond floor the convertible costs, as a fraction."""
    if floor <= 0:
        return None
    return price / floor - 1.0


def straight_bond_ytm(
    bond: ConvertibleBond,
    price: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Continuously compounded yield that discounts the cashflows to `price`."""
    if price <= 0:
        raise ValueError("price must be positive")
    flows = bond.cashflows()

    def pv(rate: float) -> float:
        return sum(amt * math.exp(-rate * t) for t, amt in flows)

    lo, hi = -0.99, 5.0
    if pv(lo) < price:
        raise ValueError("price above the maximum attainable present value")
    if pv(hi) > price:
        raise ValueError("price below the minimum attainable present value")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if pv(mid) > price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def breakeven_years(
    bond: ConvertibleBond,
    price: float,
    spot: float,
    dividend_yield: float = 0.0,
) -> Optional[float]:
    """Years of income advantage needed to recover the conversion premium.

    The classic desk measure: the premium paid over parity, divided by the
    convertible's annual income minus the dividends forgone by not holding the
    shares outright. None when the convert yields no more than the stock.
    """
    par = parity(bond, spot)
    if par <= 0:
        return None
    premium = price - par
    if premium <= 0:
        return 0.0
    bond_income = bond.face * bond.coupon_rate
    share_income = bond.conversion_ratio * spot * dividend_yield
    advantage = bond_income - share_income
    if advantage <= 0:
        return None
    return premium / advantage


@dataclass
class Greeks:
    """Sensitivities per 100 of face.

    delta               value change per 1.00 move in the share price
    gamma               change in delta per 1.00 move in the share price
    vega                value change per 1 percentage point of volatility
    rho                 value change per 1bp move in the risk-free rate
    spread_sensitivity  value change per 1bp move in the credit spread
    """

    delta: float
    gamma: float
    vega: float
    rho: float
    spread_sensitivity: float
    conversion_ratio: float = 0.0

    @property
    def equity_participation(self) -> float:
        """Delta as a fraction of full conversion: 1.0 means it moves like the shares."""
        if self.conversion_ratio <= 0:
            return 0.0
        return self.delta / self.conversion_ratio


def greeks(
    bond: ConvertibleBond,
    spot: float,
    volatility: float,
    risk_free: float,
    dividend_yield: float = 0.0,
    steps: int = 400,
) -> Greeks:
    """Finite-difference sensitivities around the current market state."""

    def price_at(
        s: float = spot,
        vol: float = volatility,
        r: float = risk_free,
        spread: Optional[float] = None,
    ) -> float:
        b = bond
        if spread is not None and spread != bond.credit_spread:
            b = ConvertibleBond(
                face=bond.face,
                coupon_rate=bond.coupon_rate,
                frequency=bond.frequency,
                maturity=bond.maturity,
                conversion_ratio=bond.conversion_ratio,
                redemption=bond.redemption,
                call_windows=list(bond.call_windows),
                put_dates=list(bond.put_dates),
                credit_spread=spread,
            )
        return price_convertible(b, s, vol, r, dividend_yield, steps).price

    base_res = price_convertible(
        bond, spot, volatility, risk_free, dividend_yield, steps
    )
    delta = base_res.delta
    gamma = base_res.gamma

    dv = 0.01
    vega = (price_at(vol=volatility + dv) - price_at(vol=max(0.0, volatility - dv))) / (
        2.0 * dv
    ) / 100.0

    dr = 0.0001
    rho = (price_at(r=risk_free + dr) - price_at(r=risk_free - dr)) / 2.0

    ds = 0.0001
    spread_sens = (
        price_at(spread=bond.credit_spread + ds)
        - price_at(spread=max(0.0, bond.credit_spread - ds))
    ) / 2.0

    return Greeks(
        conversion_ratio=bond.conversion_ratio,
        delta=delta,
        gamma=gamma,
        vega=vega,
        rho=rho,
        spread_sensitivity=spread_sens,
    )
