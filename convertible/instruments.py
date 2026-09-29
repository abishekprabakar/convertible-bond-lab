"""Instrument definitions for convertible bonds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class CallWindow:
    """Issuer call right, live over [start, end] years at `price` per 100 face.

    `trigger` is an optional soft-call provision: the issuer may only call when
    the share price is at least `trigger` (a multiple of the conversion price).
    """

    start: float
    end: float
    price: float
    trigger: Optional[float] = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("call window end must be >= start")
        if self.price <= 0:
            raise ValueError("call price must be positive")

    def live_at(self, t: float, eps: float = 1e-9) -> bool:
        return self.start - eps <= t <= self.end + eps


@dataclass(frozen=True)
class PutDate:
    """Holder put right on a single date, at `price` per 100 face."""

    time: float
    price: float

    def __post_init__(self) -> None:
        if self.time <= 0:
            raise ValueError("put time must be positive")
        if self.price <= 0:
            raise ValueError("put price must be positive")


@dataclass
class ConvertibleBond:
    """A convertible bond quoted per 100 of face value.

    face               : redemption amount (100 = par quoting convention)
    coupon_rate        : annual coupon as a decimal of face (0.02 = 2%)
    frequency          : coupon payments per year (0 for a zero-coupon issue)
    maturity           : years to maturity
    conversion_ratio   : shares received per bond of `face`
    redemption         : redemption price at maturity (defaults to face)
    call_windows       : issuer call rights
    put_dates          : holder put rights
    credit_spread      : issuer spread over the risk-free rate, continuous, decimal
    """

    face: float = 100.0
    coupon_rate: float = 0.0
    frequency: int = 2
    maturity: float = 5.0
    conversion_ratio: float = 0.0
    redemption: Optional[float] = None
    call_windows: List[CallWindow] = field(default_factory=list)
    put_dates: List[PutDate] = field(default_factory=list)
    credit_spread: float = 0.0

    def __post_init__(self) -> None:
        if self.face <= 0:
            raise ValueError("face must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.frequency < 0:
            raise ValueError("frequency cannot be negative")
        if self.coupon_rate and not self.frequency:
            raise ValueError("a coupon-paying bond needs a positive frequency")
        if self.conversion_ratio < 0:
            raise ValueError("conversion ratio cannot be negative")
        if self.credit_spread < 0:
            raise ValueError("credit spread cannot be negative")
        if self.redemption is None:
            self.redemption = self.face
        for w in self.call_windows:
            if w.start < 0 or w.end > self.maturity + 1e-9:
                raise ValueError("call window must lie within the bond's life")
        for p in self.put_dates:
            if p.time > self.maturity + 1e-9:
                raise ValueError("put date must lie within the bond's life")

    @property
    def coupon_amount(self) -> float:
        """Cash paid per coupon period."""
        if not self.frequency:
            return 0.0
        return self.face * self.coupon_rate / self.frequency

    @property
    def conversion_price(self) -> float:
        """Face value surrendered per share received."""
        if self.conversion_ratio <= 0:
            return float("inf")
        return self.face / self.conversion_ratio

    def coupon_times(self) -> List[float]:
        """Coupon dates in years, ascending, stepping back from maturity."""
        if not self.frequency or not self.coupon_rate:
            return []
        step = 1.0 / self.frequency
        times: List[float] = []
        t = self.maturity
        while t > 1e-9:
            times.append(round(t, 10))
            t -= step
        return sorted(times)

    def cashflows(self) -> List[Tuple[float, float]]:
        """Straight-bond cashflows as (time, amount), redemption included."""
        flows = [(t, self.coupon_amount) for t in self.coupon_times()]
        merged: List[Tuple[float, float]] = []
        for t, amt in flows:
            if abs(t - self.maturity) < 1e-9:
                continue
            merged.append((t, amt))
        final = self.redemption + (self.coupon_amount if self.coupon_times() else 0.0)
        merged.append((self.maturity, final))
        return merged

    def call_price_at(self, t: float) -> Optional[CallWindow]:
        """The live call window at time `t`, cheapest first, or None."""
        live = [w for w in self.call_windows if w.live_at(t)]
        if not live:
            return None
        return min(live, key=lambda w: w.price)
