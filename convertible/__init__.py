"""convertible-bond-lab: convertible bond pricing and analytics.

Prices convertible bonds with the Tsiveriotis-Fernandes (1998) binomial model,
which splits the instrument into an equity-linked component discounted at the
risk-free rate and a cash-only component discounted at the credit-risky rate.
"""

from .instruments import CallWindow, ConvertibleBond, PutDate
from .tree import PricingResult, price_convertible, price_refined
from .analytics import (
    breakeven_years,
    conversion_premium,
    greeks,
    investment_value,
    parity,
    straight_bond_ytm,
)

__all__ = [
    "CallWindow",
    "ConvertibleBond",
    "PutDate",
    "PricingResult",
    "price_convertible",
    "price_refined",
    "parity",
    "conversion_premium",
    "investment_value",
    "straight_bond_ytm",
    "breakeven_years",
    "greeks",
]
__version__ = "1.0.0"
