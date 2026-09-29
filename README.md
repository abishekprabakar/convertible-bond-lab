# convertible-bond-lab

Convertible bond pricing with the **Tsiveriotis–Fernandes** binomial model, plus
the analytics a convert desk actually quotes: parity, investment value,
conversion and risk premium, breakeven, and tree-consistent greeks.

A convertible is a bond you can swap for shares, so it is exposed to issuer
credit on one leg and to equity on the other — and not to both at once. That
asymmetry is the whole modelling problem. Discount every cashflow at the
credit-risky rate and you underprice a convert that is destined to convert into
shares the issuer cannot default on. Discount everything at the risk-free rate
and you overprice a busted convert that is really just a risky bond.

Tsiveriotis–Fernandes (1998) fixes this by carrying **two** values back through
the tree:

| stream | what it is | discounted at |
|---|---|---|
| `B` | the cash-only part: coupons and redemption the issuer must pay in cash | `r + s` |
| `U − B` | the equity-linked part: value the holder gets as shares | `r` |

`U` is the convertible. At every node the model applies, in order, the coupon,
the issuer's call, the holder's put, and the holder's conversion right — and
resets `B` to zero wherever the holder ends up with shares instead of a promise.

No dependencies beyond the Python standard library.

## Install and run

```bash
git clone https://github.com/abishekprabakar/convertible-bond-lab.git
cd convertible-bond-lab
python3 -m convertible.cli --help
```

Requires Python 3.9+. Tests:

```bash
python3 -m unittest discover -s tests -v
```

## Sample output

A 5-year 2% convert struck at parity, 32% share vol, 250bp issuer spread,
soft-callable from year 3 at 100 once the shares trade 1.3× the conversion
price, and putable at 100 in year 3. Pasted from an actual run:

```
$ python3 -m convertible.cli --spot 25 --vol 0.32 --rate 0.042 --dividend 0.005 \
    --coupon 0.02 --maturity 5 --ratio 4 --spread 0.025 \
    --call 3:5:100:1.3 --put 3:100 --grid 10,15,20,25,30,40,60

====================================================================
CONVERTIBLE BOND  (values per 100 of face)
====================================================================
  2.00% coupon, 5y, ratio 4 shares/bond
  conversion price 25.00   share 25.00   vol 32.0%
  risk-free 4.20%   credit spread 250bp   dividend 0.50%
  callable 3y-5y at 100.00, trigger 1.3x conv px
  putable at 3y at 100.00
--------------------------------------------------------------------
  Fair value (extrapolated)   118.1632
    cash-only component        44.3863
    equity component           73.7768
  Parity (conversion value)   100.0000
  Investment value (floor)     79.8896
--------------------------------------------------------------------
  Conversion premium            18.16%
  Risk premium over floor       47.91%
  Breakeven                  12.11 yrs
  Straight-bond YTM at floor    6.700%
--------------------------------------------------------------------
  Delta (shares per bond)       2.8667
  Equity participation          71.67%
  Gamma                         0.0849
  Vega  (per 1 vol point)       0.6492
  Rho   (per 1bp)              -0.0152
  Spread dv (per 1bp)          -0.0150
====================================================================
     share      price    parity   premium      cash    delta
--------------------------------------------------------------------
     10.00     89.063     40.00   122.66%    81.722    0.726
     15.00     94.990     60.00    58.32%    70.040    1.617
     20.00    105.010     80.00    31.26%    57.154    2.348
     25.00    118.163    100.00    18.16%    44.386    2.867
     30.00    133.092    120.00    10.91%    34.798    3.214
     40.00    167.368    160.00     4.60%    20.612    3.602
     60.00    243.036    240.00     1.26%     9.272    3.870
====================================================================
```

The grid is the convert's whole life story in seven rows. At 10 the bond behaves
like credit — 82 of its 89 points are the cash leg, delta is 0.7 shares, and the
premium over parity is meaningless. At 60 it is a stock substitute: the cash leg
has decayed to 9 points, delta is 3.87 of a possible 4.0, and it trades 1.26%
over the shares. The interesting region is 20–30, where both legs matter and the
premium is paid for real optionality.

## Library use

```python
from convertible import CallWindow, ConvertibleBond, PutDate, price_refined, greeks

bond = ConvertibleBond(
    coupon_rate=0.02,
    frequency=2,
    maturity=5.0,
    conversion_ratio=4.0,        # shares per 100 of face
    credit_spread=0.025,         # continuous, over the risk-free rate
    call_windows=[CallWindow(3.0, 5.0, 100.0, trigger=1.3)],
    put_dates=[PutDate(3.0, 100.0)],
)

res = price_refined(bond, spot=25.0, volatility=0.32, risk_free=0.042,
                    dividend_yield=0.005, steps=400)
print(res.price, res.cash_component, res.parity)

g = greeks(bond, 25.0, 0.32, 0.042, 0.005, steps=400)
print(g.delta, g.equity_participation, g.spread_sensitivity)
```

## What each module does

| file | contents |
|---|---|
| `convertible/instruments.py` | `ConvertibleBond`, `CallWindow`, `PutDate`; coupon schedule, cashflows, conversion price, input validation |
| `convertible/tree.py` | the CRR lattice and the Tsiveriotis–Fernandes backward induction; `price_convertible` and the extrapolated `price_refined` |
| `convertible/analytics.py` | investment value, premiums, continuous-yield solver, breakeven, greeks |
| `convertible/cli.py` | argument parsing and the report above |
| `tests/test_convertible.py` | 37 tests |

## Two modelling notes worth reading

**Convergence is `N^-0.5`, not `N^-1`.** The cash component `B` has a
*discontinuous* terminal payoff — the full redemption if the holder does not
convert, zero if they do — and a binomial tree converges at only `N^-0.5`
against a discontinuity. Measured on the bond above:

```
N=  100  117.705816
N=  200  117.819740  diff +0.113924
N=  400  117.899940  diff +0.080200
N=  800  117.956168  diff +0.056228
N= 1600  117.993859  diff +0.037691
N= 3200  118.020389  diff +0.026530
```

Each doubling removes only ~30% of the remaining error, and `N = 400` is still
0.18 short of the limit. So `price_refined` prices at `N` and `2N` and
extrapolates on that known rate:

```
V_inf = (sqrt(2) * V_2N - V_N) / (sqrt(2) - 1)
```

which moves ~0.01 as the base `N` goes from 100 to 1600, against ~0.46 for the
raw tree. `--raw` reports the unextrapolated number if you want to see it.

**Greeks come off the tree, not from re-pricing.** Bumping the spot and
re-pricing shifts the entire lattice, so a finite-difference delta on this model
jumps around by tenths of a share and is not even monotone in the spot. Delta
and gamma are read off the tree's own step-1 and step-2 nodes instead, which is
both stable and free. Vega, rho and spread sensitivity stay finite-difference,
since those bumps leave the lattice in place.

## Tests

37 tests in `tests/test_convertible.py`, stdlib `unittest` only. The ones that
earn their keep are the limiting cases where the answer is known in closed form:

- with the conversion right removed, the tree reproduces the discounted
  cashflows **to eight decimal places**, at both zero and non-zero spread
- a put dated on the maturity date reduces to a higher redemption, matched
  analytically (this test caught a real bug — the backward induction never
  visits the terminal layer, so maturity-dated puts were silently ignored)
- deep in the money with no coupon and no dividend, the price collapses onto
  parity and the cash component onto zero
- extrapolation leaves a straight bond exactly alone

plus monotonicity in spot, vol and spread; the cash leg decaying as equity takes
over; a call capping the price and a soft-call trigger being worth more to the
holder than a hard call; delta bounded by the conversion ratio and rising with
the spot; and a yield-solver round trip.

```
$ python3 -m unittest discover -s tests
.....................................
----------------------------------------------------------------------
Ran 37 tests in 5.884s

OK
```

## Scope

Single-factor share price, deterministic rates, and a constant credit spread —
the standard TF setup. No stochastic rates, no equity-dependent hazard rate, no
dilution or make-whole adjustment, and clean/dirty price conventions are left
out (everything is a dirty price per 100 of face). Accrued interest and a real
day-count would be the first things to add for anything quoted off a screen.

## Reference

Tsiveriotis, K. and Fernandes, C. (1998), "Valuing Convertible Bonds with Credit
Risk", *Journal of Fixed Income* 8(2), 95–102.

## License

MIT — see [LICENSE](LICENSE).
