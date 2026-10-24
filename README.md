# Korea Stock Breakout Lab English Edition

An English-language Streamlit application for researching Korean equities on
KOSPI and KOSDAQ. This edition preserves the technical calculations and eight
feature categories of the Korean application. It does not analyze US stocks.
Prices, capital, and reference ranges are in Korean won (KRW).

## Features

- Stock Analysis: moving averages, price/volume breakout checks, relative
  strength, Weinstein stages, VCP-like contraction, and analytical commentary.
- Compare Stocks: compare up to ten stocks in the selected market.
- Top 10 Candidates: daily, weekly, and monthly momentum-assisted rankings.
- Top Gainers and Losers: daily, 20-session, and 240-session return rankings.
- Unusual Activity: technical anomalies, not a manipulation detector.
- Investment Planner: horizon, style, capital, staged entries and exits.
- Historical Case Studies: retrospectively selected winning examples.
- Investor Board: session-only discussion demonstration.

## Run locally

Use Python 3.12 or a compatible newer version.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Search using a six-digit stock code, a Korean company name, or a supported
English alias such as Samsung Electronics, SK hynix, NAVER, Hyundai Motor,
Kia, Celltrion, or Ecopro BM. English aliases and general business summaries
are curated for selected companies in company_profiles.py. Other companies
retain their provider-supplied Korean names; every stock supports code entry.

## Data and methodology

FinanceDataReader supplies historical data and listings. A Naver listing
fallback is used when the primary listing is unavailable. The fallback
currently covers up to 12 market-cap pages per market; this is not a full
exchange-universe screen. Listed trading value may be estimated as close
times volume in the fallback.

Scores use Minervini 35%, O'Neil-style price/volume 25%, Weinstein 15%, market
relative strength 15%, and VCP-like contraction 10%. These are simplified
implementations of public technical principles, not the authors' complete
investment methodologies.

Quick candidate scans cover up to 30 stocks per market; Broad scans cover
up to 80. Monthly/yearly movers cover up to 60/160 candidates. Rankings are
within filtered universes, not guaranteed exchange-wide leaders. Refreshes
are button-driven with caching, not an unattended daily scheduler.

The planner preserves the original rule-based sizing and reference prices.
It is not a price-path simulator, optimizer, or a guarantee of maximum loss.
The loss parameter can differ from the actual defense distance and risk
budget. Fees, taxes, slippage, and real execution are not modeled. Even an
entry-on-hold setup can display hypothetical quantities; the hold takes
precedence over the reference plan.

Historical cases select winners after the fact and use signal-day close as
the return reference, not a simulated fill. Peak returns are not realized
profits. No aggregate hit rate or survivorship-free backtest is claimed.

Board posts and interest registrations exist only in the current session.
They do not create accounts, send email, or share posts across visitors.
External data may be delayed or unavailable, including on cloud servers.

## Deployment

On Streamlit Community Cloud select this repository, branch main, entrypoint
app.py, and Python 3.12. Choose a public app audience for link sharing.
No API keys or brokerage credentials are required.

## Verification

```sh
python -m py_compile app.py analyzer.py market_scanner.py listing_provider.py company_profiles.py
python -m unittest discover -s tests -v
```

This application is a research tool, not automated trading or a guarantee of
investment performance.
