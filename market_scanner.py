from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import FinanceDataReader as fdr
import pandas as pd

from analyzer import (
    calculate_final_score,
    calculate_indicators,
    calculate_minervini,
    calculate_oneil,
    calculate_relative_strength,
    calculate_trade_plan,
    calculate_vcp,
    calculate_volume_analysis,
    calculate_weinstein,
)
from listing_provider import load_market_snapshot


RECOMMEND_LOOKBACKS = {
    "Daily": 1,
    "Weekly": 5,
    "Monthly": 20,
}

MOVER_LOOKBACKS = {
    "Daily": 1,
    "Monthly": 20,
    "Yearly": 240,
}


def analysis_link(code: str, market: str) -> str:
    return f"?analyze_code={str(code).zfill(6)}&market={market}"


def market_snapshot() -> pd.DataFrame:
    return load_market_snapshot()


def recommendation_top10_by_market(mode: str = "Daily", per_market_universe: int = 30) -> dict[str, pd.DataFrame]:
    listing = market_snapshot()
    start_date = (datetime.today() - timedelta(days=365 * 3 + 45)).strftime("%Y-%m-%d")
    benchmarks = {
        "KOSPI": calculate_indicators(_clean_ohlcv(fdr.DataReader("KS11", start_date))),
        "KOSDAQ": calculate_indicators(_clean_ohlcv(fdr.DataReader("KQ11", start_date))),
    }

    output = {}
    for market in ["KOSPI", "KOSDAQ"]:
        universe = _recommendation_universe(listing, market, per_market_universe)
        rows = _scan_recommendations(universe, benchmarks[market], start_date, mode)
        frame = pd.DataFrame(rows)
        if frame.empty:
            output[market] = frame
            continue
        frame = frame.sort_values(
            ["Candidate score", "Overall score", "3-month excess return", "Volume multiple"],
            ascending=[False, False, False, False],
        ).head(10)
        output[market] = _format_recommendation_frame(frame)
    return output


def movers_top10_by_market(period: str = "Daily", direction: str = "Rising", per_market_universe: int = 60) -> dict[str, pd.DataFrame]:
    bundle = movers_top10_bundle_by_market(period, per_market_universe)
    return {market: bundle[market][direction] for market in ["KOSPI", "KOSDAQ"]}


def movers_top10_bundle_by_market(period: str = "Daily", per_market_universe: int = 60) -> dict[str, dict[str, pd.DataFrame]]:
    listing = market_snapshot()
    output = {}
    for market in ["KOSPI", "KOSDAQ"]:
        market_listing = listing[listing["Market"] == market].copy()
        if period == "Daily":
            frame = market_listing.dropna(subset=["ChagesRatio"]).copy()
            frame["Return"] = frame["ChagesRatio"]
        else:
            universe = _mover_universe(market_listing, per_market_universe)
            frame = pd.DataFrame(_scan_movers(universe, MOVER_LOOKBACKS[period]))

        if frame.empty:
            output[market] = {"Rising": frame, "Falling": frame}
            continue

        gainers = frame.sort_values("Return", ascending=False).head(10)
        losers = frame.sort_values("Return", ascending=True).head(10)
        output[market] = {
            "Rising": _format_mover_frame(gainers),
            "Falling": _format_mover_frame(losers),
        }
    return output


def unusual_risk_top10_by_market(per_market_universe: int = 80) -> dict[str, pd.DataFrame]:
    listing = market_snapshot()
    output = {}
    start_date = (datetime.today() - timedelta(days=210)).strftime("%Y-%m-%d")
    for market in ["KOSPI", "KOSDAQ"]:
        universe = _unusual_universe(listing, market, per_market_universe)
        rows = _scan_unusual(universe, start_date)
        frame = pd.DataFrame(rows)
        if frame.empty:
            output[market] = frame
            continue
        frame = frame.sort_values(["Risk signal score", "Volume multiple", "Monthly return"], ascending=False).head(10)
        output[market] = _format_unusual_frame(frame)
    return output


def historical_success_cases(per_market_universe: int = 35, forward_days: int = 60, limit: int = 10) -> list[dict[str, Any]]:
    listing = market_snapshot()
    start_date = (datetime.today() - timedelta(days=365 * 3 + 90)).strftime("%Y-%m-%d")
    benchmarks = {
        "KOSPI": calculate_indicators(_clean_ohlcv(fdr.DataReader("KS11", start_date))),
        "KOSDAQ": calculate_indicators(_clean_ohlcv(fdr.DataReader("KQ11", start_date))),
    }
    universes = [
        _recommendation_universe(listing, "KOSPI", per_market_universe),
        _recommendation_universe(listing, "KOSDAQ", per_market_universe),
    ]
    universe = pd.concat(universes, ignore_index=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_historical_case_candidate, row.Code, row.Name, row.Market, benchmarks[row.Market], start_date, forward_days): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)

    return sorted(rows, key=lambda item: (item["Peak return"], item["Signal score"]), reverse=True)[:limit]


def _scan_recommendations(universe: pd.DataFrame, benchmark: pd.DataFrame, start_date: str, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_analyze_candidate, row.Code, row.Name, row.Market, benchmark, start_date, mode): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    return rows


def _analyze_candidate(code: str, name: str, market: str, benchmark: pd.DataFrame, start_date: str, mode: str) -> dict[str, Any] | None:
    try:
        stock = calculate_indicators(_clean_ohlcv(fdr.DataReader(code, start_date)))
        common_dates = stock.index.intersection(benchmark.index)
        stock = stock.loc[common_dates].copy()
        aligned_benchmark = benchmark.loc[common_dates].copy()
        if len(stock) < 80:
            return None

        minervini = calculate_minervini(stock)
        weinstein = calculate_weinstein(stock)
        oneil = calculate_oneil(stock)
        rs = calculate_relative_strength(stock, aligned_benchmark)
        vcp = calculate_vcp(stock)
        volume = calculate_volume_analysis(stock)
        final = calculate_final_score(minervini, oneil, weinstein, rs, vcp, stock)
        if final["forced_exclusion"]:
            return None

        trade_plan = calculate_trade_plan(stock, oneil, final)
        latest = stock.iloc[-1]
        lookback = RECOMMEND_LOOKBACKS.get(mode, 1)
        recent_return = _return_over_days(stock, lookback)
        momentum_bonus = _momentum_bonus(recent_return)
        recommendation_score = round(final["total_score"] * 0.9 + momentum_bonus, 1)

        return {
            "Stock": f"{name} ({code})",
            "Analysis link": analysis_link(code, market),
            "Market": market,
            "Last close": float(latest["Close"]),
            "Overall score": final["total_score"],
            "Candidate score": recommendation_score,
            "Period return": recent_return,
            "3-month excess return": rs["values"].get("3 months"),
            "Volume multiple": volume["ratio"],
            "Rating": f"{final['stars']} {final['rating']}",
            "Entry status": final["entry_status"],
            "Entry approach": trade_plan["entry_method"],
            "Entry range": _price_range(trade_plan["buy_low"], trade_plan["buy_high"]),
            "Defensive exit range": _price_range(trade_plan["risk_sell_low"], trade_plan["risk_sell_high"]),
            "Profit-taking range": _price_range(trade_plan["profit_sell_low"], trade_plan["profit_sell_high"]),
        }
    except Exception:
        return None


def _scan_movers(universe: pd.DataFrame, lookback: int) -> list[dict[str, Any]]:
    start_date = (datetime.today() - timedelta(days=max(lookback * 2, 365) + 15)).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_mover_candidate, row.Code, row.Name, row.Market, start_date, lookback): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    return rows


def _mover_candidate(code: str, name: str, market: str, start_date: str, lookback: int) -> dict[str, Any] | None:
    try:
        df = _clean_ohlcv(fdr.DataReader(code, start_date))
        if len(df) <= lookback:
            return None
        current = float(df["Close"].iloc[-1])
        past = float(df["Close"].iloc[-lookback - 1])
        if past <= 0:
            return None
        return {
            "Stock": f"{name} ({code})",
            "Analysis link": analysis_link(code, market),
            "Market": market,
            "Last close": current,
            "Return": (current / past - 1) * 100,
            "Trading value": None,
        }
    except Exception:
        return None


def _scan_unusual(universe: pd.DataFrame, start_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_unusual_candidate, row.Code, row.Name, row.Market, row.Close, row.ChagesRatio, row.Volume, row.Amount, row.Marcap, start_date): row.Code
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    return rows


def _unusual_candidate(
    code: str,
    name: str,
    market: str,
    close: float,
    daily_change: float,
    volume: float,
    amount: float,
    marcap: float,
    start_date: str,
) -> dict[str, Any] | None:
    try:
        df = calculate_indicators(_clean_ohlcv(fdr.DataReader(code, start_date)))
        if len(df) < 40:
            return None
        latest = df.iloc[-1]
        monthly_return = _return_over_days(df, 20)
        weekly_return = _return_over_days(df, 5)
        volume_ratio = float(volume) / float(latest["VOL20"]) if pd.notna(latest.get("VOL20")) and latest["VOL20"] else 0
        ma20_gap = (float(close) / float(latest["MA20"]) - 1) * 100 if pd.notna(latest.get("MA20")) and latest["MA20"] else 0
        intraday_range = (float(latest["High"]) - float(latest["Low"])) / float(latest["Close"]) * 100 if latest["Close"] else 0

        score = 0.0
        reasons = []
        if abs(float(daily_change)) >= 20:
            score += 25
            reasons.append(f"Daily Return {float(daily_change):+.1f}%")
        elif abs(float(daily_change)) >= 10:
            score += 15
            reasons.append(f"Daily Return {float(daily_change):+.1f}%")
        if monthly_return >= 80:
            score += 25
            reasons.append(f"Monthly surge {monthly_return:+.1f}%")
        elif monthly_return >= 40:
            score += 15
            reasons.append(f"Monthly Rising {monthly_return:+.1f}%")
        if volume_ratio >= 8:
            score += 25
            reasons.append(f"Volume {volume_ratio:.1f}x")
        elif volume_ratio >= 4:
            score += 15
            reasons.append(f"Volume {volume_ratio:.1f}x")
        if ma20_gap >= 35:
            score += 15
            reasons.append(f"MA20 extension {ma20_gap:+.1f}%")
        elif ma20_gap >= 20:
            score += 8
            reasons.append(f"MA20 extension {ma20_gap:+.1f}%")
        if intraday_range >= 20:
            score += 10
            reasons.append(f"Intraday range {intraday_range:.1f}%")
        if float(marcap) < 150_000_000_000:
            score += 8
            reasons.append("Small market capitalization")

        if score < 25:
            return None

        return {
            "Stock": f"{name} ({code})",
            "Analysis link": analysis_link(code, market),
            "Market": market,
            "Last close": float(close),
            "Risk signal score": round(min(score, 100), 1),
            "Daily return": float(daily_change),
            "Weekly return": weekly_return,
            "Monthly return": monthly_return,
            "Volume multiple": volume_ratio,
            "Trading value": float(amount),
            "Market cap": float(marcap),
            "Risk signals": ", ".join(reasons[:4]),
        }
    except Exception:
        return None


def _historical_case_candidate(code: str, name: str, market: str, benchmark: pd.DataFrame, start_date: str, forward_days: int) -> dict[str, Any] | None:
    try:
        stock = calculate_indicators(_clean_ohlcv(fdr.DataReader(code, start_date)))
        common_dates = stock.index.intersection(benchmark.index)
        stock = stock.loc[common_dates].copy()
        aligned_benchmark = benchmark.loc[common_dates].copy()
        if len(stock) < 280:
            return None

        best_case = None
        end_limit = len(stock) - forward_days
        start_idx = max(220, len(stock) - 360)
        for idx in range(start_idx, end_limit, 5):
            sample = stock.iloc[:idx].copy()
            sample_benchmark = aligned_benchmark.iloc[:idx].copy()
            if len(sample) < 220:
                continue

            minervini = calculate_minervini(sample)
            weinstein = calculate_weinstein(sample)
            oneil = calculate_oneil(sample)
            rs = calculate_relative_strength(sample, sample_benchmark)
            vcp = calculate_vcp(sample)
            volume = calculate_volume_analysis(sample)
            final = calculate_final_score(minervini, oneil, weinstein, rs, vcp, sample)
            if final["forced_exclusion"] or final["total_score"] < 75:
                continue

            entry_price = float(stock["Close"].iloc[idx - 1])
            future_window = stock.iloc[idx : idx + forward_days]
            if future_window.empty:
                continue
            high_after = float(future_window["High"].max())
            low_after = float(future_window["Low"].min())
            final_after = float(future_window["Close"].iloc[-1])
            max_return = (high_after / entry_price - 1) * 100
            close_return = (final_after / entry_price - 1) * 100
            drawdown = (low_after / entry_price - 1) * 100
            if max_return < 15 or close_return < 5:
                continue

            trade_plan = calculate_trade_plan(sample, oneil, final)
            score = max_return + final["total_score"] * 0.2 - abs(min(drawdown, 0)) * 0.5
            case = {
                "Stock": f"{name} ({code})",
                "Market": market,
                "Signal date": sample.index[-1].strftime("%Y-%m-%d"),
                "Signal price": entry_price,
                "Signal score": final["total_score"],
                "Signal rating": final["rating"],
                "Entry status": final["entry_status"],
                "Entry approach": trade_plan["entry_method"],
                "Entry range": _price_range(trade_plan["buy_low"], trade_plan["buy_high"]),
                "Peak return": round(max_return, 1),
                "Closing return": round(close_return, 1),
                "Lowest price return": round(drawdown, 1),
                "Analysis link": analysis_link(code, market),
                "Case score": score,
                "Key signals": _case_reasons(minervini, oneil, rs, volume, vcp),
            }
            if best_case is None or case["Case score"] > best_case["Case score"]:
                best_case = case

        return best_case
    except Exception:
        return None


def _case_reasons(minervini: dict[str, Any], oneil: dict[str, Any], rs: dict[str, Any], volume: dict[str, Any], vcp: dict[str, Any]) -> str:
    reasons = []
    if minervini["score"] >= 75:
        reasons.append(f"Minervini {minervini['score']:.1f} points")
    if oneil["breakout"]:
        reasons.append("60-session high breakout")
    elif oneil["near_high"]:
        reasons.append("60-session high distance under 5% within")
    rs_3m = rs["values"].get("3 months")
    if rs_3m is not None and rs_3m > 0:
        reasons.append(f"3 months RS {rs_3m:+.1f}%p")
    if volume["ratio"] >= 1.2:
        reasons.append(f"Volume {volume['ratio']:.1f}x")
    if vcp["score"] >= 50:
        reasons.append(vcp["status"])
    return ", ".join(reasons[:5])


def _recommendation_universe(listing: pd.DataFrame, market: str, limit: int) -> pd.DataFrame:
    market_listing = listing[
        (listing["Market"] == market)
        & (listing["Marcap"] >= 100_000_000_000)
        & (listing["Amount"] >= 3_000_000_000)
    ].copy()
    return market_listing.sort_values(["Amount", "Marcap"], ascending=False).head(limit)


def _mover_universe(listing: pd.DataFrame, limit: int) -> pd.DataFrame:
    return listing[
        (listing["Marcap"] >= 30_000_000_000)
        & (listing["Amount"] >= 500_000_000)
    ].sort_values(["Amount", "Marcap"], ascending=False).head(limit)


def _unusual_universe(listing: pd.DataFrame, market: str, limit: int) -> pd.DataFrame:
    market_listing = listing[
        (listing["Market"] == market)
        & (listing["Marcap"] >= 20_000_000_000)
        & (listing["Amount"] >= 300_000_000)
    ].copy()
    market_listing["AbsChange"] = market_listing["ChagesRatio"].abs()
    return market_listing.sort_values(["AbsChange", "Amount"], ascending=False).head(limit)


def _format_recommendation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in ["Last close"]:
        result[col] = result[col].map(lambda v: f"{v:,.0f} KRW")
    for col in ["Overall score", "Candidate score", "Period return", "3-month excess return", "Volume multiple"]:
        result[col] = result[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):,.1f}")
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result[
        [
            "Rank",
            "Stock",
            "Analysis link",
            "Candidate score",
            "Overall score",
            "Period return",
            "3-month excess return",
            "Volume multiple",
            "Rating",
            "Entry status",
            "Entry approach",
            "Entry range",
            "Defensive exit range",
            "Profit-taking range",
        ]
    ]


def _format_mover_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "Stock" not in result.columns and {"Name", "Code"}.issubset(result.columns):
        result["Stock"] = result["Name"].astype(str) + " (" + result["Code"].astype(str).str.zfill(6) + ")"
    if "Analysis link" not in result.columns and {"Code", "Market"}.issubset(result.columns):
        result["Analysis link"] = [analysis_link(code, market) for code, market in zip(result["Code"], result["Market"])]
    if "Last close" not in result.columns and "Close" in result.columns:
        result["Last close"] = pd.to_numeric(result["Close"], errors="coerce")
    result["Last close"] = result["Last close"].map(lambda v: f"{v:,.0f} KRW")
    result["Return"] = result["Return"].map(lambda v: f"{float(v):+.1f}%")
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result[["Rank", "Stock", "Analysis link", "Last close", "Return"]]


def _format_unusual_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Last close"] = result["Last close"].map(lambda v: f"{v:,.0f} KRW")
    result["Risk signal score"] = result["Risk signal score"].map(lambda v: f"{float(v):.1f}")
    for col in ["Daily return", "Weekly return", "Monthly return"]:
        result[col] = result[col].map(lambda v: f"{float(v):+.1f}%")
    result["Volume multiple"] = result["Volume multiple"].map(lambda v: f"{float(v):.1f}x")
    result["Trading value"] = result["Trading value"].map(lambda v: f"{float(v) / 100_000_000:,.0f} hundred million KRW")
    result["Market cap"] = result["Market cap"].map(lambda v: f"{float(v) / 100_000_000:,.0f} hundred million KRW")
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result[
        [
            "Rank",
            "Stock",
            "Analysis link",
            "Risk signal score",
            "Last close",
            "Daily return",
            "Weekly return",
            "Monthly return",
            "Volume multiple",
            "Trading value",
            "Market cap",
            "Risk signals",
        ]
    ]


def _return_over_days(df: pd.DataFrame, days: int) -> float:
    if len(df) <= days:
        return 0.0
    return (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-days - 1]) - 1) * 100


def _momentum_bonus(value: float) -> float:
    if value >= 20:
        return 10.0
    if value >= 10:
        return 7.5
    if value >= 5:
        return 5.0
    if value > 0:
        return 2.5
    return 0.0


def _price_range(low: float | None, high: float | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "Not provided"
    return f"{low:,.0f}~{high:,.0f} KRW"


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    result.index = pd.to_datetime(result.index)
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.dropna(subset=["Close"]).sort_index()
