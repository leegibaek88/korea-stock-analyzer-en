from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

from listing_provider import load_listing


PERIOD_DAYS = {
    "1 year": 365,
    "2 years": 365 * 2,
    "3 years": 365 * 3,
    "5 years": 365 * 5,
}


def load_stock_listing(market: str) -> pd.DataFrame:
    listing = load_listing(market)
    if listing is None or listing.empty:
        raise ValueError("Unable to load the stock listing.")
    return listing.drop_duplicates(subset=["Code"])


def resolve_stock_query(query: str, market: str, listing: pd.DataFrame | None = None) -> dict[str, str]:
    text = str(query).strip()
    if not text:
        raise ValueError("Enter a stock code or company name.")

    if text.isdigit():
        code = normalize_code(text)
        name = ""
        if listing is not None and not listing.empty:
            matched = listing[listing["Code"] == code]
            if not matched.empty:
                name = str(matched.iloc[0]["Name"])
        return {"code": code, "name": name or code}

    listing = listing if listing is not None else load_stock_listing(market)
    normalized = text.casefold().replace(' ', '')
    match_name = listing["Name"].str.casefold().str.replace(' ', '', regex=False)
    native = listing.get('NativeName', listing['Name']).str.casefold()
    exact = listing[(match_name == normalized) | (native == text.casefold())]
    if not exact.empty:
        row = exact.iloc[0]
        return {"code": str(row["Code"]), "name": str(row["Name"])}

    contains = listing[match_name.str.contains(normalized, na=False, regex=False) | native.str.contains(text.casefold(), na=False, regex=False)]
    if contains.empty:
        raise ValueError("Company not found. Check the code, name, or selected market.")
    if len(contains) > 1:
        examples = ", ".join(f"{row.Name}({row.Code})" for row in contains.head(8).itertuples())
        raise ValueError(f"Multiple matches found. Please use a more specific name: {examples}")

    row = contains.iloc[0]
    return {"code": str(row["Code"]), "name": str(row["Name"])}


def load_stock_data(code: str, market: str, period: str = "3 years") -> tuple[pd.DataFrame, pd.DataFrame]:
    code = normalize_code(code)
    days = PERIOD_DAYS.get(period, PERIOD_DAYS["3 years"])
    start_date = (datetime.today() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
    benchmark_code = "KS11" if market == "KOSPI" else "KQ11"

    stock = fdr.DataReader(code, start_date)
    benchmark = fdr.DataReader(benchmark_code, start_date)

    if stock is None or stock.empty:
        raise ValueError("Check the stock code or data provider availability.")
    if benchmark is None or benchmark.empty:
        raise ValueError("Benchmark data is unavailable. Please try again later.")

    stock = _clean_ohlcv(stock)
    benchmark = _clean_ohlcv(benchmark)
    common_dates = stock.index.intersection(benchmark.index)
    stock = stock.loc[common_dates].copy()
    benchmark = benchmark.loc[common_dates].copy()

    if stock.empty:
        raise ValueError("Check the stock code or data provider availability.")

    return stock, benchmark


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["MA20"] = result["Close"].rolling(20).mean()
    result["MA50"] = result["Close"].rolling(50).mean()
    result["MA150"] = result["Close"].rolling(150).mean()
    result["MA200"] = result["Close"].rolling(200).mean()
    result["VOL20"] = result["Volume"].rolling(20).mean()
    result["VOL50"] = result["Volume"].rolling(50).mean()
    return result


def calculate_minervini(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    high_52w = float(df["Close"].tail(min(252, len(df))).max())
    low_52w = float(df["Close"].tail(min(252, len(df))).min())
    ma200_20d_ago = _value_days_ago(df["MA200"], 20)

    checks = [
        _check("Last close > MA150", current, "MA150", latest.get("MA150"), current > latest.get("MA150", np.nan)),
        _check("Last close > MA200", current, "MA200", latest.get("MA200"), current > latest.get("MA200", np.nan)),
        _check("MA150 > MA200", latest.get("MA150"), "MA200", latest.get("MA200"), latest.get("MA150", np.nan) > latest.get("MA200", np.nan)),
        _check("MA200 Rising", latest.get("MA200"), "20sessions ago MA200", ma200_20d_ago, latest.get("MA200", np.nan) > ma200_20d_ago),
        _check("MA50 > MA150", latest.get("MA50"), "MA150", latest.get("MA150"), latest.get("MA50", np.nan) > latest.get("MA150", np.nan)),
        _check("MA50 > MA200", latest.get("MA50"), "MA200", latest.get("MA200"), latest.get("MA50", np.nan) > latest.get("MA200", np.nan)),
        _check("At least 30% above the 52-week closing low", current, f"{low_52w * 1.30:,.0f}", low_52w * 1.30, current >= low_52w * 1.30),
        _check("At least 75% of the 52-week closing high", current, f"{high_52w * 0.75:,.0f}", high_52w * 0.75, current >= high_52w * 0.75),
    ]
    valid = [item for item in checks if item["result"] != "Insufficient data"]
    passed = sum(1 for item in valid if item["result"] == "PASS")
    score = passed / len(valid) * 100 if valid else 0
    return {"checks": checks, "score": round(score, 1), "passed": passed, "valid": len(valid), "high_52w": high_52w, "low_52w": low_52w}


def calculate_weinstein(df: pd.DataFrame) -> dict[str, Any]:
    weekly = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    weekly["WMA30"] = weekly["Close"].rolling(30).mean()
    latest = weekly.iloc[-1]
    current = float(latest["Close"])
    wma30 = latest.get("WMA30", np.nan)
    wma30_4w_ago = _value_days_ago(weekly["WMA30"], 4)
    high_20w = float(weekly["Close"].tail(min(20, len(weekly))).max())

    if pd.isna(wma30) or pd.isna(wma30_4w_ago):
        stage = "Stage 1/3 sideways"
        score = 50
        reason = "Insufficient 30-week history; provisionally classified as a sideways phase."
    elif current > wma30 and wma30 > wma30_4w_ago and current >= high_20w * 0.90:
        stage = "Stage 2 uptrend"
        score = 100
        reason = "Price is above a rising 30-week moving average."
    elif current < wma30 and wma30 < wma30_4w_ago:
        stage = "Stage 4 downtrend"
        score = 0
        reason = "Price is below a falling 30-week moving average."
    else:
        stage = "Stage 1/3 sideways"
        score = 50
        reason = "No clear rising or falling trend is confirmed."

    return {"stage": stage, "score": score, "weekly": weekly, "wma30": _float_or_none(wma30), "reason": reason}


def calculate_oneil(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    high_60d = float(df["High"].tail(min(60, len(df))).max())
    previous_60d_high = df["High"].shift(1).rolling(60).max().iloc[-1]
    distance_to_60d_high = (high_60d - current) / high_60d * 100 if high_60d else np.nan
    near_high = current >= high_60d * 0.95 if high_60d else False
    breakout = bool(pd.notna(previous_60d_high) and current > previous_60d_high)
    volume_ratio = _safe_ratio(latest["Volume"], latest.get("VOL20"))

    checks = {
        "Last close > MA50": bool(pd.notna(latest.get("MA50")) and current > latest["MA50"]),
        "Last close > MA200": bool(pd.notna(latest.get("MA200")) and current > latest["MA200"]),
        "MA50 > MA200": bool(pd.notna(latest.get("MA50")) and pd.notna(latest.get("MA200")) and latest["MA50"] > latest["MA200"]),
        "Within 5% of the 60-session high": bool(near_high),
        "60-session breakout": breakout,
    }

    score = 0
    score += 15 if checks["Last close > MA50"] else 0
    score += 15 if checks["Last close > MA200"] else 0
    score += 15 if checks["MA50 > MA200"] else 0
    score += 15 if near_high else 0
    score += 20 if breakout else 0
    score += 20 if volume_ratio >= 1.5 else 10 if volume_ratio >= 1.2 else 0

    if breakout:
        breakout_status = "New-high breakout"
    elif near_high:
        breakout_status = "Near breakout zone"
    else:
        breakout_status = "Outside breakout zone"

    return {
        "score": min(score, 100),
        "checks": checks,
        "high_60d": high_60d,
        "previous_60d_high": _float_or_none(previous_60d_high),
        "distance_to_60d_high": _float_or_none(distance_to_60d_high),
        "near_high": near_high,
        "breakout": breakout,
        "volume_ratio": volume_ratio,
        "volume_status": classify_breakout_volume(volume_ratio),
        "breakout_status": breakout_status,
    }


def calculate_relative_strength(stock: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    periods = {"1 month": 20, "3 months": 60, "6 months": 120, "12 months": 240}
    current_stock = float(stock["Close"].iloc[-1])
    current_market = float(benchmark["Close"].iloc[-1])
    values: dict[str, float | None] = {}
    score = 0

    for label, days in periods.items():
        if len(stock) <= days or len(benchmark) <= days:
            values[label] = None
            continue
        stock_return = current_stock / float(stock["Close"].iloc[-days - 1]) - 1
        market_return = current_market / float(benchmark["Close"].iloc[-days - 1]) - 1
        rs = (stock_return - market_return) * 100
        values[label] = round(rs, 1)
        score += 25 if rs > 0 else 0

    return {"values": values, "score": score}


def calculate_vcp(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 60:
        return {"status": "Insufficient data", "score": 0, "volatilities": {}, "volumes": {}, "volatility_pass": False, "volume_pass": False}

    recent = df.tail(60)
    segments = {
        "A": recent.iloc[0:20],
        "B": recent.iloc[20:40],
        "C": recent.iloc[40:60],
    }
    volatilities = {k: _segment_volatility(v) for k, v in segments.items()}
    volumes = {k: float(v["Volume"].mean()) for k, v in segments.items()}
    volatility_pass = volatilities["A"] > volatilities["B"] > volatilities["C"]
    volume_pass = volumes["C"] < volumes["A"]

    if volatility_pass and volume_pass:
        status = "VCP like contraction detected"
        score = 100
    elif volatility_pass or volume_pass:
        status = "Partial contraction"
        score = 50
    else:
        status = "VCP conditions not met"
        score = 0

    return {
        "status": status,
        "score": score,
        "volatilities": {k: round(v, 1) for k, v in volatilities.items()},
        "volumes": {k: round(v, 0) for k, v in volumes.items()},
        "volatility_pass": volatility_pass,
        "volume_pass": volume_pass,
    }


def calculate_volume_analysis(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    ratio = _safe_ratio(latest["Volume"], latest.get("VOL20"))
    if ratio >= 2.0:
        status = "Volume surge"
    elif ratio >= 1.5:
        status = "Strong volume"
    elif ratio >= 1.2:
        status = "Above average"
    elif ratio >= 0.8:
        status = "Normal"
    else:
        status = "Declining volume"
    return {
        "today": int(latest["Volume"]),
        "vol20": _float_or_none(latest.get("VOL20")),
        "vol50": _float_or_none(latest.get("VOL50")),
        "ratio": ratio,
        "status": status,
    }


def calculate_final_score(
    minervini: dict[str, Any],
    oneil: dict[str, Any],
    weinstein: dict[str, Any],
    rs: dict[str, Any],
    vcp: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    ma50 = latest.get("MA50", np.nan)
    ma200 = latest.get("MA200", np.nan)

    total_score = (
        minervini["score"] * 0.35
        + oneil["score"] * 0.25
        + weinstein["score"] * 0.15
        + rs["score"] * 0.15
        + vcp["score"] * 0.10
    )
    total_score = round(total_score, 1)

    forced_exclusion = (
        (pd.notna(ma200) and current < ma200)
        or (pd.notna(ma50) and pd.notna(ma200) and ma50 < ma200)
        or weinstein["stage"] == "Stage 4 downtrend"
    )
    if forced_exclusion:
        rating = "Excluded / downtrend"
        stars = "★"
    elif total_score >= 85:
        rating = "Priority watchlist"
        stars = "★★★★★"
    elif total_score >= 75:
        rating = "Technical candidate"
        stars = "★★★★"
    elif total_score >= 65:
        rating = "Watchlist"
        stars = "★★★"
    elif total_score >= 50:
        rating = "Watch"
        stars = "★★"
    else:
        rating = "Excluded"
        stars = "★"

    ma20_gap = _gap(current, latest.get("MA20"))
    ma50_gap = _gap(current, ma50)
    entry_status = calculate_entry_status(oneil, total_score, current, ma50, ma20_gap, ma50_gap)
    good_stock = "YES" if not forced_exclusion and total_score >= 75 else "WATCH" if total_score >= 65 else "NO"
    good_price = "BUY ZONE" if entry_status == "Confirmed breakout zone" else "WAIT" if entry_status in {"Awaiting breakout", "Watch"} else "CAUTION"

    return {
        "total_score": total_score,
        "rating": rating,
        "stars": stars,
        "forced_exclusion": forced_exclusion,
        "entry_status": entry_status,
        "good_stock": good_stock,
        "good_price": good_price,
        "ma20_gap": _float_or_none(ma20_gap),
        "ma50_gap": _float_or_none(ma50_gap),
    }


def calculate_trade_plan(df: pd.DataFrame, oneil: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    latest = df.iloc[-1]
    current = float(latest["Close"])
    ma20 = _float_or_none(latest.get("MA20"))
    ma50 = _float_or_none(latest.get("MA50"))
    low_20d = float(df["Low"].tail(min(20, len(df))).min())
    pivot = oneil.get("previous_60d_high") or oneil.get("high_60d")

    if final["forced_exclusion"]:
        method = "Entry on hold"
        buy_low = buy_high = None
        note = "Key trend conditions are weak. Wait for confirmation before considering a new position."
    elif final["entry_status"] == "Confirmed breakout zone" and pivot:
        method = "Staged entry after breakout confirmation"
        buy_low = max(float(pivot), current * 0.98)
        buy_high = min(max(current, float(pivot) * 1.03), current * 1.04)
        note = "If price and volume confirm the breakout, review entries near the pivot without chasing."
    elif final["entry_status"] == "Awaiting breakout" and pivot:
        method = "Wait for a 60-session high breakout"
        buy_low = float(pivot)
        buy_high = float(pivot) * 1.03
        note = "Price is near its high. Wait for a pivot breakout with volume confirmation."
    elif final["entry_status"] == "Extended / avoid chasing":
        method = "Wait for a pullback"
        anchor = ma20 or ma50 or current
        buy_low = anchor * 0.98
        buy_high = anchor * 1.03
        note = "Price is extended. Wait for stabilization near a moving average."
    elif ma20 and current >= ma20:
        method = "MA20 review staged entries nearby"
        buy_low = ma20 * 0.98
        buy_high = ma20 * 1.03
        note = "Review pullbacks near the short-term moving average while the trend remains intact."
    elif ma50:
        method = "MA50 Entry after recovery confirmation"
        buy_low = ma50
        buy_high = ma50 * 1.03
        note = "Below MA50, a new entry requires caution until a recovery is confirmed."
    else:
        method = "Watch"
        buy_low = buy_high = None
        note = "There is not enough moving-average history to provide an entry range."

    defense_candidates = [low_20d]
    if ma20:
        defense_candidates.append(ma20)
    if ma50:
        defense_candidates.append(ma50)
    defense_level = min(defense_candidates)
    risk_sell_low = defense_level * 0.97
    risk_sell_high = defense_level

    profit_base = buy_high or current
    profit_sell_low = profit_base * 1.15
    profit_sell_high = profit_base * 1.25

    return {
        "entry_method": method,
        "buy_low": _float_or_none(buy_low),
        "buy_high": _float_or_none(buy_high),
        "risk_sell_low": _float_or_none(risk_sell_low),
        "risk_sell_high": _float_or_none(risk_sell_high),
        "profit_sell_low": _float_or_none(profit_sell_low),
        "profit_sell_high": _float_or_none(profit_sell_high),
        "note": note,
    }


def calculate_entry_status(oneil: dict[str, Any], total_score: float, current: float, ma50: float, ma20_gap: float | None, ma50_gap: float | None) -> str:
    overheated = (ma20_gap is not None and ma20_gap > 15) or (ma50_gap is not None and ma50_gap > 25)
    if oneil["breakout"] and oneil["volume_ratio"] >= 1.5 and total_score >= 75:
        return "Confirmed breakout zone"
    if overheated:
        return "Extended / avoid chasing"
    if oneil["near_high"] and not oneil["breakout"] and total_score >= 70:
        return "Awaiting breakout"
    if pd.notna(ma50) and current < ma50:
        return "Entry on hold"
    return "Watch"


def generate_analysis_text(analysis: dict[str, Any]) -> dict[str, Any]:
    latest = analysis["df"].iloc[-1]
    current = float(latest["Close"])
    oneil = analysis["oneil"]
    rs = analysis["rs"]
    vcp = analysis["vcp"]
    volume = analysis["volume"]
    final = analysis["final"]
    minervini = analysis["minervini"]

    positives: list[str] = []
    risks: list[str] = []
    paragraphs: list[str] = []

    if _ordered_ma(latest):
        positives.append("MA50 > MA150 > MA200 alignment")
        paragraphs.append("Price is above the main moving averages, with MA50 > MA150 > MA200 supporting an upward trend.")
    elif pd.notna(latest.get("MA200")) and current < latest["MA200"]:
        risks.append("Price below MA200")
        paragraphs.append("Price is below MA200. A trend-following approach calls for caution.")
    else:
        paragraphs.append("The moving-average structure is only partially aligned. Further trend confirmation is needed.")

    ma200_rising = next((item for item in minervini["checks"] if item["condition"] == "MA200 Rising"), None)
    if ma200_rising and ma200_rising["result"] == "PASS":
        positives.append("MA200 rising")
    elif ma200_rising and ma200_rising["result"] == "FAIL":
        risks.append("MA200 rising conditions not met")

    high_gap = _gap_from_high(current, minervini["high_52w"])
    if high_gap <= 5:
        positives.append(f"Distance below 52-week closing high: {high_gap:.1f}% within")
        paragraphs.append(f"Distance below the 52-week closing high: {high_gap:.1f}% below the high, within the near-high zone.")
    else:
        risks.append(f"Below 52-week closing high: {high_gap:.1f}% below")

    rs_3m = rs["values"].get("3 months")
    if rs_3m is not None:
        if rs_3m > 0:
            positives.append(f"3-month excess return: {rs_3m:+.1f}%p")
        else:
            risks.append(f"3-month excess return: {rs_3m:+.1f}%p")
        paragraphs.append(f"Three-month relative strength versus the benchmark: {rs_3m:+.1f}%p: {'outperforming the benchmark' if rs_3m > 0 else 'underperforming the benchmark'}.")

    if volume["ratio"] >= 1.2:
        positives.append(f"Volume / 20-session average: {volume['ratio']:.2f}x")
        paragraphs.append(f"Volume relative to its 20-session average: {volume['ratio']:.2f}x: {volume['status']} .")
    else:
        risks.append(f"Volume / 20-session average: {volume['ratio']:.2f}x")

    if vcp["status"] == "VCP like contraction detected":
        positives.append("Volatility and volume contraction across 60 sessions")
        paragraphs.append("The last three 20-session segments show contracting volatility and lower volume, resembling VCP-style compression.")
    elif vcp["status"] == "Partial contraction":
        risks.append("VCP conditions partially met")
    else:
        risks.append("VCP conditions not met")

    if final["ma20_gap"] is not None and final["ma20_gap"] > 15:
        risks.append(f"MA20 extension +{final['ma20_gap']:.1f}%")
        paragraphs.append(f"Price is above MA20 by {final['ma20_gap']:.1f}% which increases the risk of chasing an extended move.")
    if final["ma50_gap"] is not None and final["ma50_gap"] > 25:
        risks.append(f"MA50 extension +{final['ma50_gap']:.1f}%")

    if oneil["breakout"]:
        positives.append("Breakout above the previous 60-session high")
    else:
        risks.append("Previous 60-session high not yet cleared")

    trade_plan = analysis["trade_plan"]
    paragraphs.append(
        f"Entry approach: '{trade_plan['entry_method']}'. Entry and exit ranges are rule-based planning references, not automatic orders."
    )

    trend_summary = "Strong uptrend" if analysis["weinstein"]["score"] == 100 and minervini["score"] >= 75 else "Downtrend risk" if final["forced_exclusion"] else "Trend confirmation needed"
    strength_summary = "Strong relative performance" if rs["score"] >= 75 else "Mixed relative performance" if rs["score"] >= 50 else "Weak relative performance"
    if final["entry_status"] == "Extended / avoid chasing":
        timing_summary = "Strong setup but extended in the short term"
    elif final["entry_status"] == "Awaiting breakout":
        timing_summary = "Near highs; await breakout confirmation"
    elif final["entry_status"] == "Confirmed breakout zone":
        timing_summary = "Price breakout confirmed with volume"
    else:
        timing_summary = final["entry_status"]

    return {
        "paragraphs": paragraphs,
        "positive_factors": positives,
        "risk_factors": risks,
        "diagnosis": {"Trend": trend_summary, "Relative strength": strength_summary, "Entry timing": timing_summary},
    }


def analyze_stock(code: str, market: str = "KOSPI", period: str = "3 years", name: str | None = None) -> dict[str, Any]:
    stock, benchmark = load_stock_data(code, market, period)
    df = calculate_indicators(stock)
    benchmark = calculate_indicators(benchmark)
    minervini = calculate_minervini(df)
    weinstein = calculate_weinstein(df)
    oneil = calculate_oneil(df)
    rs = calculate_relative_strength(df, benchmark)
    vcp = calculate_vcp(df)
    volume = calculate_volume_analysis(df)
    final = calculate_final_score(minervini, oneil, weinstein, rs, vcp, df)
    trade_plan = calculate_trade_plan(df, oneil, final)
    analysis = {
        "code": normalize_code(code),
        "name": name or normalize_code(code),
        "market": market,
        "period": period,
        "df": df,
        "benchmark": benchmark,
        "minervini": minervini,
        "weinstein": weinstein,
        "oneil": oneil,
        "rs": rs,
        "vcp": vcp,
        "volume": volume,
        "final": final,
        "trade_plan": trade_plan,
    }
    analysis["text"] = generate_analysis_text(analysis)
    return analysis


def normalize_code(code: str) -> str:
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    return digits.zfill(6)[:6]


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    for col in required:
        if col not in result.columns:
            raise ValueError("Check the stock code or data provider availability.")
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result[required].dropna(subset=["Close"]).sort_index()


def _check(condition: str, current_value: Any, benchmark_label: str, benchmark_value: Any, passed: bool) -> dict[str, Any]:
    if pd.isna(current_value) or pd.isna(benchmark_value):
        result = "Insufficient data"
    else:
        result = "PASS" if passed else "FAIL"
    return {
        "condition": condition,
        "current": _format_number(current_value),
        "criteria": benchmark_label if isinstance(benchmark_label, str) and not benchmark_label.replace(",", "").replace(".", "").isdigit() else str(benchmark_label),
        "result": result,
    }


def _value_days_ago(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return float(series.iloc[-days - 1])


def _safe_ratio(value: Any, base: Any) -> float:
    if pd.isna(value) or pd.isna(base) or float(base) == 0:
        return 0.0
    return round(float(value) / float(base), 2)


def _gap(current: float, base: Any) -> float | None:
    if pd.isna(base) or float(base) == 0:
        return None
    return (current / float(base) - 1) * 100


def _gap_from_high(current: float, high: float) -> float:
    if not high:
        return 0.0
    return max((high - current) / high * 100, 0.0)


def _segment_volatility(segment: pd.DataFrame) -> float:
    high = float(segment["High"].max())
    low = float(segment["Low"].min())
    return (high - low) / high * 100 if high else 0.0


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _format_number(value: Any) -> str:
    if pd.isna(value):
        return "Insufficient data"
    return f"{float(value):,.0f}"


def _ordered_ma(latest: pd.Series) -> bool:
    values = [latest.get("MA50"), latest.get("MA150"), latest.get("MA200")]
    return all(pd.notna(v) for v in values) and values[0] > values[1] > values[2] and latest["Close"] > values[0]


def classify_breakout_volume(volume_ratio: float) -> str:
    if volume_ratio >= 1.5:
        return "Strong volume"
    if volume_ratio >= 1.2:
        return "Healthy"
    return "Weak volume"
