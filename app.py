from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from company_profiles import overview

from analyzer import analyze_stock, load_stock_listing, normalize_code, resolve_stock_query
from market_scanner import historical_success_cases, movers_top10_bundle_by_market, recommendation_top10_by_market, unusual_risk_top10_by_market


st.set_page_config(page_title="Korea Stock Breakout Lab", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze_stock(code: str, market: str, period: str) -> dict:
    return analyze_stock(code, market, period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_listing(market: str) -> pd.DataFrame:
    return load_stock_listing(market)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze_resolved(code: str, name: str, market: str, period: str) -> dict:
    return analyze_stock(code, market, period, name)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_company_overview(code: str) -> dict[str, str]:
    url = f"https://finance.naver.com/item/main.naver?code={normalize_code(code)}"
    return {"text": overview(normalize_code(code)), "source": url}


@st.cache_data(ttl=86400, show_spinner=False)
def cached_recommendations(mode: str, universe_size: int, cache_key: str) -> dict[str, pd.DataFrame]:
    return recommendation_top10_by_market(mode, per_market_universe=universe_size)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_movers(period: str, universe_size: int, cache_key: str) -> dict[str, dict[str, pd.DataFrame]]:
    return movers_top10_bundle_by_market(period, per_market_universe=universe_size)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_unusual_risk(universe_size: int, cache_key: str) -> dict[str, pd.DataFrame]:
    return unusual_risk_top10_by_market(per_market_universe=universe_size)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_success_cases(universe_size: int, forward_days: int, cache_key: str) -> list[dict]:
    return historical_success_cases(per_market_universe=universe_size, forward_days=forward_days, limit=10)


def main() -> None:
    init_session()
    render_style()

    render_sidebar_membership()
    render_product_header()

    linked = render_query_analysis()
    if linked:
        st.divider()

    tab_single, tab_compare, tab_recommend, tab_movers, tab_unusual, tab_simulator, tab_cases, tab_board = st.tabs(
        ["Stock Analysis", "Compare Stocks", "Top 10 Candidates", "Top Gainers and Losers", "Unusual Activity", "Investment Planner", "Historical Case Studies", "Investor Board"]
    )

    with tab_single:
        render_single_analysis()

    with tab_compare:
        render_multi_compare()

    with tab_recommend:
        render_market_recommendations()

    with tab_movers:
        render_market_movers()

    with tab_unusual:
        render_unusual_risk()

    with tab_simulator:
        render_investment_simulator()

    with tab_cases:
        render_success_cases()

    with tab_board:
        render_investor_board()


def render_product_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">Korea Breakout Intelligence</div>
                <h1>Korea Stock Breakout Lab</h1>
                <p>Minervini + O'Neil + Weinstein A technical dashboard that separates setup strength from entry timing using public trading principles.</p>
            </div>
            <div class="hero-panel">
                <div class="panel-label">Research workflow</div>
                <div class="panel-main">Find stocks, review trends, plan entries, compare market candidates</div>
                <div class="panel-sub">Review the signals and reasoning behind each technical assessment.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_membership() -> None:
    with st.sidebar:
        st.markdown("### Investor Club")
        st.caption("Interest registrations are saved in this browser session only.")
        with st.expander("Register interest (session demo)", expanded=False):
            email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            style = st.selectbox("Preferred style", ["Trend breakout", "Pullback", "Low volatility", "Avoid high-risk short-term trades"], key="signup_style")
            if st.button("Save interest", use_container_width=True):
                if email and "@" in email:
                    st.session_state["members"].insert(0, {"Email": email, "Preferred style": style, "Saved at": datetime.now().strftime("%Y-%m-%d %H:%M")})
                    st.success("Saved for this session.")
                else:
                    st.warning("Enter a valid email address.")


def render_query_analysis() -> bool:
    params = st.query_params
    code = params.get("analyze_code")
    market = params.get("market", "KOSPI")
    if not code:
        return False
    code = normalize_code(code)
    market = market if market in {"KOSPI", "KOSDAQ"} else "KOSPI"
    try:
        listing = cached_stock_listing(market)
        resolved = resolve_stock_query(code, market, listing)
        with st.spinner(f"{resolved['name']} ({code}) Loading stock analysis."):
            analysis = cached_analyze_resolved(resolved["code"], resolved["name"], market, "3 years")
        st.subheader("Linked Stock Analysis")
        render_analysis(analysis)
        if st.button("Close linked analysis"):
            st.query_params.clear()
        return True
    except Exception as exc:
        st.error(str(exc) if str(exc) else "Unable to analyze the linked stock.")
        return True


def render_single_analysis() -> None:
    with st.sidebar:
        st.header("Analysis Settings")
        market = st.selectbox("Market", ["KOSPI", "KOSDAQ"], index=0)
        query = st.text_input("Stock code or company name", value="005930", placeholder="005930 or Samsung Electronics", key="single_query")
        render_search_suggestions("single_query", market)
        period = st.selectbox("History window", ["1 year", "2 years", "3 years", "5 years"], index=2)
        run = st.button("Analyze", type="primary", use_container_width=True)

    if not run and not st.session_state.get("last_analysis"):
        st.info("Enter a six-digit Korean stock code or company name, then select Analyze.")
        render_history()
        return

    if run:
        try:
            with st.spinner("Loading market data and calculating trend and breakout signals."):
                listing = cached_stock_listing(market)
                resolved = resolve_stock_query(query, market, listing)
                analysis = cached_analyze_resolved(resolved["code"], resolved["name"], market, period)
            st.session_state["last_analysis"] = analysis
            add_history(analysis)
        except Exception as exc:
            st.error(str(exc) if str(exc) else "Check the stock code or data provider availability.")
            return

    analysis = st.session_state["last_analysis"]
    render_analysis(analysis)
    render_history()


def render_multi_compare() -> None:
    st.subheader("Compare Stocks")
    market = st.selectbox("Comparison market", ["KOSPI", "KOSDAQ"], index=0, key="compare_market")
    codes_text = st.text_input("Stocks to compare", value="005930,000660,NAVER", help="Enter up to 10 codes or company names, separated by commas.", key="compare_query")
    render_search_suggestions("compare_query", market, multi=True)
    period = st.selectbox("Comparison history window", ["1 year", "2 years", "3 years", "5 years"], index=2, key="compare_period")

    if st.button("Compare", type="primary"):
        queries = [item.strip() for item in codes_text.split(",") if item.strip()]
        queries = list(dict.fromkeys(queries))[:10]
        if not queries:
            st.error("Enter the stocks to compare.")
            return

        rows = []
        errors = []
        progress = st.progress(0)
        listing = cached_stock_listing(market)
        for idx, query in enumerate(queries, start=1):
            try:
                resolved = resolve_stock_query(query, market, listing)
                analysis = cached_analyze_resolved(resolved["code"], resolved["name"], market, period)
                rows.append(comparison_row(analysis))
            except Exception as exc:
                errors.append(f"{query}: {exc}")
            progress.progress(idx / len(queries))

        if rows:
            result = pd.DataFrame(rows).sort_values("Overall score", ascending=False)
            st.dataframe(result, use_container_width=True, hide_index=True)
        if errors:
            st.warning("Some stocks could not be analyzed.\n\n" + "\n".join(errors))


def render_market_recommendations() -> None:
    st.subheader("Current Technical Candidates")
    st.caption("Scans KOSPI and KOSDAQ candidates separately using the same analysis rules and trend exclusion checks.")
    mode = st.radio("Ranking basis", ["Daily", "Weekly", "Monthly"], horizontal=True)
    scan_mode = st.segmented_control("Scan scope", ["Quick", "Broad"], default="Quick", key="recommend_scan_mode")
    universe_size = 30 if scan_mode == "Quick" else 80
    cache_key = cache_bucket(mode)

    run = st.button("Calculate candidates", type="primary", key="refresh_recommendations")
    if run:
        cached_recommendations.clear()
        try:
            with st.spinner(f"Scanning {mode.lower()} candidates across up to {universe_size} stocks per market."):
                st.session_state["recommendation_result"] = {
                    "mode": mode,
                    "scan_mode": scan_mode,
                    "data": cached_recommendations(mode, universe_size, cache_key),
                }
        except Exception as exc:
            st.error(f"Unable to load candidates: {exc}")
            return

    saved = st.session_state.get("recommendation_result")
    if not saved:
        st.info("Select Calculate to scan candidates. Use Quick for a smaller universe.")
        return

    data = saved["data"]
    st.caption(f"Currently displayed: {saved['mode']} · {saved['scan_mode']}")

    kospi, kosdaq = st.columns(2)
    with kospi:
        st.markdown("**KOSPI Top 10 Candidates**")
        render_rank_table(data.get("KOSPI"))
    with kosdaq:
        st.markdown("**KOSDAQ Top 10 Candidates**")
        render_rank_table(data.get("KOSDAQ"))

    st.info("Price ranges are rule-based references, not execution instructions or guaranteed outcomes.")


def render_market_movers() -> None:
    st.subheader("Gainers and Losers TOP10")
    st.caption("Daily rankings use the listing provider's daily change. Monthly and yearly rankings compare closes across 20 and 240 trading sessions.")
    period = st.radio("Return horizon", ["Daily", "Monthly", "Yearly"], horizontal=True)
    scan_mode = st.segmented_control("Monthly and yearly scan scope", ["Quick", "Broad"], default="Quick", key="mover_scan_mode")
    universe_size = 60 if scan_mode == "Quick" else 160
    cache_key = cache_bucket("Daily")

    run = st.button("Calculate gainers and losers", type="primary", key="refresh_movers")
    if run:
        cached_movers.clear()
        try:
            with st.spinner(f"{period} Calculating gainers and losers."):
                st.session_state["mover_result"] = {
                    "period": period,
                    "scan_mode": scan_mode,
                    "data": cached_movers(period, universe_size, cache_key),
                }
        except Exception as exc:
            st.error(f"Unable to load movers: {exc}")
            return

    saved = st.session_state.get("mover_result")
    if not saved:
        st.info("Select Calculate to load rankings. Monthly and yearly scans need historical data and can take longer.")
        return

    movers = saved["data"]
    st.caption(f"Currently displayed: {saved['period']} · {saved['scan_mode']}")

    for market in ["KOSPI", "KOSDAQ"]:
        st.markdown(f"### {market}")
        up_col, down_col = st.columns(2)
        with up_col:
            st.markdown("**Top 10 gainers in the selected universe**")
            render_rank_table(movers.get(market, {}).get("Rising"))
        with down_col:
            st.markdown("**Bottom 10 returns in the selected universe**")
            render_rank_table(movers.get(market, {}).get("Falling"))


def render_unusual_risk() -> None:
    st.subheader("Unusual Activity Watchlist")
    st.caption("A technical risk screen combining volume spikes, rapid price moves, extension, intraday range, and small capitalization.")
    scan_mode = st.segmented_control("Scan scope", ["Quick", "Broad"], default="Quick", key="unusual_scan_mode")
    universe_size = 60 if scan_mode == "Quick" else 140
    cache_key = cache_bucket("Daily")

    run = st.button("Calculate risk signals", type="primary", key="refresh_unusual")
    if run:
        cached_unusual_risk.clear()
        try:
            with st.spinner(f"Scanning risk signals across up to {universe_size} stocks per market."):
                st.session_state["unusual_result"] = {
                    "scan_mode": scan_mode,
                    "data": cached_unusual_risk(universe_size, cache_key),
                }
        except Exception as exc:
            st.error(f"Unable to load risk signals: {exc}")
            return

    saved = st.session_state.get("unusual_result")
    if not saved:
        st.info("Select Calculate to load up to 10 unusual-activity candidates for each market.")
        return

    st.caption(f"Currently displayed: {saved['scan_mode']} · Refresh on request; daily cache")
    data = saved["data"]
    kospi, kosdaq = st.columns(2)
    with kospi:
        st.markdown("**KOSPI Top 10 Risk Signals**")
        render_rank_table(data.get("KOSPI"))
    with kosdaq:
        st.markdown("**KOSDAQ Top 10 Risk Signals**")
        render_rank_table(data.get("KOSDAQ"))

    st.warning("These signals do not establish market manipulation or misconduct. This list highlights technical anomalies for further research.")


def render_investment_simulator() -> None:
    st.subheader("Interactive Investment Planner")
    st.caption("Choose a horizon, style, capital, and loss parameter to generate a rule-based entry and exit plan.")

    source = st.radio("Stock source", ["Enter a stock", "Choose a technical candidate"], horizontal=True)
    selected_query = "005930"
    selected_market = "KOSPI"

    if source == "Choose a technical candidate":
        sim_mode = st.selectbox("Ranking basis", ["Daily", "Weekly", "Monthly"], index=1, key="sim_reco_mode")
        if st.button("Load candidates", key="sim_load_recommendations"):
            with st.spinner("Loading technical candidates."):
                data = cached_recommendations(sim_mode, 30, cache_bucket(sim_mode))
                choices = []
                for market, frame in data.items():
                    if frame is not None and not frame.empty:
                        for row in frame.itertuples(index=False):
                            choices.append({"label": f"{market} · {row.Stock}", "market": market, "query": row.Stock.split("(")[-1].replace(")", "")})
                st.session_state["simulation_choices"] = choices
        choices = st.session_state.get("simulation_choices", [])
        if choices:
            choice_label = st.selectbox("Select stock", [item["label"] for item in choices])
            selected = next(item for item in choices if item["label"] == choice_label)
            selected_query = selected["query"]
            selected_market = selected["market"]
        else:
            st.info("Load candidates first, or choose direct stock entry.")
    else:
        col_a, col_b = st.columns([0.7, 0.3])
        with col_b:
            selected_market = st.selectbox("Market", ["KOSPI", "KOSDAQ"], index=0, key="sim_market")
        with col_a:
            selected_query = st.text_input("Stock code or company name", value="005930", key="sim_query")
            render_search_suggestions("sim_query", selected_market)

    with st.form("simulation_form"):
        c1, c2, c3, c4 = st.columns(4)
        horizon = c1.selectbox("Planning horizon", ["1 month", "3 months", "6 months", "1 year"], index=1)
        temperament = c2.selectbox("Risk style", ["Conservative", "Balanced", "Aggressive"], index=1)
        capital = c3.number_input("Planned capital in KRW", min_value=100000, max_value=1000000000, value=5000000, step=100000)
        max_risk = c4.slider("Loss parameter (%)", min_value=2.0, max_value=20.0, value=7.0, step=0.5)
        submitted = st.form_submit_button("Build plan", type="primary")

    if not submitted:
        render_simulator_guide()
        return

    try:
        listing = cached_stock_listing(selected_market)
        resolved = resolve_stock_query(selected_query, selected_market, listing)
        with st.spinner(f"{resolved['name']} building an investment plan."):
            analysis = cached_analyze_resolved(resolved["code"], resolved["name"], selected_market, "3 years")
        plan = build_simulation_plan(analysis, horizon, temperament, capital, max_risk)
        render_simulation_result(analysis, plan)
    except Exception as exc:
        st.error(str(exc) if str(exc) else "Unable to build the plan.")


def render_success_cases() -> None:
    st.subheader("Historical Case Studies")
    st.caption("Finds retrospectively selected examples with strong historical scores followed by a price rise. This is not an overall strategy hit rate.")
    horizon = st.radio("Forward observation window", ["1 month", "3 months", "6 months"], index=1, horizontal=True)
    scan_mode = st.segmented_control("Scan scope", ["Quick", "Broad"], default="Quick", key="case_scan_mode")
    forward_days = {"1 month": 20, "3 months": 60, "6 months": 120}[horizon]
    universe_size = 25 if scan_mode == "Quick" else 60
    cache_key = cache_bucket("Daily")

    run = st.button("Find historical cases", type="primary", key="find_success_cases")
    if run:
        cached_success_cases.clear()
        try:
            with st.spinner(f"Searching historical cases over a {horizon} window. The initial scan may take a few minutes."):
                st.session_state["success_cases"] = {
                    "horizon": horizon,
                    "scan_mode": scan_mode,
                    "cases": cached_success_cases(universe_size, forward_days, cache_key),
                }
        except Exception as exc:
            st.error(f"Unable to load historical cases: {exc}")
            return

    saved = st.session_state.get("success_cases")
    if not saved:
        st.info("Run the search to display up to 10 retrospectively selected rising-stock case studies.")
        return

    st.caption(f"Currently displayed: {saved['horizon']} · {saved['scan_mode']} · Refresh on request; daily cache")
    cases = saved["cases"]
    if not cases:
        st.warning("No matching cases were found. Try a broader scan or a different horizon.")
        return

    for idx, case in enumerate(cases, start=1):
        render_case_post(idx, case, saved["horizon"])

    st.info("Selected past winners do not establish future performance. Consider market conditions, disclosures, financial events, and losing cases.")


def render_case_post(rank: int, case: dict, horizon: str) -> None:
    st.markdown(
        f"""
        <div class="case-card">
            <div class="case-kicker">CASE #{rank} · {case['Market']} · Signal date {case['Signal date']}</div>
            <div class="case-title">{case['Stock']} - A historical technical setup</div>
            <div class="case-lead">
                The signal-day score was <b>{case['Signal score']:.1f} points</b>, with a rating of <b>{case['Signal rating']}</b>.
                The peak price change within {horizon} was <b>{case['Peak return']:+.1f}%</b>.
            </div>
            <div class="case-grid">
                <div><span>Signal price</span><b>{case['Signal price']:,.0f} KRW</b></div>
                <div><span>Entry status</span><b>{case['Entry status']}</b></div>
                <div><span>Entry approach</span><b>{case['Entry approach']}</b></div>
                <div><span>Entry range</span><b>{case['Entry range']}</b></div>
                <div><span>Return at observation end</span><b>{case['Closing return']:+.1f}%</b></div>
                <div><span>Lowest price versus signal</span><b>{case['Lowest price return']:+.1f}%</b></div>
            </div>
            <div class="case-body">
                <b>Signals at the time:</b> {case['Key signals'] or 'Trend conditions and relative strength improved together.'}
            </div>
            <a class="case-link" href="{case['Analysis link']}">Open current stock analysis</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simulator_guide() -> None:
    a, b, c = st.columns(3)
    a.markdown('<div class="sim-card"><div class="sim-title">1. Select stock</div><div class="sim-body">Enter a stock directly or select a technical candidate.</div></div>', unsafe_allow_html=True)
    b.markdown('<div class="sim-card"><div class="sim-title">2. Choose a risk style</div><div class="sim-body">Position tranches and target assumptions vary by risk style.</div></div>', unsafe_allow_html=True)
    c.markdown('<div class="sim-card"><div class="sim-title">3. Review the plan</div><div class="sim-body">Review staged entries, defense levels, profit targets, and estimated shares.</div></div>', unsafe_allow_html=True)


def build_simulation_plan(analysis: dict, horizon: str, temperament: str, capital: int, max_risk: float) -> dict:
    latest = analysis["df"].iloc[-1]
    current = float(latest["Close"])
    trade = analysis["trade_plan"]
    final = analysis["final"]
    oneil = analysis["oneil"]

    months = {"1 month": 1, "3 months": 3, "6 months": 6, "1 year": 12}[horizon]
    settings = {
        "Conservative": {"risk": 0.6, "first": 0.35, "second": 0.35, "third": 0.30, "target": (0.08, 0.16), "trail": 0.07},
        "Balanced": {"risk": 1.0, "first": 0.45, "second": 0.35, "third": 0.20, "target": (0.12, 0.24), "trail": 0.10},
        "Aggressive": {"risk": 1.35, "first": 0.60, "second": 0.25, "third": 0.15, "target": (0.18, 0.35), "trail": 0.14},
    }[temperament]

    buy_low = trade["buy_low"] or current * 0.98
    buy_high = trade["buy_high"] or current * 1.02
    entry_mid = (buy_low + buy_high) / 2
    stop_reference = trade["risk_sell_high"] or current * (1 - max_risk / 100)
    stop_price = max(min(stop_reference, entry_mid * (1 - max_risk / 100)), entry_mid * 0.75)
    per_share_risk = max(entry_mid - stop_price, entry_mid * 0.01)
    risk_budget = capital * (max_risk / 100) * settings["risk"]
    max_position_value = min(capital, risk_budget / per_share_risk * entry_mid)
    shares = int(max_position_value // entry_mid)
    used_cash = shares * entry_mid

    target_low = max(trade["profit_sell_low"] or entry_mid * 1.12, entry_mid * (1 + settings["target"][0]))
    target_high = max(trade["profit_sell_high"] or entry_mid * 1.22, entry_mid * (1 + settings["target"][1]))
    time_stop_days = {1: 10, 3: 20, 6: 35, 12: 55}[months]

    if final["forced_exclusion"]:
        scenario = "Wait for trend recovery"
        summary = "Key trend conditions are weak; keep this stock on a watchlist instead of treating the plan as an entry signal."
    elif final["entry_status"] == "Confirmed breakout zone":
        scenario = "Breakout follow-through plan"
        summary = "Price and volume confirm a breakout. Review staged entries near the pivot and manage the defense level closely."
    elif final["entry_status"] == "Awaiting breakout":
        scenario = "Wait for pivot confirmation"
        summary = "The breakout is not confirmed. Wait for price and volume confirmation."
    elif final["entry_status"] == "Extended / avoid chasing":
        scenario = "Wait for a pullback"
        summary = "Price is extended. Review behavior near MA20 or the previous pivot before entering."
    else:
        scenario = "Watch and confirm"
        summary = "The trend is not fully confirmed. Review a smaller staged position or wait for stronger signals."

    entry_steps = [
        {"Step": "First", "Condition": trade["entry_method"], "Allocation": f"{settings['first'] * 100:.0f}%", "Price range": price_range(buy_low, buy_high)},
        {"Step": "Second", "Condition": "Support above pivot or MA20 after entry", "Allocation": f"{settings['second'] * 100:.0f}%", "Price range": price_range(entry_mid * 0.98, entry_mid * 1.04)},
        {"Step": "Third", "Condition": "Sustained volume or another new-high breakout", "Allocation": f"{settings['third'] * 100:.0f}%", "Price range": price_range(entry_mid * 1.02, entry_mid * 1.08)},
    ]
    exit_steps = [
        {"Type": "Defense", "Condition": "Entry thesis fails or reference support breaks", "Price range": price_range(stop_price * 0.99, stop_price), "Action": "Review reducing or closing the position"},
        {"Type": "First profit-taking", "Condition": "Lower target reached", "Price range": won(target_low), "Action": "30~50% review a partial exit"},
        {"Type": "Second profit-taking", "Condition": "Upper target reached or extension returns", "Price range": won(target_high), "Action": "Review staged exits for remaining shares"},
        {"Type": "Trailing defense", "Condition": f"From peak: {settings['trail'] * 100:.0f}% or greater pullback", "Price range": "Track from market price", "Action": "Review protecting remaining shares"},
    ]

    return {
        "scenario": scenario,
        "summary": summary,
        "horizon": horizon,
        "temperament": temperament,
        "capital": capital,
        "shares": shares,
        "used_cash": used_cash,
        "entry_mid": entry_mid,
        "stop_price": stop_price,
        "risk_budget": risk_budget,
        "target_low": target_low,
        "target_high": target_high,
        "time_stop_days": time_stop_days,
        "entry_steps": entry_steps,
        "exit_steps": exit_steps,
        "signals": [
            f"Overall score {final['total_score']:.1f} points / {final['rating']}",
            f"Entry status: {final['entry_status']}",
            f"60-session breakout: {'Confirmed' if oneil['breakout'] else 'Pending'}",
            f"Volume / 20-session average: {analysis['volume']['ratio']:.2f}x",
        ],
    }


def render_simulation_result(analysis: dict, plan: dict) -> None:
    st.markdown(
        f"""
        <div class="stock-header">
            <div>
                <div class="stock-meta">{analysis['market']} · {analysis['code']} · {plan['horizon']} · {plan['temperament']}</div>
                <div class="stock-name">{analysis['name']} Plan</div>
                <div class="stock-sub">{plan['summary']}</div>
            </div>
            <div class="badge-watch">{plan['scenario']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Planned capital", won(plan["capital"]))
    c2.metric("Estimated shares", f"{plan['shares']:,} shares")
    c3.metric("Reference entry price", won(plan["entry_mid"]))
    c4.metric("Defense reference", won(plan["stop_price"]))

    c5, c6, c7 = st.columns(3)
    c5.metric("Estimated position value", won(plan["used_cash"]))
    c6.metric("Risk budget", won(plan["risk_budget"]))
    c7.metric("Profit-taking range", price_range(plan["target_low"], plan["target_high"]))

    st.markdown("**Entry Plan**")
    st.dataframe(pd.DataFrame(plan["entry_steps"]), use_container_width=True, hide_index=True)
    st.markdown("**Exit and Defense Plan**")
    st.dataframe(pd.DataFrame(plan["exit_steps"]), use_container_width=True, hide_index=True)
    st.markdown("**Key Signals**")
    for signal in plan["signals"]:
        st.markdown(f"- {signal}")
    st.info(f"{plan['time_stop_days']}trading sessions: reassess if the entry thesis has not strengthened. This is a hypothetical planning tool; returns and maximum losses are not guaranteed.")


def render_rank_table(frame: pd.DataFrame | None) -> None:
    if frame is None or frame.empty:
        st.warning("No qualifying stocks to display.")
        return
    config = {}
    if "Analysis link" in frame.columns:
        config["Analysis link"] = st.column_config.LinkColumn("Stock analysis", display_text="Open")
    st.dataframe(frame, use_container_width=True, hide_index=True, column_config=config)


def render_analysis(analysis: dict) -> None:
    df = analysis["df"]
    latest = df.iloc[-1]
    final = analysis["final"]
    rs_3m = analysis["rs"]["values"].get("3 months")

    render_stock_header(analysis, latest, final)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(f"""
        <div class="question-card">
            <div class="question-title">Is the technical setup strong?</div>
            <div class="question-answer">{final["good_stock"]} - {final["total_score"]:.1f} points</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown(f"""
        <div class="question-card">
            <div class="question-title">Is this a suitable entry zone?</div>
            <div class="question-answer">{final["good_price"]} - {final["entry_status"]}</div>
        </div>
        """, unsafe_allow_html=True)

    cols = st.columns(5)
    cols[0].metric("Last close", f"{latest['Close']:,.0f} KRW")
    cols[1].metric("Overall score", f"{final['total_score']:.1f} / 100")
    cols[2].metric("Final rating", final["stars"], final["rating"])
    cols[3].metric("Entry status", final["entry_status"])
    cols[4].metric("3-month excess return", format_rs(rs_3m))

    st.divider()
    left, right = st.columns([1.1, 0.9])
    with left:
        render_scoreboard(analysis)
        render_diagnosis(analysis)
    with right:
        render_positive_risk(analysis)

    render_trade_plan(analysis)

    render_company_overview(analysis)
    render_chart(analysis)

    c1, c2 = st.columns(2)
    with c1:
        render_minervini_table(analysis)
        render_price_table(analysis)
    with c2:
        render_volume_vcp(analysis)
        render_support_levels(analysis)

    render_text_analysis(analysis)


def render_stock_header(analysis: dict, latest: pd.Series, final: dict) -> None:
    badge_class = "badge-good" if final["good_stock"] == "YES" else "badge-watch" if final["good_stock"] == "WATCH" else "badge-risk"
    st.markdown(
        f"""
        <div class="stock-header">
            <div>
                <div class="stock-meta">{analysis['market']} · {analysis['code']}</div>
                <div class="stock-name">{analysis['name']}</div>
                <div class="stock-sub">Last close {float(latest['Close']):,.0f} KRW · Overall score {final['total_score']:.1f} · {final['stars']} {final['rating']}</div>
            </div>
            <div class="{badge_class}">{final['good_stock']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scoreboard(analysis: dict) -> None:
    st.subheader("Score Breakdown")
    rows = [
        {"Component": "Minervini Trend", "Score": analysis["minervini"]["score"]},
        {"Component": "O'Neil Breakout", "Score": analysis["oneil"]["score"]},
        {"Component": "Weinstein Stage", "Score": analysis["weinstein"]["score"]},
        {"Component": "Relative Strength", "Score": analysis["rs"]["score"]},
        {"Component": "VCP", "Score": analysis["vcp"]["score"]},
        {"Component": "Overall", "Score": analysis["final"]["total_score"]},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_diagnosis(analysis: dict) -> None:
    st.subheader("Core Assessment")
    diagnosis = analysis["text"]["diagnosis"]
    st.write(f"**Trend:** {diagnosis['Trend']}")
    st.write(f"**Relative strength:** {diagnosis['Relative strength']}")
    st.write(f"**Entry timing:** {diagnosis['Entry timing']}")


def render_positive_risk(analysis: dict) -> None:
    st.subheader("Reasons Behind the Score")
    pos_col, risk_col = st.columns(2)
    with pos_col:
        st.markdown("**Positive Signals**")
        for item in analysis["text"]["positive_factors"] or ["Few clear positive signals."]:
            st.markdown(f"- {item}")
    with risk_col:
        st.markdown("**Risk Signals**")
        for item in analysis["text"]["risk_factors"] or ["Few clear risk signals."]:
            st.markdown(f"- {item}")


def render_trade_plan(analysis: dict) -> None:
    st.subheader("Entry and Exit Reference Ranges")
    plan = analysis["trade_plan"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry approach", plan["entry_method"])
    c2.metric("Entry range", price_range(plan["buy_low"], plan["buy_high"]))
    c3.metric("Defensive exit range", price_range(plan["risk_sell_low"], plan["risk_sell_high"]))
    c4.metric("Profit-taking range", price_range(plan["profit_sell_low"], plan["profit_sell_high"]))
    st.caption(plan["note"])
    st.caption("Ranges are analytical references. They do not guarantee outcomes or trigger orders.")


def render_company_overview(analysis: dict) -> None:
    st.subheader("Company Overview")
    try:
        overview = cached_company_overview(analysis["code"])
        st.markdown(overview["text"].replace("\n", "\n\n"))
        st.caption("General English business summary. Not a live company filing or an automatically translated disclosure.")
        st.link_button("Original company information (Korean)", overview["source"])
    except Exception:
        st.caption("Company information is currently unavailable.")


def render_chart(analysis: dict) -> None:
    st.subheader("Chart")
    chart_period = st.radio("Chart window", ["3 months", "6 months", "1 year", "2 years"], index=2, horizontal=True)
    days = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}[chart_period]
    df = analysis["df"].tail(min(days, len(analysis["df"])))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    for ma, color in [("MA20", "#2563eb"), ("MA50", "#16a34a"), ("MA150", "#f59e0b"), ("MA200", "#dc2626")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1.6, color=color)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#64748b"), row=2, col=1)
    fig.update_layout(
        title=f"{analysis['name']} ({analysis['code']}) Price / Moving averages / Volume",
        height=660,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=30),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def render_minervini_table(analysis: dict) -> None:
    st.subheader("Minervini Checklist")
    rows = [
        {"Condition": item["condition"], "Current value": item["current"], "Reference": item["criteria"], "Result": item["result"]}
        for item in analysis["minervini"]["checks"]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_price_table(analysis: dict) -> None:
    st.subheader("Key Price Levels")
    df = analysis["df"]
    latest = df.iloc[-1]
    minervini = analysis["minervini"]
    oneil = analysis["oneil"]
    final = analysis["final"]
    low_20d = float(df["Low"].tail(min(20, len(df))).min())
    high_gap = (minervini["high_52w"] - latest["Close"]) / minervini["high_52w"] * 100 if minervini["high_52w"] else 0
    rows = [
        ("Last close", won(latest["Close"])),
        ("MA20", won(latest.get("MA20"))),
        ("MA50", won(latest.get("MA50"))),
        ("MA150", won(latest.get("MA150"))),
        ("MA200", won(latest.get("MA200"))),
        ("52-week closing high", won(minervini["high_52w"])),
        ("52-week closing low", won(minervini["low_52w"])),
        ("60-session high", won(oneil["high_60d"])),
        ("20-session low", won(low_20d)),
        ("Distance below 52-week closing high", f"{high_gap:.1f}%"),
        ("Distance from MA20", pct(final["ma20_gap"])),
        ("Distance from MA50", pct(final["ma50_gap"])),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Value"]), use_container_width=True, hide_index=True)


def render_volume_vcp(analysis: dict) -> None:
    st.subheader("Volume and VCP")
    volume = analysis["volume"]
    vcp = analysis["vcp"]
    st.write(f"**Latest volume:** {volume['today']:,.0f} shares")
    st.write(f"**20-session average:** {num(volume['vol20'])} shares")
    st.write(f"**50-session average:** {num(volume['vol50'])} shares")
    st.write(f"**Volume multiple:** {volume['ratio']:.2f}x - {volume['status']}")
    st.write(f"**VCP:** {vcp['status']}")
    if vcp["volatilities"]:
        st.dataframe(pd.DataFrame([
            {"Segment": k, "Volatility": f"{vcp['volatilities'][k]:.1f}%", "Average volume": f"{vcp['volumes'][k]:,.0f} shares"}
            for k in ["A", "B", "C"]
        ]), use_container_width=True, hide_index=True)


def render_support_levels(analysis: dict) -> None:
    st.subheader("Technical Support References")
    df = analysis["df"]
    latest = df.iloc[-1]
    low_20d = float(df["Low"].tail(min(20, len(df))).min())
    st.caption("Technical reference levels; they are not guaranteed execution or stop-loss prices.")
    rows = [
        {"Component": "MA20", "Price": won(latest.get("MA20"))},
        {"Component": "MA50", "Price": won(latest.get("MA50"))},
        {"Component": "20-session low", "Price": won(low_20d)},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_text_analysis(analysis: dict) -> None:
    st.subheader("Analysis Notes")
    for paragraph in analysis["text"]["paragraphs"]:
        st.write(paragraph)
    st.info("This app provides research only. It does not connect to brokerage accounts or place orders.")


def render_investor_board() -> None:
    st.subheader("Investor Board")
    st.caption("Session-only community demo. Posts are not shared between visitors. Avoid misleading claims or guaranteed-return statements.")

    left, right = st.columns([0.72, 0.28])
    with right:
        st.markdown("**Write a Post**")
        nickname = st.text_input("Display name", value="Guest investor", key="board_nickname")
        category = st.selectbox("Category", ["Watchlist", "Breakout watch", "Risk notes", "Question"], key="board_category")
        title = st.text_input("Title", placeholder="Example: Is volume confirming the 005930 breakout?", key="board_title")
        body = st.text_area("Body", height=140, placeholder="Include your reasoning and the signals you are monitoring.", key="board_body")
        if st.button("Post", type="primary", use_container_width=True):
            if title.strip() and body.strip():
                st.session_state["board_posts"].insert(
                    0,
                    {
                        "Posted at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Display name": nickname.strip() or "Guest investor",
                        "Category": category,
                        "Title": title.strip(),
                        "Body": body.strip(),
                    },
                )
                st.success("Post added to this session.")
            else:
                st.warning("Enter a title and body.")

    with left:
        st.markdown("**Session Discussion Demo**")
        if not st.session_state["board_posts"]:
            st.info("No posts yet. Add your first research note.")
            return
        for post in st.session_state["board_posts"][:20]:
            st.markdown(
                f"""
                <div class="post-card">
                    <div class="post-top"><span>{escape(post['Category'])}</span><span>{escape(post['Posted at'])} · {escape(post['Display name'])}</span></div>
                    <div class="post-title">{escape(post['Title'])}</div>
                    <div class="post-body">{escape(post['Body'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history() -> None:
    st.subheader("Recent Analyses")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("Stocks analyzed in this session appear here.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def comparison_row(analysis: dict) -> dict:
    latest = analysis["df"].iloc[-1]
    return {
        "Stock": f"{analysis['name']} ({analysis['code']})",
        "Last close": round(float(latest["Close"]), 0),
        "Overall score": analysis["final"]["total_score"],
        "Minervini": analysis["minervini"]["score"],
        "O'Neil": analysis["oneil"]["score"],
        "Weinstein": analysis["weinstein"]["score"],
        "RS": analysis["rs"]["score"],
        "VCP": analysis["vcp"]["score"],
        "Rating": f"{analysis['final']['stars']} {analysis['final']['rating']}",
        "Entry status": analysis["final"]["entry_status"],
        "Entry approach": analysis["trade_plan"]["entry_method"],
        "Entry range": price_range(analysis["trade_plan"]["buy_low"], analysis["trade_plan"]["buy_high"]),
    }


def add_history(analysis: dict) -> None:
    latest = analysis["df"].iloc[-1]
    row = {
        "Company": analysis["name"],
        "Stock code": analysis["code"],
        "Analysis time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Last close": f"{latest['Close']:,.0f} KRW",
        "Overall score": analysis["final"]["total_score"],
        "Rating": f"{analysis['final']['stars']} {analysis['final']['rating']}",
        "Entry status": analysis["final"]["entry_status"],
    }
    st.session_state["history"] = [row] + [item for item in st.session_state["history"] if item["Stock code"] != analysis["code"]]
    st.session_state["history"] = st.session_state["history"][:20]


def render_search_suggestions(input_key: str, market: str, multi: bool = False) -> None:
    current = str(st.session_state.get(input_key, "")).strip()
    search_text = current.split(",")[-1].strip() if multi else current
    if len(search_text) < 1 or search_text.isdigit():
        return

    try:
        listing = cached_stock_listing(market)
        native = listing.get('NativeName', listing['Name'])
        matches = listing[listing["Name"].str.contains(search_text, case=False, na=False, regex=False) | native.str.contains(search_text, case=False, na=False, regex=False)].head(6)
    except Exception:
        return

    if matches.empty:
        st.caption("No suggestions found.")
        return

    st.markdown('<div class="suggest-label">Search suggestions</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, len(matches)))
    for idx, row in enumerate(matches.itertuples(index=False)):
        label = f"{row.Name} ({row.Code})"
        value = _suggestion_value(input_key, row.Name, multi)
        cols[idx % len(cols)].button(
            label,
            key=f"suggest_{input_key}_{row.Code}_{idx}",
            on_click=set_search_value,
            args=(input_key, value),
            use_container_width=True,
        )


def set_search_value(input_key: str, value: str) -> None:
    st.session_state[input_key] = value


def _suggestion_value(input_key: str, name: str, multi: bool) -> str:
    if not multi:
        return name
    current = str(st.session_state.get(input_key, ""))
    parts = current.split(",")
    prefix = ",".join(parts[:-1]).strip()
    return f"{prefix},{name}" if prefix else name


def init_session() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_analysis", None)
    st.session_state.setdefault("recommendation_result", None)
    st.session_state.setdefault("mover_result", None)
    st.session_state.setdefault("unusual_result", None)
    st.session_state.setdefault("simulation_choices", [])
    st.session_state.setdefault("success_cases", None)
    st.session_state.setdefault("members", [])
    st.session_state.setdefault(
        "board_posts",
        [
            {
                "Posted at": "Today 09:20",
                "Display name": "BreakoutLab",
                "Category": "Breakout watch",
                "Title": "Demo note: watching support after a volume breakout",
                "Body": "I am watching whether price holds above the pivot as volume settles after the initial breakout. This is a demonstration post.",
            },
            {
                "Posted at": "Today 08:45",
                "Display name": "RSWatch",
                "Category": "Watchlist",
                "Title": "Demo note: tracking stocks with positive relative strength",
                "Body": "I am keeping a separate watchlist of stocks holding above MA50 during benchmark pullbacks. This is a demonstration post.",
            },
        ],
    )


def valid_code(code: str) -> bool:
    return len(str(code).strip()) == 6 and str(code).strip().isdigit()


def won(value) -> str:
    if value is None or pd.isna(value):
        return "Insufficient data"
    return f"{float(value):,.0f} KRW"


def price_range(low, high) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "Not provided"
    return f"{float(low):,.0f}~{float(high):,.0f} KRW"


def num(value) -> str:
    if value is None or pd.isna(value):
        return "Insufficient data"
    return f"{float(value):,.0f}"


def pct(value) -> str:
    if value is None or pd.isna(value):
        return "Insufficient data"
    return f"{float(value):+.1f}%"


def format_rs(value) -> str:
    if value is None:
        return "Insufficient data"
    return f"{value:+.1f}%p"


def cache_bucket(mode: str) -> str:
    now = datetime.now()
    if mode == "Weekly":
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"
    if mode == "Monthly":
        return now.strftime("%Y-%m")
    return now.strftime("%Y-%m-%d")


def render_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748b;
            --line: #d8dee9;
            --surface: #ffffff;
            --soft: #f6f8fb;
            --accent: #0f766e;
            --accent-2: #1d4ed8;
            --danger: #b91c1c;
        }
        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, #eef3f8 46%, #f8fafc 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.4rem;
            max-width: 1500px;
        }
        section[data-testid="stSidebar"] {
            background: #0f172a;
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #e5e7eb;
        }
        .hero {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
            gap: 18px;
            align-items: stretch;
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 26px 28px;
            background:
                linear-gradient(135deg, rgba(255,255,255,.96), rgba(240,247,244,.94)),
                repeating-linear-gradient(90deg, rgba(15,118,110,.08) 0, rgba(15,118,110,.08) 1px, transparent 1px, transparent 42px);
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }
        .eyebrow {
            color: var(--accent);
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .hero h1 {
            font-size: 2.4rem;
            line-height: 1.15;
            margin: 0 0 10px 0;
            letter-spacing: 0;
            color: #0b1220;
        }
        .hero p {
            margin: 0;
            max-width: 860px;
            color: #475569;
            font-size: 1.04rem;
        }
        .hero-panel {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 18px;
            background: rgba(255,255,255,.82);
        }
        .panel-label {
            color: #64748b;
            font-size: .86rem;
            font-weight: 700;
            margin-bottom: 7px;
        }
        .panel-main {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .panel-sub {
            color: #64748b;
            font-size: .92rem;
            line-height: 1.45;
        }
        .stock-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 22px 24px;
            background: #ffffff;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin: 12px 0 16px 0;
        }
        .stock-meta {
            color: var(--accent-2);
            font-size: .9rem;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .stock-name {
            color: #0f172a;
            font-size: 2.2rem;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: 0;
        }
        .stock-sub {
            color: #64748b;
            margin-top: 7px;
            font-size: 1rem;
        }
        .badge-good,
        .badge-watch,
        .badge-risk {
            min-width: 96px;
            text-align: center;
            border-radius: 8px;
            padding: 12px 14px;
            font-weight: 900;
            border: 1px solid;
        }
        .badge-good { color: #065f46; background: #ecfdf5; border-color: #a7f3d0; }
        .badge-watch { color: #92400e; background: #fffbeb; border-color: #fde68a; }
        .badge-risk { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
        .question-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 20px;
            background: var(--surface);
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .question-title {
            color: #475569;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .question-answer {
            color: #0f172a;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.25;
            word-break: keep-all;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stTabs"] button {
            font-weight: 750;
        }
        div[data-testid="stMetricValue"] { font-size: 1.15rem; white-space: normal; overflow-wrap: anywhere; }
        .stock-name, .question-answer, .stock-sub, .hero h1 { overflow-wrap: anywhere; }
        .post-card {
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 16px 18px;
            background: #ffffff;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .post-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: #64748b;
            font-size: .86rem;
            margin-bottom: 8px;
        }
        .post-top span:first-child {
            color: var(--accent);
            font-weight: 800;
        }
        .post-title {
            font-size: 1.05rem;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 7px;
        }
        .post-body {
            color: #475569;
            line-height: 1.55;
        }
        .suggest-label {
            color: #64748b;
            font-size: .82rem;
            font-weight: 800;
            margin: 4px 0 6px 0;
        }
        .sim-card {
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 18px;
            background: #ffffff;
            min-height: 118px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .sim-title {
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .sim-body {
            color: #64748b;
            line-height: 1.5;
        }
        .case-card {
            border: 1px solid #d7e2ea;
            border-radius: 8px;
            padding: 22px 24px;
            background: #ffffff;
            margin: 0 0 18px 0;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        .case-kicker {
            color: #0f766e;
            font-size: .86rem;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .case-title {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 900;
            line-height: 1.28;
            margin-bottom: 10px;
        }
        .case-lead {
            color: #475569;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 16px;
        }
        .case-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }
        .case-grid div {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            background: #f8fafc;
        }
        .case-grid span {
            display: block;
            color: #64748b;
            font-size: .78rem;
            font-weight: 750;
            margin-bottom: 4px;
        }
        .case-grid b {
            color: #0f172a;
            font-size: .96rem;
        }
        .case-body {
            color: #475569;
            line-height: 1.55;
            margin-bottom: 12px;
        }
        .case-link {
            color: #1d4ed8;
            font-weight: 850;
            text-decoration: none;
        }
        @media (max-width: 900px) {
            .hero {
                grid-template-columns: 1fr;
                padding: 22px;
            }
            .hero h1 {
                font-size: 1.9rem;
            }
            .stock-header {
                align-items: flex-start;
                flex-direction: column;
            }
            .stock-name {
                font-size: 1.8rem;
            }
            .case-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
