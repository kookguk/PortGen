import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

# -------------------------
# OpenAI 클라이언트(Streamlit Secrets에서 API Key 불러오기)
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# -------------------------
# Streamlit UI
# 대제목 & 소제목
st.markdown(
    """
    <h1 style="margin-bottom:0;">💹 PortGen 
        <span style="font-size:30px; font-weight:normal; color:gray;">
        나의 주식 포트폴리오 매니저
        </span>
    </h1>
    """,
    unsafe_allow_html=True
)
st.write(
    """
    투자한 국내/미국 주식 종목코드와 각 비중을 입력하면 Maximum Sharpe Ratio 기반 포트폴리오 최적화로 새로운 투자 전략을 제안합니다.  

    📌 Maximum Sharpe Ratio란, 경제학자 William Sharpe가 제안한 Sharpe Ratio를 최대화하는 투자 전략으로, 위험(변동성) 대비 수익률을 가장 효율적으로 높이는 방법론을 의미합니다.
    """
)

# -------------------------
# 사용자 입력란
st.sidebar.header("📝 내 주식 포트폴리오 입력하기")

user_stocks_input = st.sidebar.text_input(
    "종목코드 (최대 10개)",
    value="",
    placeholder="종목 코드를 입력해주세요"
)

user_weights_input = st.sidebar.text_input(
    "비중",
    value="", 
    placeholder="각 종목 별 투자 비중을 입력해주세요"
)

analyze_button = st.sidebar.button("🚀 포트폴리오 분석하기")

# 예시 입력 & 주의점 콜아웃
st.sidebar.info(
    """
    💡 **예시 입력**  
    - 종목코드: <code>MSFT<code>, AAPL, 005930.KQ
    - 비중: 0.3, 0.4, 0.3

    🚨 **주의점**  
    1) 국내 주식의 경우 기업명이 아닌 정확한 종목코드를 입력해주세요. 
    ex) 삼성전자 ❌, 005930.KQ ✅ 
    2) 국내 주식의 경우 종목 코드를 코스피는 '.KS', 코스닥은 '.KQ' 형식으로 입력해주세요. 
    ex) 005930 ❌, 005930.KQ ✅ 
    3) 비중은 반드시 소수점으로 입력해주세요. 
    ex) 30% ❌, 0.3 ✅ 
    4) 비중의 전체 합은 반드시 1이어야 합니다. 
    ex) 0.1, 0.3, 0.3 ❌, 0.4, 0.3, 0.3 ✅
    """
)

user_stocks = [s.strip().upper() for s in user_stocks_input.split(",") if s.strip()]

# -------------------------
# yfinance 종가 데이터 다운로드 함수
@st.cache_data
def load_data(tickers):
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today().replace(year=datetime.today().year - 5)).strftime("%Y-%m-%d")
    data = pd.DataFrame()
    failed_tickers = []

    for t in tickers:
        df = yf.download(t, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if not df.empty:
            data[t] = df["Adj Close"]
            time.sleep(1)
        else:
            failed_tickers.append(t)

    return data, failed_tickers

# -------------------------
if analyze_button:
     
    # 0. 입력 검증
    if not user_stocks_input or not user_weights_input:
        st.error("❌ 종목코드와 비중을 모두 입력해주세요.")
        st.stop()

    if len(user_stocks) > 10:
        st.error("❌ 종목 개수는 최대 10개까지 입력 가능합니다.")
        st.stop()

    try:
        user_weights = np.array([float(x) for x in user_weights_input.split(",")])
    except ValueError:
        st.error("❌ 비중은 반드시 소수점 형식으로 입력해야 합니다.")
        st.stop()

    if len(user_stocks) != len(user_weights):
        st.error("❌ 종목 개수와 비중 개수가 일치하지 않습니다.")
        st.stop()

    if len(user_stocks) != len(set(user_stocks)):
        st.error("❌ 동일한 종목코드를 두 번 이상 입력하셨습니다. 중복 없이 입력해주세요.")
        st.stop()

    if not np.isclose(np.sum(user_weights), 1.0):
        st.error("❌ 투자 비중의 합은 반드시 1이어야 합니다.")
        st.stop()

    # 1. 데이터 수집
    data, failed_tickers = load_data(user_stocks)
    if failed_tickers:
        st.error(f"❌ 잘못된 종목코드: {', '.join(failed_tickers)}")
        st.stop()
    if data.empty:
        st.error("❌ 데이터 다운로드 실패.")
        st.stop()

    # 2. 수익률 & 공분산행렬
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # 3. 성과 함수
    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_return, port_volatility

    def neg_sharpe(weights, risk_free_rate=0.0):
        r, vol = portfolio_performance(weights)
        return -(r - risk_free_rate) / vol

    # 4. 최적 Sharpe 비중 탐색
    num_assets = len(user_stocks)
    init_guess = num_assets * [1.0 / num_assets]
    bounds = tuple((0,1) for _ in range(num_assets))
    constraints = {"type":"eq", "fun": lambda w: np.sum(w) - 1}

    opt_result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_weights = np.array(opt_result.x)

    # 5. 기존 vs 최적 성과
    orig_return, orig_vol = portfolio_performance(user_weights)
    orig_sharpe = orig_return / orig_vol
    opt_return, opt_vol = portfolio_performance(opt_weights)
    opt_sharpe = opt_return / opt_vol

    # 6. 시각화 (Plotly 사용)
    # 기존 vs 최적 비중 비교(Bar plot)
    fig1 = go.Figure(data=[
        go.Bar(name="기존", x=user_stocks, y=user_weights),
        go.Bar(name="최적", x=user_stocks, y=opt_weights)
    ])
    fig1.update_layout(
        barmode='group',
        title="기존 vs 최적 포트폴리오 비중 비교",
        xaxis_title="종목",
        yaxis_title="비중"
    )

    # 기존 vs 최적 누적 수익률 비교(Line chart)
    orig_port_returns = returns @ user_weights
    opt_port_returns = returns @ opt_weights
    cum_returns = pd.DataFrame({
        "기존": (1 + orig_port_returns).cumprod(),
        "최적": (1 + opt_port_returns).cumprod()
        })

    fig2 = px.line(cum_returns, title="기존 vs 최적 누적 수익률 비교")
    fig2.update_layout(
        xaxis_title="날짜",
        yaxis_title="누적 수익률",
        legend_title_text=""
        )

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    # 7. AI 요약
    full_report = f"""
    [기존 포트폴리오]
    연평균 수익률: {orig_return:.2%}, 변동성: {orig_vol:.2%}, 샤프: {orig_sharpe:.2f}
    비중: {dict(zip(user_stocks, [f"{w:.2%}" for w in user_weights]))}

    [최적 포트폴리오]
    연평균 수익률: {opt_return:.2%}, 변동성: {opt_vol:.2%}, 샤프: {opt_sharpe:.2f}
    비중: {dict(zip(user_stocks, [f"{w:.2%}" for w in opt_weights]))}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"너는 금융 포트폴리오 결과를 요약하는 AI다."},
            {"role":"user","content":f"아래 분석결과를 간단히 요약해줘:\n{full_report}"}
        ]
    )
    summary = resp.choices[0].message.content.strip()

    orig_allocation = ", ".join([f"{s}: {w:.2%}" for s, w in zip(user_stocks, user_weights)])
    opt_allocation  = ", ".join([f"{s}: {w:.2%}" for s, w in zip(user_stocks, opt_weights)])

    fixed_part = f"""
📊 기존 포트폴리오 구성
{orig_allocation}

📊 최적 포트폴리오 구성
{opt_allocation}
    """

    final_summary = (
        "최근 5년간 성과를 바탕으로 Maximum Sharpe Ratio 기반 포트폴리오 최적화 결과입니다.\n\n"
        f"{summary}\n\n"
        "⚠️ 투자 결정은 본인의 몫이며, 본 결과는 참고용입니다."
    )

    st.subheader("🗒️ AI 요약 결과")
    st.code(fixed_part.strip(), language="markdown")
    st.write(final_summary)