import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(layout="wide", page_title="주가 예측 서비스")

# 메뉴 옵션
menu_options = [
    {"icon": "house", "label": "홈"},
    {"icon": "graph-up", "label": "모델 설명"},
    {"icon": "database", "label": "데이터셋 설명"},
    {"icon": "lightning", "label": "예측"}
]

# 상단 메뉴 구성
selected = option_menu(
    menu_title=None,
    options=[option["label"] for option in menu_options],
    icons=[option["icon"] for option in menu_options],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "lightblue"},
    }
)

# 섹터 및 종목 데이터 (기존 코드와 동일)
sectors = {
    "금융": ["JPMorgan Chase & Co. (JPM)"],
    "IT": ["Apple Inc. (AAPL)"],
    "필수 소비재": ["The Coca-Cola Co. (KO)"],
    "헬스케어": ["Johnson & Johnson (JNJ)"]
}

# 홈 페이지
def home():
    st.title("주가 예측 서비스")
    st.write("""
    ## 인공지능 기반 주식 시장 분석 및 예측 플랫폼
    
    최신 머신러닝 기술을 활용하여 주식 시장의 동향을 분석하고 예측합니다. 
    다양한 섹터와 종목에 대한 심층 분석을 제공하며, 고급 GRU 모델을 사용하여 높은 정확도의 예측을 제공합니다.
    """)

    st.markdown("""
    ### 🌟 주요 기능
    - 주가 데이터 분석
    - 섹터별, 종목별 분석
    - GRU 모델 기반 정확한 주가 예측
    - 사용자 친화적 인터페이스
    """)

    # 최근 시장 동향 차트
    st.subheader("최근 시장 동향")
    market_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'S&P 500': np.cumsum(np.random.randn(365) * 0.1) + 100,
        'NASDAQ': np.cumsum(np.random.randn(365) * 0.2) + 100,
        'DOW JONES': np.cumsum(np.random.randn(365) * 0.15) + 100
    })
    fig = px.line(market_data, x='Date', y=['S&P 500', 'NASDAQ', 'DOW JONES'], title='주요 지수 동향')
    st.plotly_chart(fig, use_container_width=True)

def model_explanation():
    st.title("GRU 모델 설명")
    
    st.write("""
    GRU(Gated Recurrent Unit)는 순환 신경망(RNN)의 변형으로, 시계열 데이터 처리에 탁월한 성능을 보입니다.
    특히 장기 의존성 문제를 효과적으로 해결하여 복잡한 시퀀스 데이터 분석에 적합합니다.
    """)
    
    # GRU 구조 설명
    st.header("GRU의 구조")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        GRU 셀의 주요 구성 요소:
        1. **Update Gate**: 새로운 정보 반영 정도 결정
        2. **Reset Gate**: 과거 정보 무시 정도 결정
        3. **Hidden State**: 현재 시점의 정보 저장
        """)
    
    with col2:
        # GRU 셀 구조 다이어그램 (SVG로 변경)
        gru_cell_svg = """
        <svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="280" height="180" fill="lightblue" opacity="0.3" stroke="royalblue" stroke-width="2"/>
            <circle cx="150" cy="100" r="60" fill="lavender" stroke="purple" stroke-width="2"/>
            <text x="150" y="105" text-anchor="middle" fill="purple">Hidden State</text>
            <rect x="20" y="20" width="80" height="40" fill="lightgreen" stroke="green" stroke-width="2"/>
            <text x="60" y="45" text-anchor="middle" fill="green">Update Gate</text>
            <rect x="20" y="140" width="80" height="40" fill="lightpink" stroke="red" stroke-width="2"/>
            <text x="60" y="165" text-anchor="middle" fill="red">Reset Gate</text>
            <line x1="100" y1="40" x2="130" y2="70" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
            <line x1="100" y1="160" x2="130" y2="130" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs>
        </svg>
        """
        st.components.v1.html(gru_cell_svg, width=300, height=200)
    
    # GRU의 작동 원리
    st.subheader("GRU의 작동 원리")
    st.write("""
    GRU는 각 시점에서 다음과 같은 과정을 거칩니다:
    1. Update Gate와 Reset Gate 계산
    2. 후보 Hidden State 생성
    3. 최종 Hidden State 갱신
    """)
    
    # Gate 설명
    st.subheader("Gate의 역할")
    gate_explanation = """
    <style>
    .gate-box {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .gate-title {
        font-weight: bold;
        color: #333;
    }
    </style>
    <div class="gate-box">
        <p class="gate-title">Update Gate</p>
        <p>새로운 정보를 얼마나 반영할지 결정합니다. 값이 1에 가까울수록 새 정보를 많이 반영하고, 0에 가까울수록 이전 정보를 유지합니다.</p>
    </div>
    <div class="gate-box">
        <p class="gate-title">Reset Gate</p>
        <p>과거의 정보를 얼마나 무시할지 결정합니다. 값이 0에 가까울수록 과거 정보를 많이 무시하고, 1에 가까울수록 과거 정보를 유지합니다.</p>
    </div>
    """
    st.markdown(gate_explanation, unsafe_allow_html=True)
    
    # 모델 비교
    st.header("모델 성능 비교")
    st.write("다양한 시계열 예측 모델의 성능을 비교해 보겠습니다.")
    
    comparison_data = pd.DataFrame({
        'Model': ['GRU', 'LSTM', 'Transformer', 'XGBoost'],
        'MSE': [0.015, 0.018, 0.012, 0.020],
        'MAE': [0.095, 0.105, 0.090, 0.110],
        'R2 Score': [0.92, 0.90, 0.94, 0.88]
    })
    
    fig = go.Figure()
    for metric in ['MSE', 'MAE', 'R2 Score']:
        fig.add_trace(go.Bar(
            x=comparison_data['Model'],
            y=comparison_data[metric],
            name=metric
        ))
    
    fig.update_layout(
        title='모델 성능 비교',
        xaxis_title='모델',
        yaxis_title='성능 지표',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    위 그래프에서 볼 수 있듯이, GRU 모델은 다른 모델들과 비교했을 때 좋은 성능을 보입니다.
    특히 LSTM과 비슷한 성능을 보이면서도 더 단순한 구조를 가져 학습 속도가 빠르다는 장점이 있습니다.
    Transformer 모델이 일부 지표에서 더 좋은 성능을 보이지만, 계산 복잡도가 높아 실시간 예측에는 GRU가 더 적합할 수 있습니다.
    """)

    # GRU의 주가 예측 적용
    st.header("GRU의 주가 예측 적용")
    st.write("""
    GRU 모델이 주가 예측에 효과적인 이유:
    1. 장기 의존성 포착: 과거의 중요한 정보를 오랫동안 기억할 수 있습니다.
    2. 노이즈 필터링: Update Gate와 Reset Gate를 통해 불필요한 정보를 걸러냅니다.
    3. 비선형성 모델링: 복잡한 주가 패턴을 효과적으로 학습할 수 있습니다.
    """)
    

def dataset_explanation():
    st.title("데이터셋 설명")
    st.write("""
    주가 예측을 위해 세 가지 주요 데이터셋을 활용합니다. 각 데이터셋은 서로 다른 측면의 정보를 제공하여 
    모델의 예측 성능을 향상시킵니다.
    """)
    
    # D1 데이터셋 설명
    st.header("D1: 종가 + 기술지표")
    st.write("""
    D1 데이터셋은 주식의 종가와 다양한 기술적 지표를 포함합니다. 이 데이터셋은 주로 주가의 과거 패턴과 
    추세를 분석하는 데 사용됩니다.
    """)
    
    st.subheader("주요 특성")
    d1_features = ["종가", "거래량", "RSI", "MACD", "볼린저 밴드"]
    for feature in d1_features:
        st.write(f"- {feature}")
    
    # D1 시각화
    st.subheader("D1 데이터셋 시각화")
    
    # 가상의 데이터 생성
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    close = np.cumsum(np.random.randn(len(dates))) + 100
    volume = np.random.randint(1000000, 10000000, size=len(dates))
    rsi = np.random.randint(0, 100, size=len(dates))
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    fig.add_trace(go.Scatter(x=dates, y=close, name="종가"), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=volume, name="거래량"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rsi, name="RSI"), row=3, col=1)
    
    fig.update_layout(height=600, title_text="D1 데이터셋 시각화")
    st.plotly_chart(fig, use_container_width=True)
    
    # D2 데이터셋 설명
    st.header("D2: 종가 + 외부요인")
    st.write("""
    D2 데이터셋은 주식의 종가와 함께 다양한 외부 경제 지표를 포함합니다. 이 데이터셋은 주가에 영향을 
    미치는 거시경제적 요인을 고려합니다.
    """)
    
    st.subheader("주요 특성")
    d2_features = ["종가", "금리", "환율", "S&P 500 지수", "원유 가격"]
    for feature in d2_features:
        st.write(f"- {feature}")
    
    # D2 시각화
    st.subheader("D2 데이터셋 시각화")
    
    # 가상의 데이터 생성
    interest_rate = np.random.uniform(1, 5, size=len(dates))
    exchange_rate = np.random.uniform(1000, 1200, size=len(dates))
    sp500 = np.cumsum(np.random.randn(len(dates))) + 3000
    
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.1, horizontal_spacing=0.05)
    
    fig.add_trace(go.Scatter(x=dates, y=close, name="종가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=interest_rate, name="금리"), row=1, col=2)
    fig.add_trace(go.Scatter(x=dates, y=exchange_rate, name="환율"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sp500, name="S&P 500"), row=2, col=2)
    
    fig.update_layout(height=600, title_text="D2 데이터셋 시각화")
    st.plotly_chart(fig, use_container_width=True)
    
    # D3 데이터셋 설명
    st.header("D3: 종가 + 기술지표 + 외부요인")
    st.write("""
    D3 데이터셋은 D1과 D2의 모든 특성을 결합한 가장 포괄적인 데이터셋입니다. 이 데이터셋은 기술적 분석과 
    기본적 분석을 모두 고려하여 가장 정확한 예측을 제공할 수 있습니다.
    """)
    
    st.subheader("주요 특성")
    d3_features = d1_features + d2_features
    d3_features = list(set(d3_features))  # 중복 제거
    for feature in d3_features:
        st.write(f"- {feature}")
    
    # D3 시각화 (3D 산점도)
    st.subheader("D3 데이터셋 3D 시각화")
    
    fig = go.Figure(data=[go.Scatter3d(
        x=close,
        y=rsi,
        z=sp500,
        mode='markers',
        marker=dict(
            size=5,
            color=close,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene = dict(
            xaxis_title='종가',
            yaxis_title='RSI',
            zaxis_title='S&P 500'
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10),
        title_text="D3 데이터셋: 종가, RSI, S&P 500 관계"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 데이터 전처리 과정
    st.header("데이터 전처리 과정")
    preprocessing_steps = [
        "1. 데이터 수집 및 통합", 
        "2. 결측치 처리",
        "3. 이상치 제거",
        "4. 특성 스케일링 (Min-Max 정규화)",
        "5. 시계열 데이터 분할 (훈련/검증/테스트 세트)"
    ]
    
    for step in preprocessing_steps:
        st.write(step)
    
    # 결론
    st.header("결론")
    st.write("""
    세 가지 데이터셋(D1, D2, D3)을 활용함으로써, 주가 예측 모델의 정확도와 신뢰성을 향상시킬 수 있습니다. 
    D3 데이터셋은 기술적 지표와 외부 요인을 모두 포함하고 있어, 가장 포괄적인 정보를 제공합니다.
    그러나 데이터의 품질과 적절한 전처리가 모델 성능에 큰 영향을 미치므로, 
    데이터 준비 단계에 충분한 시간과 노력을 투자해야 합니다.
    각 데이터셋의 특성을 잘 이해하고 활용하면, 더 정확한 주가 예측이 가능할 것입니다.
    """)

# 예측 페이지
def prediction():
    st.title("주가 예측")
    
    sector = st.selectbox("섹터 선택", list(sectors.keys()))
    stock = st.selectbox("종목 선택", sectors[sector])

    if stock:
        st.header(f"{stock} 주가 예측")
        
        # 실제 데이터 대신 임의의 데이터 사용
        today_price = np.random.randint(100, 200)
        next_week_prediction = today_price + np.random.randint(-10, 11)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("현재 주가 정보")
            st.metric("현재 가격", f"${today_price:,.2f}")
            
            # 주가 차트
            dates = pd.date_range(start="2023-01-01", end="2023-12-31")
            prices = np.cumsum(np.random.randn(len(dates))) * 5 + 100
            df = pd.DataFrame({"Date": dates, "Price": prices})
            
            fig = px.line(df, x="Date", y="Price", title="주가 추이")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("주가 예측 결과")
            delta = next_week_prediction - today_price
            st.metric("다음 주 예상 가격", f"${next_week_prediction:,.2f}", delta=f"{delta:+.2f}")
            
            if delta > 0:
                st.success("주가가 상승 추세를 보일 것으로 예상됩니다.")
            else:
                st.error("주가가 하락 추세를 보일 것으로 예상됩니다.")


# 메인 앱 로직
if selected == "홈":
    home()
elif selected == "모델 설명":
    model_explanation()
elif selected == "데이터셋 설명":
    dataset_explanation()
elif selected == "예측":
    prediction()

# 앱 정보 (footer로 이동)
footer_html = f"""
<style>
.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: black;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}}
</style>
<div class="footer">
    <p>이 앱은 데모용입니다. 실제 투자 결정에 사용하지 마세요. | Version 1.0 | Last updated: {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)