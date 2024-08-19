import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
import yfinance as yf

# 페이지 설정
st.set_page_config(layout="wide", page_title="주가 예측 서비스")

# 메뉴 옵션
menu_options = [
    {"icon": "house", "label": "홈"},
    {"icon": "graph-up", "label": "모델 소개"},
    {"icon": "database", "label": "데이터셋 소개"},
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
    ## 딥러닝 기반 주식 가격 예측 플랫폼
    
    딥러닝 기술을 활용하여 주식 시장의 동향을 분석하고 예측합니다. 
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
    GRU(Gated Recurrent Unit)는 순환 신경망(RNN)의 변형으로, 
    시계열 데이터 처리에 탁월한 성능을 보입니다.
    특히 장기 의존성 문제를 효과적으로 해결하여 
    복잡한 시퀀스 데이터 분석에 적합합니다.
    """)
    
    # GRU 구조 설명
    st.header("GRU의 구조")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("""
        GRU 셀의 주요 구성 요소:
        1. **Update Gate**: 새로운 정보 반영 정도 결정
        2. **Reset Gate**: 과거 정보 무시 정도 결정
        3. **Hidden State**: 현재 시점의 정보 저장
        """)
    
    with col2:
        # GRU 셀 구조 다이어그램 (SVG)
        gru_cell_svg = """
        <svg width="100%" height="250" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="95%" height="230" fill="lightblue" opacity="0.3" stroke="royalblue" stroke-width="2"/>
            <circle cx="50%" cy="50%" r="80" fill="lavender" stroke="purple" stroke-width="2"/>
            <text x="50%" y="50%" text-anchor="middle" fill="purple" font-size="16">Hidden State</text>
            <rect x="20" y="20" width="120" height="50" fill="lightgreen" stroke="green" stroke-width="2"/>
            <text x="80" y="50" text-anchor="middle" fill="green" font-size="14">Update Gate</text>
            <rect x="20" y="180" width="120" height="50" fill="lightpink" stroke="red" stroke-width="2"/>
            <text x="80" y="210" text-anchor="middle" fill="red" font-size="14">Reset Gate</text>
            <line x1="140" y1="45" x2="200" y2="90" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
            <line x1="140" y1="205" x2="200" y2="160" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
            </defs>
        </svg>
        """
        st.components.v1.html(gru_cell_svg, height=250)
    
    # 모델 비교 (RMSE 값만 사용)
    st.header("모델 성능 비교")
    st.write("""
    다양한 시계열 예측 모델의 Test RMSE를 비교해 보겠습니다. 
    RMSE(Root Mean Square Error)는 예측값과 실제값의 차이를 나타내는 지표로, 
    낮을수록 예측 정확도가 높음을 의미합니다.
    """)
    
    comparison_data = pd.DataFrame({
        'Model': ['GRU', 'LSTM', 'Transformer', 'XGB'],
        'Test RMSE': [0.0144, 0.022, 0.0572, 0.0674]
    })
    
    fig = px.bar(comparison_data, x='Model', y='Test RMSE', title='D1 Dataset: Test RMSE Comparison')
    fig.update_layout(
        xaxis_title='Model', 
        yaxis_title='Test RMSE',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor='lightgrey'
    )
    fig.update_traces(marker_color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("""
    위 그래프에서 볼 수 있듯이, GRU 모델은 다른 모델들과 비교했을 때 가장 우수한 성능을 보입니다.

    - GRU의 Test RMSE는 0.0144로, 두 번째로 좋은 성능을 보인 LSTM(0.022)보다도 약 34.5% 낮은 오차를 보여줍니다.
    - 이는 GRU가 주가 예측 태스크에서 더 정확한 예측을 할 수 있음을 의미합니다.
    - Transformer(0.0572)와 XGB(0.0674) 모델은 GRU에 비해 현저히 높은 RMSE를 보이고 있습니다.
    - 이는 이 특정 데이터셋과 태스크에 대해 GRU가 더 적합한 모델임을 시사합니다.
    """)

    # GRU의 주가 예측 적용
    st.header("GRU의 주가 예측 적용")
    st.write("""
    GRU 모델이 주가 예측에 특히 효과적인 이유:
    """)
    
    reasons = [
        "**장기 의존성 포착**: 과거의 중요한 정보를 오랫동안 기억할 수 있어, 주가의 장기적 트렌드를 잘 파악합니다.",
        "**노이즈 필터링**: Update Gate와 Reset Gate를 통해 불필요한 정보를 걸러내어, 주가의 일시적 변동에 덜 민감합니다.",
        "**비선형성 모델링**: 복잡한 주가 패턴을 효과적으로 학습할 수 있어, 시장의 다양한 상황에 대응 가능합니다.",
        "**계산 효율성**: LSTM보다 단순한 구조로 비슷한 성능을 내기 때문에, 실시간 예측이나 빠른 모델 업데이트에 유리합니다."
    ]
    
    for reason in reasons:
        st.markdown(f"- {reason}")

    st.write("""
    이러한 특성들이 결합되어 GRU가 다른 모델들보다 더 낮은 RMSE를 달성할 수 있었으며, 
    이는 주가 예측에 있어 GRU의 우수성을 입증합니다.
    """)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

def dataset_explanation():
    st.title("데이터셋 설명")
    st.write("""
    주가 예측을 위해 세 가지 주요 데이터셋을 활용합니다. 각 데이터셋은 서로 다른 측면의 정보를 제공하여 
    모델의 예측 성능을 향상시킵니다.
    """)
    
    # 실제 데이터 가져오기
    stock = yf.Ticker("AAPL")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # 최근 10년
    data = stock.history(start=start_date, end=end_date)
    
    # D1 데이터셋 설명
    st.header("D1: 종가 + 기술지표")
    st.write("""
    D1 데이터셋은 주식의 종가와 다양한 기술적 지표를 포함합니다. 이 데이터셋은 주로 주가의 과거 패턴과 
    추세를 분석하는 데 사용됩니다.
    """)
    
    st.subheader("주요 특성")
    d1_features = ["종가", "거래량", "RSI", "MACD", "볼린저 밴드"]
    for feature in d1_features:
        st.markdown(f"- **{feature}**")
    
    # D1 시각화
    st.subheader("D1 데이터셋 시각화")
    
    # RSI 계산
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 계산
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=("종가", "거래량", "RSI", "MACD"))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="종가"), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="거래량"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI"), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], name="Signal Line"), row=4, col=1)
    
    fig.update_layout(height=1000)  # 그래프 높이 증가
    st.plotly_chart(fig, use_container_width=True)
    
    # D2 데이터셋 설명
    st.header("D2: 종가 + 외부요인")
    st.write("""
    D2 데이터셋은 주식의 종가와 함께 다양한 외부 경제 지표를 포함합니다. 이 데이터셋은 주가에 영향을 
    미치는 거시경제적 요인을 고려합니다.
    """)
    
    st.subheader("주요 특성")
    d2_features = ["금리", "환율", "장단기 금리차"]
    for feature in d2_features:
        st.markdown(f"- **{feature}**")
    
    # D2 시각화
    st.subheader("D2 데이터셋 시각화")
    
    # 실제 데이터 가져오기
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # 최근 10년
    
    # 금리 데이터 (10년물 국채 수익률)
    treasury_10y = yf.Ticker("^TNX").history(start=start_date, end=end_date)['Close']
    
    # 환율 데이터 (USD/KRW)
    exchange_rate = yf.Ticker("KRW=X").history(start=start_date, end=end_date)['Close']
    
    # 장단기 금리차 데이터 (10년물 - 3개월물)
    treasury_3m = yf.Ticker("^IRX").history(start=start_date, end=end_date)['Close']
    yield_spread = treasury_10y - treasury_3m
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("금리 (10년물 국채 수익률)", "환율 (USD/KRW)", "장단기 금리차 (10년 - 3개월)"))
    
    fig.add_trace(go.Scatter(x=treasury_10y.index, y=treasury_10y, name="금리"), row=1, col=1)
    fig.add_trace(go.Scatter(x=exchange_rate.index, y=exchange_rate, name="환율"), row=2, col=1)
    fig.add_trace(go.Scatter(x=yield_spread.index, y=yield_spread, name="장단기 금리차"), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=False)
    fig.update_yaxes(title_text="금리 (%)", row=1, col=1)
    fig.update_yaxes(title_text="환율 (원/달러)", row=2, col=1)
    fig.update_yaxes(title_text="금리차 (%p)", row=3, col=1)
    fig.update_xaxes(title_text="날짜", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    # D3 데이터셋 설명
    st.header("D3: 종가 + 기술지표 + 외부요인")
    st.write("""
    D3 데이터셋은 D1과 D2의 모든 특성을 결합한 가장 포괄적인 데이터셋입니다. 이 데이터셋은 기술적 분석과 
    기본적 분석을 모두 고려하여 가장 정확한 예측을 제공할 수 있습니다.
    """)
    
    st.subheader("주요 특성")
    d3_features = list(set(d1_features + d2_features))  # 중복 제거
    for feature in d3_features:
        st.markdown(f"- **{feature}**")
    
    # 데이터 분할 시각화
    st.header("데이터 분할 (Train/Validation/Test)")
    
    total_samples = len(data)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], name='Train', mode='lines'))
    fig.add_trace(go.Scatter(x=val_data.index, y=val_data['Close'], name='Validation', mode='lines'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], name='Test', mode='lines'))
    
    fig.update_layout(
        title='주가 데이터 분할 (6:2:2)',
        xaxis_title='날짜',
        yaxis_title='종가',
        legend_title='데이터셋'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    

import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

from collections import OrderedDict

def prediction():
    st.title("주가 예측")
    
    # OrderedDict를 사용하여 순서를 유지
    sectors = OrderedDict([
        ("IT", ["Apple Inc. (AAPL)"]),
        ("필수 소비재", ["The Coca-Cola Co. (KO)"]),
        ("헬스케어", ["Johnson & Johnson (JNJ)"]),
        ("금융", ["JPMorgan Chase & Co. (JPM)"])
    ])

    # 각 주식의 월요일 주가를 설정하는 딕셔너리
    monday_prices = {
        "AAPL": 180.00,  # Apple Inc.의 월요일 주가
        "KO": 60.00,  # The Coca-Cola Co.의 월요일 주가
        "JNJ": 165.00,  # Johnson & Johnson의 월요일 주가
        "JPM": 150.00  # JPMorgan Chase & Co.의 월요일 주가
    }

    sector = st.selectbox("섹터 선택", list(sectors.keys()))
    stock = st.selectbox("종목 선택", sectors[sector])

    if stock:
        st.header(f"{stock} 주가 예측")
        
        ticker = stock.split('(')[1].split(')')[0]
        
        # 최근 30일간의 주가 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        current_price = stock_data['Close'].iloc[-1]
        
        predictions_data = {
            "AAPL": [214.18, 209.28, 207.48, 206.65, 206.26],
            "KO": [68.63, 66.72, 65.40, 64.65, 64.24],
            "JNJ": [160.30, 158.46, 156.09, 154.50, 153.50],
            "JPM": [211.91, 212.11, 211.99, 211.93, 211.92]
        }
        
        prediction_dates = ["월", "화", "수", "목", "금"]
        predictions = predictions_data[ticker]
        
        # 해당 주식의 월요일 주가 가져오기
        monday_price = monday_prices[ticker]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("현재 주가 정보")
            st.metric("현재 가격", f"${current_price:.2f}")
            
            # 최근 30일 주가 그래프
            fig_recent = go.Figure()
            fig_recent.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='실제 주가',
                                            line=dict(color='royalblue', width=2)))
            fig_recent.update_layout(
                title="최근 30일 주가 추이",
                xaxis_title="날짜",
                yaxis_title="주가 ($)",
                hovermode="x unified",
                template="plotly_white"
            )
            fig_recent.update_xaxes(
                rangebreaks=[dict(bounds=["sat", "mon"])],  # 주말 제외
                showgrid=True, gridwidth=1, gridcolor='lightgrey'
            )
            fig_recent.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
            st.plotly_chart(fig_recent, use_container_width=True)
        
        with col2:
            st.subheader("주가 예측 결과")
            next_day_prediction = predictions[0]
            delta = next_day_prediction - current_price
            st.metric("다음 거래일 예상 가격", f"${next_day_prediction:.2f}", delta=f"{delta:.2f}")
            
            if delta > 0:
                st.success("주가가 상승할 것으로 예상됩니다.")
            else:
                st.error("주가가 하락할 것으로 예상됩니다.")
        
        # 주가 예측 그래프
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, name='예측 주가',
                                 line=dict(color='firebrick', width=2)))
        fig.add_trace(go.Scatter(x=prediction_dates, y=[monday_price] + [None]*4, name='실제 주가 (월요일)',
                                 mode='markers', marker=dict(color='royalblue', size=10)))
        fig.update_layout(
            title="주가 예측 vs 실제 주가 (월-금)",
            xaxis_title="요일",
            yaxis_title="주가 ($)",
            hovermode="x unified",
            template="plotly_white"
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        st.plotly_chart(fig, use_container_width=True)
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
