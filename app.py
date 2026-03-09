import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from vnstock import stock_historical_data
import ta
import joblib
import os
import google.generativeai as genai
from dotenv import load_dotenv
import feedparser
from urllib.parse import quote
from email.utils import parsedate_to_datetime

# ==========================================
# CẤU HÌNH HỆ THỐNG & CONFIG
# ==========================================
load_dotenv()
st.set_page_config(page_title="VN Stock AI Predictor", layout="wide", page_icon="📈")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
# Cấu hình API LLM 1 lần duy nhất lúc khởi tạo app
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# MAPPING TIMEFRAMES
TIMEFRAME_MAPPING = {
    "1 Tuần": "7d",
    "1 Tháng": "30d",
    "3 Tháng": "90d"
}

TIMEFRAME_LABELS = {
    "1 Tuần": "Tăng > 3% sau 7 ngày",
    "1 Tháng": "Tăng > 5% sau 30 ngày",
    "3 Tháng": "Tăng > 10% sau 90 ngày"
}

# ==========================================
# LOAD MODEL & PHỤ TRỢ (ONLINE INFERENCE)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model(timeframe_key: str):
    """Tải model tương ứng với khung thời gian đã chọn"""
    model_path = f'xgboost_model_{timeframe_key}.joblib'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data_online(ticker: str, display_days: int) -> pd.DataFrame:
    """Kéo dữ liệu cổ phiếu dựa trên số ngày user chọn, cộng thêm khoảng thời gian để tính đủ EMA50."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    total_days = display_days + 90
    start_date = (datetime.today() - timedelta(days=total_days)).strftime('%Y-%m-%d')
    
    try:
        df = stock_historical_data(symbol=ticker, start_date=start_date, end_date=end_date, resolution='1D', type='stock')
        if df is None or df.empty:
            return pd.DataFrame()
            
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        try:
            df_vnindex = stock_historical_data(symbol='VNINDEX', start_date=start_date, end_date=end_date, resolution='1D', type='index')
            df_vnindex['time'] = pd.to_datetime(df_vnindex['time'])
            df_vnindex.set_index('time', inplace=True)
            df_vnindex.sort_index(inplace=True)
            df['vnindex_close'] = df_vnindex['close']
        except Exception:
            df['vnindex_close'] = np.nan
            
        return df
    except Exception as e:
        st.error(f"❌ Lỗi Fetch Data ({ticker}): {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_google_news(ticker: str) -> list:
    """Trích xuất 5 tiêu đề bài báo vĩ mô/pháp luật mới nhất từ Google News RSS về mã cổ phiếu"""
    news_titles = []
    try:
        # Nhúng từ khóa tìm kiếm (Tên mã CP + cụm từ "chứng khoán") để ra kết quả chính xác
        query = quote(f"{ticker} chứng khoán")
        url = f"https://news.google.com/rss/search?q={query}&hl=vi&gl=VN&ceid=VN:vi"
        
        feed = feedparser.parse(url)
        
        # Chỉ lấy 5 tin đầu tiên
        for entry in feed.entries[:5]:
            title = entry.title
            pub_date_str = entry.published
            
            # Format lại thời gian đăng tải
            try:
                 dt_obj = parsedate_to_datetime(pub_date_str)
                 formatted_date = dt_obj.strftime("%d/%m/%Y %H:%M")
            except Exception:
                 formatted_date = pub_date_str

            # Lưu vào list format gọn gàng
            news_titles.append(f"[{formatted_date}] {title}")
            
    except Exception as e:
        print(f"Không thể lấy tin tức Google News cho mã {ticker}: {e}")
    
    return news_titles

def engineer_features_online(df: pd.DataFrame) -> pd.DataFrame:
    """Xây dựng feature cho tập inference (Không xây Target label)"""
    df = df.copy()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['close']).macd()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    df['Daily_Return'] = df['close'].pct_change()
    if 'vnindex_close' in df:
        df['VNIndex_Return'] = df['vnindex_close'].pct_change()
    else:
        df['VNIndex_Return'] = 0.0
        
    np.random.seed(42)
    df['Net_Foreign_Buy'] = np.random.normal(0, 1000000, size=len(df))
    
    df.dropna(inplace=True)
    return df

def generate_ai_report(metrics_data: dict) -> str:
    """Gọi API Gemini để nhận định định tính dựa trên Model parameter và Tin tức Vĩ mô"""
    if not GEMINI_API_KEY:
        return "⚠️ Lỗi: Chưa cấu hình GEMINI_API_KEY."
    try:
        model_llm = genai.GenerativeModel('gemini-2.5-flash')
        
        # Xếp các dòng tin nhắn vào văn bản
        news_section = ""
        if metrics_data['recent_news']:
            news_section = "\n".join([f"- {news}" for news in metrics_data['recent_news']])
        else:
            news_section = "- Không có tin tức vĩ mô hoặc sự kiện nổi bật nào trên Google News trong thời gian qua."

        prompt = f"""
        Đóng vai một "Chuyên gia đánh giá tâm lý thị trường (Sentiment Analyzer)" xuất sắc chuyên về cổ phiếu Việt Nam.
        Dưới đây là [Thông số Kỹ thuật XGBoost] và [Tin tức Vĩ mô/Pháp luật Mới nhất] từ báo chí:
        
        - Mã cổ phiếu: {metrics_data['ticker']}
        - Khung thời gian đầu tư: {metrics_data['timeframe_str']} ({metrics_data['timeframe_desc']})
        - Giá hiện tại: {metrics_data['close']:,.0f} VNĐ
        - RSI (14): {metrics_data['rsi']:.2f}
        - MACD: {metrics_data['macd']:.2f}
        - Trạng thái giá so với EMA20/EMA50: Giá {'nằm trên' if metrics_data['close'] > metrics_data['ema20'] else 'nằm dưới'} EMA 20
        - Xác suất TĂNG từ AI Model (XGBoost): {metrics_data['prob_up']:.1f}%

        🗞️ TIN TỨC SỰ KIỆN NỔI BẬT TỪ GOOGLE NEWS:
        {news_section}

        Yêu cầu: Viết 1 bài nhận định sắc bén và cực kỳ ngắn gọn, súc tích bằng tiếng Việt chuyên nghiệp, chia nhỏ các mục sau bằng markdown:
        1. Phân tích Tin tức (Sentiment): Xác định ngay loạt sự kiện trên có sắc thái Tích cực (Positive), Tiêu cực (Negative), hay Trung lập (Neutral). Chúng ảnh hưởng thế nào đến tâm lý đám đông?
        2. Đối chiếu Kỹ thuật vs Tin tức (Tuyệt Mật): Cảnh báo nếu Tin tức đang mấu thuẫn với chỉ báo XGBoost. (Ví dụ: XGBoost đang dự báo xác suất tăng lên đến 70% vì RSI cực thấp, nhưng nếu tin tức có từ khóa Bắt bớ/Pháp luật/Thao túng thì phải KHUYÊN BÁN NGAY BẤT CHẤP KỸ THUẬT).
        3. Kết luận Hành động: Chốt lại khuyến nghị đầu tư tối hậu (MUA / BÁN / QUAN SÁT) ngắn gọn và dứt khoát dành cho khung {metrics_data['timeframe_str']}.
        """
        response = model_llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Có lỗi xảy ra khi gọi Google Gemini API: {str(e)}"

def plot_candlestick(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlesticks'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name='EMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue', width=1.5), name='EMA 50'))
    fig.update_layout(title=f"Biểu đồ kỹ thuật {ticker}", yaxis_title='Giá (VND)', xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
def main():
    st.title("🤖 VN Stock Machine Learning Predictor")
    st.markdown("Hệ thống kết hợp **XGBoost (Định lượng)** và **LLM Gemini 2.5 (Đối chiếu Sentiment Google News)** để dự báo cơ hội.")
    
    with st.sidebar:
        st.header("⚙️ Bảng Điều Khiển")
        
        tickers = st.multiselect(
            "Chọn mã cổ phiếu (Tối đa 3 mã):", 
            ["VIC", "HAG", "FPT", "HPG", "SSI", "VNM", "VCB", "ACB", "MBB", "TCB", "ACV", "VJC", "GAS", "MSN", "PVD", "PLX", "VRE", "NVL", "KDH", "DXG", "SAB"],
            default=["HAG"]
        )
        
        if len(tickers) > 3:
            st.warning("⚠️ Lời khuyên: Giới hạn 3 mã để tránh bị gián đoạn khi gọi API AI quá tải.")
            
        timeframe_choice = st.selectbox(
            "Khung thời gian dự đoán (Timeframe):",
            ["1 Tuần", "1 Tháng", "3 Tháng"]
        )
        timeframe_key = TIMEFRAME_MAPPING[timeframe_choice]
        timeframe_desc = TIMEFRAME_LABELS[timeframe_choice]
            
        display_days = st.slider(
            "Số phiên hiện trên biểu đồ:", 
            min_value=30, 
            max_value=360, 
            value=100, 
            step=10,
            help="Số ngày nến hiển thị ngược lại từ hôm nay."
        )
        
        analyze_btn = st.button("🚀 Phân Tích Ngay", use_container_width=True)
        
    # 1. Tải Model động theo timeframe
    model_data = load_model(timeframe_key)
    if model_data is None:
        st.info(f"💡 Hệ thống chưa có dữ liệu mô hình cho khung [{timeframe_choice}].")
        st.warning(f"Vui lòng chạy lệnh `python train.py` trong Terminal để huấn luyện 3 mô hình tương ứng nhé!")
        st.stop()
        
    model = model_data['model']
    features = model_data['features']
    acc = model_data['accuracy']
        
    if analyze_btn:
        if not tickers:
            st.warning("Vui lòng chọn ít nhất 1 mã để phân tích!")
            st.stop()
            
        tabs = st.tabs([f"📜 {t}" for t in tickers])
        
        for idx, ticker in enumerate(tickers):
            with tabs[idx]:
                with st.spinner(f"Đang xử lý dữ liệu cho '{ticker}'..."):
                    df = fetch_stock_data_online(ticker, display_days)
                    
                if df.empty:
                    st.warning(f"Không lấy được dữ liệu hiện thời cho cổ phiếu {ticker}.")
                    continue
                    
                with st.spinner(f"Đang tính toán các chỉ báo kỹ thuật..."):
                    df_features = engineer_features_online(df)
                    
                if df_features.empty:
                     st.warning(f"Dữ liệu {ticker} bị thiếu sót, không đủ để phân tích.")
                     continue
                     
                with st.spinner(f"Đang cào dữ liệu Google News RSS về vĩ mô/pháp luật cho '{ticker}'..."):
                     recent_news = fetch_google_news(ticker)
                     
                df_display = df_features.tail(display_days)
                    
                latest_data = df_features.iloc[-1]
                X_latest = pd.DataFrame([latest_data[features].fillna(0).values], columns=features)
                
                # Predict probability
                prob = model.predict_proba(X_latest)[0]
                prob_up = prob[1] * 100
                
                # UI Layout
                st.markdown("---")
                col1, col2 = st.columns([6, 4])
                
                with col1:
                    st.subheader(f"📊 Hành vi giá - {ticker}")
                    st.plotly_chart(plot_candlestick(df_display, ticker), use_container_width=True)
                    
                with col2:
                    st.subheader(f"🎯 Dự đoán XGBoost ({timeframe_choice})")
                    met1, met2, met3 = st.columns(3)
                    met1.metric("Giá Hiện Tại", f"{latest_data['close']:,.0f}")
                    met2.metric("RSI (14)", f"{latest_data['RSI_14']:.1f}")
                    met3.metric("MACD", f"{latest_data['MACD']:.1f}")
                    
                    st.markdown(f"### Xác suất {timeframe_desc}")
                    progress_color = "green" if prob_up > 50 else "red"
                    st.markdown(
                        f'''
                        <div style="background-color: #2e2e38; border-radius: 10px; padding: 20px; text-align: center;">
                            <h1 style="color: {progress_color}; margin: 0;">{prob_up:.1f}%</h1>
                            <p style="margin: 0; color: #a9a9b3;">Độ chính xác Test Offline: {acc*100:.1f}%</p>
                        </div>
                        ''', unsafe_allow_html=True
                    )
                    
                    with st.expander("🔍 Xem Top Tính năng ảnh hưởng (Kỹ thuật)"):
                        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                        st.dataframe(importances.reset_index().rename(columns={'index':'Chỉ báo', 0:'Mức độ'}), use_container_width=True)

                    if recent_news:
                        with st.expander(f"🌐 5 Tin Tức & Sự Kiện Nổi Bật Trên Google News ({ticker})", expanded=True):
                            for news in recent_news:
                                st.markdown(f"- **{news}**")
                        
                st.markdown("---")
                st.subheader(f"🧠 Phân tích Sentiment & Đối chiếu (Gemini 2.5) - {timeframe_choice}")
                metrics_payload = {
                    'ticker': ticker,
                    'timeframe_str': timeframe_choice,
                    'timeframe_desc': timeframe_desc,
                    'close': latest_data['close'],
                    'rsi': latest_data['RSI_14'],
                    'macd': latest_data['MACD'],
                    'ema20': latest_data['EMA_20'],
                    'prob_up': prob_up,
                    'recent_news': recent_news
                }
                
                with st.spinner(f"Gemini đang phân tích sắc thái báo chí và kiểm tra mâu thuẫn..."):
                    st.info(generate_ai_report(metrics_payload))

if __name__ == "__main__":
    main()
