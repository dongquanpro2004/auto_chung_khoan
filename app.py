import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vnstock import stock_historical_data
import ta
import joblib
import os
import time
import json
from google import genai
from dotenv import load_dotenv
import feedparser
from urllib.parse import quote
from email.utils import parsedate_to_datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
load_dotenv()

GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

TICKERS = [
    'VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB',        
    'SSI', 'VND', 'MBS', 'VCI', 'HCM',               
    'VHM', 'VIC', 'NVL', 'DXG', 'KBC', 'IDC',        
    'HPG', 'HSG', 'NKG',                             
    'MWG', 'MSN', 'PNJ', 'FRT',                      
    'GAS', 'PVS', 'PVD', 'POW', 'REE',               
    'FPT', 'CTR',                                    
    'HVN', 'VJC', 'GMD',                             
    'VHC', 'ANV',                                    
    'VGT', 'TCM'                                     
]
GROUP_1 = TICKERS[:19]

# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU
# ==========================================
def fetch_stock_data_online(ticker: str) -> pd.DataFrame:
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=150)).strftime('%Y-%m-%d')
    try:
        df = stock_historical_data(symbol=ticker, start_date=start_date, end_date=end_date, resolution='1D', type='stock')
        if df is None or df.empty: return pd.DataFrame()
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        try:
            df_vnindex = stock_historical_data(symbol='VNINDEX', start_date=start_date, end_date=end_date, resolution='1D', type='index')
            df_vnindex['time'] = pd.to_datetime(df_vnindex['time'])
            df_vnindex.set_index('time', inplace=True)
            df_vnindex.sort_index(inplace=True)
            df['vnindex_close'] = df_vnindex['close']
        except: df['vnindex_close'] = np.nan
        return df
    except: return pd.DataFrame()

def fetch_google_news(ticker: str) -> tuple:
    news_titles = []
    latest_url = "#"
    try:
        query = quote(f"{ticker} chứng khoán Việt Nam vĩ mô thế giới")
        url = f"https://news.google.com/rss/search?q={query}&hl=vi&gl=VN&ceid=VN:vi"
        feed = feedparser.parse(url)
        for i, entry in enumerate(feed.entries[:4]):
            news_titles.append(f"- {entry.title}")
            if i == 0: latest_url = entry.link
    except: pass
    return news_titles, latest_url

def engineer_features_online(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(close=df['close']).macd()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['Daily_Return'] = df['close'].pct_change()
    df['VNIndex_Return'] = df['vnindex_close'].pct_change() if 'vnindex_close' in df else 0.0
    np.random.seed(42)
    df['Net_Foreign_Buy'] = np.random.normal(0, 1000000, size=len(df))
    df.dropna(inplace=True)
    return df

def generate_ai_report(metrics_data: dict) -> dict:
    ticker = metrics_data['ticker']
    api_key = GEMINI_API_KEY_1 if ticker in GROUP_1 else GEMINI_API_KEY_2
    
    if not api_key: 
        return {"khuyen_nghi": "LỖI", "nhan_dinh": "Thiếu cấu hình API Key."}
        
    try:
        client = genai.Client(api_key=api_key)
        news_section = "\n".join(metrics_data['recent_news']) if metrics_data['recent_news'] else "- Không có sự kiện vĩ mô nổi bật nào trên Google News."

        prompt = f"""
        Đóng vai một "Chuyên gia đánh giá tâm lý thị trường (Sentiment Analyzer)" xuất sắc chuyên về cổ phiếu Việt Nam.
        Dưới đây là [Thông số Kỹ thuật XGBoost] và [Tin tức Vĩ mô/Pháp luật Mới nhất] từ báo chí:
        
        - Mã cổ phiếu: {metrics_data['ticker']}
        - Giá hiện tại: {metrics_data['close']:,.0f} VNĐ
        - RSI (14): {metrics_data['rsi']:.1f}
        - MACD: {metrics_data['macd']:.1f}
        - Xác suất TĂNG từ AI Model (XGBoost): {metrics_data['prob_up']:.1f}%

        🗞️ TIN TỨC SỰ KIỆN NỔI BẬT TỪ GOOGLE NEWS:
        {news_section}

        Nhiệm vụ:
        1. Phân tích Tin tức (Sentiment): Xác định ngay loạt sự kiện trên có sắc thái Tích cực (Positive), Tiêu cực (Negative), hay Trung lập (Neutral).
        2. Đối chiếu Kỹ thuật vs Tin tức (Tuyệt Mật): Cảnh báo dứt khoát nếu Tin tức đang mâu thuẫn với chỉ báo XGBoost. (Ví dụ: XGBoost dự báo xác suất tăng cực cao, nhưng nếu tin tức có từ khóa Bắt bớ/Pháp luật/Thao túng thì chốt BÁN NGAY BẤT CHẤP KỸ THUẬT).
        3. Kết luận Hành động: Chốt lại trạng thái đầu tư dứt khoát nhất cho khung 7 ngày tới dựa trên phân tích toàn diện.

        Yêu cầu: Trả về DUY NHẤT một chuỗi JSON hợp lệ (không chứa markdown ```json, không giải thích dài dòng), với cấu trúc sau:
        {{
            "khuyen_nghi": "MUA, BÁN, QUAN SÁT",
            "nhan_dinh": "1 câu nhận định sắc bén và cực kỳ ngắn gọn (dưới 25 chữ) thể hiện sự đối chiếu giữa Kỹ thuật và Tin tức để ra quyết định."
        }}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        # Tiền xử lý chuỗi trả về để loại bỏ markdown json nếu có
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(raw_text)
    except Exception as e:
        return {"khuyen_nghi": "LỖI", "nhan_dinh": f"Lỗi Gemini: {str(e)}"}

def send_html_email(subject: str, html_body: str):
    msg = MIMEMultipart('alternative')
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()

# ==========================================
# LUỒNG CHẠY CHÍNH (MAIN WORKFLOW)
# ==========================================
def main():
    print("🚀 Bắt đầu quét hệ thống...")
    model_path = 'xgboost_model_7d.joblib'
    if not os.path.exists(model_path):
        print("Không tìm thấy Model.")
        return
        
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    
    # Khởi tạo HTML Table
    html_content = f"""
    <html>
    <body style="background-color: #0b0f19; font-family: Arial, sans-serif; padding: 20px;">
        <h2 style="color: #ffffff;">BÁO CÁO CỔ PHIẾU (KHUNG 7 NGÀY) - {datetime.now().strftime('%d/%m/%Y')}</h2>
        <table style="width: 100%; border-collapse: collapse; background-color: #1a1e29; color: #ffffff; border-radius: 8px; overflow: hidden;">
            <thead style="background-color: #0d1b2a; color: #4db8ff; border-bottom: 2px solid #2d3748;">
                <tr>
                    <th style="padding: 15px; text-align: left;">MÃ CP</th>
                    <th style="padding: 15px; text-align: center;">XGBOOST (%)</th>
                    <th style="padding: 15px; text-align: center;">KHUYẾN NGHỊ AI</th>
                    <th style="padding: 15px; text-align: left;">NHẬN ĐỊNH CHI TIẾT</th>
                    <th style="padding: 15px; text-align: center;">TIN TỨC</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for idx, ticker in enumerate(TICKERS):
        print(f"Đang xử lý {ticker} ({idx+1}/{len(TICKERS)})...")
        df = fetch_stock_data_online(ticker)
        if df.empty: continue
        
        df_features = engineer_features_online(df)
        if df_features.empty: continue
        
        recent_news, latest_url = fetch_google_news(ticker)
        latest_data = df_features.iloc[-1]
        X_latest = pd.DataFrame([latest_data[features].fillna(0).values], columns=features)
        
        prob_up = model.predict_proba(X_latest)[0][1] * 100
        
        metrics_payload = {
            'ticker': ticker, 'prob_up': prob_up,
            'close': latest_data['close'], 'rsi': latest_data['RSI_14'], 'macd': latest_data['MACD'],
            'recent_news': recent_news
        }
        
        ai_data = generate_ai_report(metrics_payload)
        khuyen_nghi = str(ai_data.get('khuyen_nghi', 'QUAN SÁT')).upper()
        nhan_dinh = ai_data.get('nhan_dinh', 'Không có nhận định.')
        
        # Đổi màu Badge
        bg_color = "#f39c12" # QUAN SÁT (Vàng)
        if khuyen_nghi == "MUA": bg_color = "#00b894" # Xanh
        elif khuyen_nghi == "BÁN": bg_color = "#d63031" # Đỏ
            
        badge = f'<span style="background-color: {bg_color}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold;">{khuyen_nghi}</span>'
        row_bg = "#1a1e29" if idx % 2 == 0 else "#232836"
        
        # Thêm dòng vào bảng
        html_content += f"""
            <tr style="background-color: {row_bg}; border-bottom: 1px solid #2d3748;">
                <td style="padding: 15px; font-weight: bold; font-size: 16px;">{ticker}</td>
                <td style="padding: 15px; text-align: center; font-weight: bold;">{prob_up:.1f}%</td>
                <td style="padding: 15px; text-align: center;">{badge}</td>
                <td style="padding: 15px; font-size: 14px; line-height: 1.5; color: #d1d5db;">{nhan_dinh}</td>
                <td style="padding: 15px; text-align: center;">
                    <a href="{latest_url}" style="color: #3b82f6; text-decoration: none; display: inline-flex; align-items: center;">
                        📰 Xem tin
                    </a>
                </td>
            </tr>
        """
        
        time.sleep(10) # Chờ 10 giây để KHÔNG BAO GIỜ bị quá tải API
        
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    print("Đang gửi Email HTML...")
    send_html_email(f"[VN Stock] Báo cáo AI Định Lượng - {datetime.now().strftime('%d/%m/%Y')}", html_content)
    print("✅ Xong!")

if __name__ == "__main__":
    main()
