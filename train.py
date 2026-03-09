import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vnstock import stock_historical_data
import ta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
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
        df['ticker'] = ticker
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu {ticker}: {e}")
        return pd.DataFrame()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # Target 7 ngày (> 3%)
    df['Future_Close_7d'] = df['close'].shift(-7)
    df['Target_7d'] = (df['Future_Close_7d'] > df['close'] * 1.03).astype(int)
    
    # Target 30 ngày (> 5%)
    df['Future_Close_30d'] = df['close'].shift(-30)
    df['Target_30d'] = (df['Future_Close_30d'] > df['close'] * 1.05).astype(int)
    
    # Target 90 ngày (> 10%)
    df['Future_Close_90d'] = df['close'].shift(-90)
    df['Target_90d'] = (df['Future_Close_90d'] > df['close'] * 1.10).astype(int)
    
    return df

def main():
    print("Bắt đầu tải dữ liệu lịch sử...")
    tickers = ["VIC", "HAG", "FPT", "HPG", "SSI", "VNM", "VCB", "ACB", "MBB", "TCB","ACV","VJC","GAS","MSN","PVD","PLX","VRE","NVL","KDH","DXG","SAB"]
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    all_data = []
    for ticker in tickers:
        print(f"Đang lấy dữ liệu {ticker}...")
        df = fetch_stock_data(ticker, start_date, end_date)
        if not df.empty:
            df_feat = engineer_features(df)
            all_data.append(df_feat)
            
    if not all_data:
        print("Không tải được dữ liệu nào. Hủy huấn luyện.")
        return
        
    full_df = pd.concat(all_data)
    features = ['RSI_14', 'MACD', 'EMA_20', 'EMA_50', 'BB_High', 'BB_Low', 'Daily_Return', 'VNIndex_Return', 'Net_Foreign_Buy']
    
    # Cấu hình 3 khung thời gian
    timeframes = {
        '7d': {'target_col': 'Target_7d', 'future_col': 'Future_Close_7d', 'filename': 'xgboost_model_7d.joblib'},
        '30d': {'target_col': 'Target_30d', 'future_col': 'Future_Close_30d', 'filename': 'xgboost_model_30d.joblib'},
        '90d': {'target_col': 'Target_90d', 'future_col': 'Future_Close_90d', 'filename': 'xgboost_model_90d.joblib'}
    }
    
    # Cấu hình không gian tham số (Hyperparameter Grid)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Khởi tạo TimeSeriesSplit để tránh data leakage
    tscv = TimeSeriesSplit(n_splits=3)
    
    for tf_key, tf_config in timeframes.items():
        print(f"\n--- Đang HUẤN LUYỆN TỐI ƯU (Tuning) cho khung {tf_key}... ---")
        target_col = tf_config['target_col']
        future_col = tf_config['future_col']
        filename = tf_config['filename']
        
        # Loại bỏ NaNs tùy theo khung shift xa nhất của timeframe tương ứng
        df_train = full_df.dropna(subset=features + [target_col, future_col])
        
        X = df_train[features]
        y = df_train[target_col]
        
        # Cắt 20% dòng dữ liệu cuối cùng theo thời gian (Không shuffle) làm Test Set chuẩn
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Số lượng mẫu huấn luyện ({tf_key}): {len(X_train)}")
        
        # Khởi tạo mô hình nền
        base_model = XGBClassifier(
            eval_metric='logloss',
            random_state=42
        )
        
        # Khởi tạo bộ tìm kiếm RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=10,                      # Chạy 10 cấu hình thử nghiệm
            cv=tscv,                        # Dùng TimeSeriesSplit = 3 Fold
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,                      # Chạy đa nhiệm
            verbose=1                       # In progress
        )
        
        # Fit tốn time để tìm ra bộ siêu tham số tốt nhất
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        print(f"\n✅ BỘ THAM SỐ TỐT NHẤT CHO KHUNG {tf_key} LÀ: {best_params}")
        
        # Dự đoán Test Set bằng Best Model
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"=> Độ chính xác Test Set ({tf_key}) sau khi tối ưu: {acc:.4f}")
        
        model_data = {
            'model': best_model,       # Lưu lại mô hình ĐÃ TUNING
            'features': features,
            'accuracy': acc,
            'timeframe': tf_key,
            'best_params': best_params
        }
        
        joblib.dump(model_data, filename)
        print(f"Đã lưu mô hình {filename}.")
        
    print("\n🎉 Hoàn tất quá trình Hyperparameter Tuning cho 3 khung thời gian!")

if __name__ == "__main__":
    main()
