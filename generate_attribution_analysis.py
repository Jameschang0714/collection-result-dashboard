#
# 檔名: generate_attribution_analysis.py
# 目的: 讀取通話紀錄與KPI檔案，生成包含「總撥打數」與「總接通數」的歸因分析數據集。
# 版本: 2.0
#

import pandas as pd
import datetime
from tqdm.auto import tqdm

def main():
    """主執行函數"""
    print("--- 開始執行 V2：生成歸因分析檔案 (含接通數) ---")

    # --- 1. 讀取資料 & 準備 ---
    try:
        calls_df = pd.read_csv('consolidated_report.csv')
        kpi_df = pd.read_csv('KH_DM_FACT_COLL_KPI_DTL.csv')
        print("通話紀錄與KPI檔案成功載入！")
    except FileNotFoundError as e:
        print(f"錯誤：找不到必要的檔案 {e.filename}。請確認檔案是否存在於同個目錄。")
        return

    # --- 2. 資料預處理 (Data Preprocessing) ---
    print("正在進行資料預處理...")
    # A. 處理通話紀錄檔
    calls_df['Date'] = pd.to_datetime(calls_df['Date'])
    calls_df['Case No'] = calls_df['Case No'].astype(str).str.strip()
    calls_df['Agent ID'] = calls_df['Agent ID'].astype(str).str.strip()
    # 轉換通話時長為秒
    calls_df['talk_seconds'] = calls_df['Talk Durations'].apply(lambda x: pd.to_timedelta(x).total_seconds())
    # 確保 'Connected' 欄位為數值類型
    calls_df['Connected'] = pd.to_numeric(calls_df['Connected'], errors='coerce').fillna(0).astype(int)


    # B. 處理KPI檔案，篩選出有效還款紀錄
    kpi_df['RCV_AMT_ACTUAL'] = pd.to_numeric(kpi_df['RCV_AMT_ACTUAL'], errors='coerce')
    payments_df = kpi_df[kpi_df['RCV_DT'].notna() & (kpi_df['RCV_AMT_ACTUAL'] > 0)].copy()
    payments_df['payment_date'] = pd.to_datetime(payments_df['RCV_DT'], errors='coerce')
    payments_df.dropna(subset=['payment_date'], inplace=True)

    # 將 Case No 與 Agent ID 的欄位名稱統一
    payments_df.rename(columns={
        'CNTRT_NO': 'Case No',
        'USR_ID': 'Agent ID',
        'RCV_AMT_ACTUAL': 'payment_amount'
    }, inplace=True)
    payments_df['Case No'] = payments_df['Case No'].astype(str).str.strip()

    print(f"從KPI檔案中篩選出 {len(payments_df)} 筆有效還款紀錄。")

    # --- 3. 設定歸因模型參數 ---
    ATTRIBUTION_WINDOW_DAYS = 30
    print(f"歸因窗口期設定為 {ATTRIBUTION_WINDOW_DAYS} 天。")

    # --- 4. 核心歸因邏輯 (V2 升級) ---
    print("正在對每一筆還款進行歸因計算...")

    def get_attribution_metrics(payment_row, all_calls_df):
        end_date = payment_row['payment_date']
        start_date = end_date - datetime.timedelta(days=ATTRIBUTION_WINDOW_DAYS)
        case_no = payment_row['Case No']

        window_calls = all_calls_df[
            (all_calls_df['Case No'] == case_no) &
            (all_calls_df['Date'] >= start_date) &
            (all_calls_df['Date'] <= end_date)
        ].copy()

        if window_calls.empty:
            # 如果窗口內無任何通話，所有指標均為0或None
            return pd.Series([0, 0, 0, None, None])

        # 計算總撥打次數
        calls_in_window = len(window_calls)
        # 計算總通話秒數
        talk_time_in_window = window_calls['talk_seconds'].sum()
        # V2 新增：計算總接通次數
        connected_calls_in_window = window_calls['Connected'].sum()

        # 識別互動過的催員
        agents = sorted(window_calls['Agent ID'].unique())
        agents_in_window = ', '.join(agents)

        # 識別最後一位互動的催員
        last_agent_in_window = window_calls.sort_values(by='Date', ascending=True).iloc[-1]['Agent ID']

        return pd.Series([calls_in_window, connected_calls_in_window, talk_time_in_window, agents_in_window, last_agent_in_window])

    # 應用函式到每一筆還款
    tqdm.pandas(desc="歸因計算中...")
    attribution_cols = ['calls_in_window', 'connected_calls_in_window', 'talk_time_in_window', 'agents_in_window', 'last_agent_in_window']
    payments_df[attribution_cols] = payments_df.progress_apply(lambda row: get_attribution_metrics(row, calls_df), axis=1)

    # --- 5. 儲存結果 ---
    print("計算完成，正在儲存結果...")
    # 將 Agent ID 欄位重新命名，以明確其代表「案件分配催員」
    payments_df.rename(columns={'Agent ID': 'assigned_agent_id'}, inplace=True)
    
    # 調整輸出欄位順序，將新欄位包含進來
    output_df = payments_df[[
        'Case No',
        'payment_date',
        'payment_amount',
        'assigned_agent_id', # 案件當時分配的催收員
        'FINANCE_TYPE',
        'ASSIGN_OVER_AMT'
    ] + attribution_cols]
    
    # 將最後互動的催員 (last_agent_in_window) 重新命名為 Agent ID，作為此筆還款的主要歸因對象
    output_df.rename(columns={'last_agent_in_window': 'Agent ID'}, inplace=True)


    output_filename = 'attribution_analysis.csv'
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"--- V2 腳本執行完畢！成功生成包含接通數的 `{output_filename}` ---")

if __name__ == '__main__':
    main()
