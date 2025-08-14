
#  
# 檔名: generate_agent_performance.py  
# 目的: 讀取歸因分析檔案，生成以催收員為核心的績效賦能數據集。  
#

import pandas as pd

def main():  
    """主執行函數"""  
    print("\n--- 開始執行第二階段：生成催收員績效賦能檔案 ---")

    # --- 1. 讀取基石數據 ---  
    try:  
        attr_df = pd.read_csv('attribution_analysis.csv')  
        calls_df = pd.read_csv('consolidated_report.csv')  
        print("歸因分析檔案與通話紀錄成功載入！")  
    except FileNotFoundError as e:  
        print(f"錯誤：找不到必要的檔案 {e.filename}。請先執行第一階段。")  
        return  
          
    # --- 2. 資料預處理 ---  
    calls_df['Agent ID'] = calls_df['Agent ID'].astype(str).str.strip()  
    attr_df['agents_in_window'] = attr_df['agents_in_window'].astype(str) # 確保能使用 .str.contains

    # 獲取所有不重複的催收員ID  
    all_agents = calls_df['Agent ID'].unique()  
    agent_summary = []

    print("正在為每一位催收員計算績效指標...")  
    # --- 3. 以催收員為核心，遍歷計算績效指標 ---  
    from tqdm.auto import tqdm
    for agent in tqdm(all_agents, desc="計算催員績效..."):  
        # 計算基礎指標 (總通話數/時長)  
        agent_calls_df = calls_df[calls_df['Agent ID'] == agent]  
        agent_total_calls = len(agent_calls_df)  
        agent_total_talk_time = pd.to_timedelta(agent_calls_df['Talk Durations']).sum().total_seconds()  
          
        # 計算歸因指標  
        influenced_cases = attr_df[attr_df['agents_in_window'].str.contains(agent, na=False)]  
        closed_cases = attr_df[attr_df['last_agent_in_window'] == agent]  
          
        total_collections_influenced = influenced_cases['payment_amount'].sum()  
        total_collections_closed = closed_cases['payment_amount'].sum()  
        assist_value = total_collections_influenced - total_collections_closed  
          
        # 計算效率指標  
        total_talk_time_on_influenced_cases = influenced_cases['talk_time_in_window'].sum()  
          
        if total_collections_influenced > 0 and total_talk_time_on_influenced_cases > 0:  
            collection_efficiency_ratio = total_collections_influenced / total_talk_time_on_influenced_cases  
        else:  
            collection_efficiency_ratio = 0  
              
        avg_touches_per_collection = influenced_cases['calls_in_window'].mean() if not influenced_cases.empty else 0  
              
        agent_summary.append({  
            'Agent ID': agent,  
            'total_collections_influenced': total_collections_influenced,  
            'total_collections_closed': total_collections_closed,  
            'assist_value': assist_value,  
            'collection_efficiency_ratio': collection_efficiency_ratio,  
            'avg_touches_per_collection': avg_touches_per_collection,  
            'total_calls': agent_total_calls,  
            'total_talk_time_sec': agent_total_talk_time  
        })

    # --- 4. 生成最終報表 ---  
    print("計算完成，正在儲存結果...")  
    summary_df = pd.DataFrame(agent_summary)  
    output_filename = 'agent_performance_summary.csv'  
    summary_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"--- 第二階段完成！成功生成 `{output_filename}` ---")

if __name__ == '__main__':  
    main()
