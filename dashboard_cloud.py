import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Google Drive API imports
import google.oauth2.service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# --- 頁面設定 (Page Config) ---
st.set_page_config(
    page_title="電催績效追蹤歸因分析儀表板 (雲端版)",
    page_icon="🏆",
    layout="wide"
)

# --- 樣式設定 (Custom CSS) ---
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        background-color: #f0f2f6;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }
    .st-emotion-cache-1g8m2r4 { color: #0052cc; }
    .st-emotion-cache-1r4qj8v { font-size: 1.75rem; }
    .dataframe-container table { width: 100%; border-collapse: collapse; }
    .dataframe-container th, .dataframe-container td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
    .dataframe-container th { background-color: #f2f2f2; }
</style>
""", unsafe_allow_html=True)


# --- 常量管理 (Constants Management) ---
# DataFrame Column Names
COL_AGENT_ID = 'Agent ID'
COL_AGENT_NAME = '催員名稱'
COL_GROUP = '組別'
COL_PAYMENT_DATE = 'payment_date'
COL_PAYMENT_AMOUNT = 'payment_amount'
COL_CALLS_IN_WINDOW = 'calls_in_window'
COL_CONNECTED_CALLS_IN_WINDOW = 'connected_calls_in_window'
COL_TALK_TIME_IN_WINDOW = 'talk_time_in_window'
COL_CASE_NO = 'Case No'
COL_DISPLAY_NAME = 'display_name'

# Metric Keys (Internal)
METRIC_TOTAL_COLLECTIONS = 'total_collections_closed'
METRIC_TOTAL_CALLS = 'total_calls'
METRIC_TOTAL_CONNECTED_CALLS = 'total_connected_calls'
METRIC_TOTAL_TALK_TIME = 'total_talk_time_sec'
METRIC_TOTAL_CASES = 'total_cases_handled'
METRIC_AVG_TOUCHES = 'avg_touches_per_collection'
METRIC_AVG_CONNECTED_TOUCHES = 'avg_connected_touches_per_case'
METRIC_EFFICIENCY_PER_CALL = 'efficiency_per_connected_call'
METRIC_EFFICIENCY_PER_HOUR = 'efficiency_per_hour'

# Metric Translations (Display)
METRICS_TRANSLATION = {
    '結案催回總額': METRIC_TOTAL_COLLECTIONS,
    '總撥打電話數': METRIC_TOTAL_CALLS,
    '總接通電話數': METRIC_TOTAL_CONNECTED_CALLS,
    '總通話時長': METRIC_TOTAL_TALK_TIME,
    '處理案件總數': METRIC_TOTAL_CASES,
    '平均撥打次數/案': METRIC_AVG_TOUCHES,
    '平均接通次數/案': METRIC_AVG_CONNECTED_TOUCHES,
    '效率 (金額/有效通話)': METRIC_EFFICIENCY_PER_CALL,
    '效率 (金額/小時)': METRIC_EFFICIENCY_PER_HOUR,
}
REVERSE_METRICS_TRANSLATION = {v: k for k, v in METRICS_TRANSLATION.items()}


# --- 輔助函式 (Helper Functions) ---
def format_seconds_to_hms(seconds):
    if pd.isna(seconds) or not np.isfinite(seconds):
        return "00:00:00"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def conditional_format(val):
    if isinstance(val, (int, float)):
        return f"{val:,.2f}"
    return str(val)

# --- 數據載入與快取 (Data Loading & Caching) ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data_from_gdrive(file_id, file_name):
    try:
        # Authenticate with Google Service Account
        creds_json = st.secrets["gcp_service_account"]
        creds = google.oauth2.service_account.Credentials.from_service_account_info(creds_json)
        
        service = build('drive', 'v3', credentials=creds)
        
        # Download the file
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            st.spinner(f"Downloading {file_name}: {int(status.progress() * 100)}%.")
        
        fh.seek(0) # Rewind to the beginning of the file
        
        # Determine file type and read accordingly
        if file_name.endswith('.csv'):
            df = pd.read_csv(fh)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(fh, engine='openpyxl')
        else:
            st.error(f"Unsupported file type for {file_name}. Only .csv and .xlsx are supported.")
            return None
        
        st.success(f"Successfully loaded {file_name} from Google Drive.")
        return df
        
    except Exception as e:
        st.error(f"Error loading {file_name} from Google Drive: {e}")
        st.info("請確認您的 Google Drive 檔案 ID 和服務帳號金鑰設定正確。")
        return None

@st.cache_data
def load_and_prep_data():
    try:
        # Get file IDs from Streamlit secrets
        ATTRIBUTION_FILE_ID = st.secrets["ATTRIBUTION_FILE_ID"]
        GROUP_LIST_FILE_ID = st.secrets["GROUP_LIST_FILE_ID"]

        # Load dataframes from Google Drive
        attr_df = load_data_from_gdrive(ATTRIBUTION_FILE_ID, "attribution_analysis.csv")
        group_df = load_data_from_gdrive(GROUP_LIST_FILE_ID, "分組名單.xlsx")

        if attr_df is None or group_df is None:
            return None

        attr_df[COL_PAYMENT_DATE] = pd.to_datetime(attr_df[COL_PAYMENT_DATE])

        # --- Data Availability Check ---
        st.session_state['has_connected_calls_data'] = COL_CONNECTED_CALLS_IN_WINDOW in attr_df.columns and attr_df[COL_CONNECTED_CALLS_IN_WINDOW].notna().any()
        st.session_state['has_talk_time_data'] = COL_TALK_TIME_IN_WINDOW in attr_df.columns and attr_df[COL_TALK_TIME_IN_WINDOW].notna().any()
        
        # Ensure columns exist to prevent downstream errors
        if not st.session_state['has_connected_calls_data']:
            attr_df[COL_CONNECTED_CALLS_IN_WINDOW] = 0
        if not st.session_state['has_talk_time_data']:
            attr_df[COL_TALK_TIME_IN_WINDOW] = 0

        # Ensure column names for group_df are correct after loading from Google Drive
        group_df.columns = [COL_GROUP, COL_AGENT_ID, COL_AGENT_NAME] # Force column names

        merged_df = pd.merge(attr_df, group_df, on=COL_AGENT_ID, how='inner')
        return merged_df

    except KeyError as e:
        st.error(f"請在 `.streamlit/secrets.toml` 中設定必要的 Google Drive 檔案 ID 或服務帳號金鑰: {e}")
        return None
    except Exception as e:
        st.error(f"讀取或處理檔案時發生錯誤：{e}")
        return None

# --- 可視化圖表函式 (Visualization Functions) ---
def create_performance_trend_chart(df, metric_col, time_unit, group_sizes, show_per_capita=True):
    if df.empty or metric_col not in df.columns:
        return go.Figure().update_layout(title_text='沒有足夠的數據生成趨勢圖')
    
    df_copy = df.copy()
    df_copy = df_copy.set_index(COL_PAYMENT_DATE)
    
    unit_map = {'日': 'D', '週': 'W'}
    resample_unit = unit_map.get(time_unit, 'D')
    
    metric_display_name = REVERSE_METRICS_TRANSLATION.get(metric_col, metric_col)
    
    total_metrics = [METRIC_TOTAL_COLLECTIONS, METRIC_TOTAL_CALLS, METRIC_TOTAL_CONNECTED_CALLS, METRIC_TOTAL_TALK_TIME, METRIC_TOTAL_CASES]
    is_total_metric = metric_col in total_metrics
    
    fig = go.Figure()
    
    yaxis_title = f'{time_unit} {metric_display_name}'
    if show_per_capita and is_total_metric:
        yaxis_title = f'{time_unit}人均{metric_display_name}'

    for group in sorted(df_copy[COL_GROUP].unique()):
        group_df = df_copy[df_copy[COL_GROUP] == group]
        
        agg_dict = {
            METRIC_TOTAL_COLLECTIONS: 'sum',
            METRIC_TOTAL_CALLS: 'sum',
            METRIC_TOTAL_CONNECTED_CALLS: 'sum',
            METRIC_TOTAL_TALK_TIME: 'sum',
            METRIC_TOTAL_CASES: 'sum',
        }
        resampled_data = group_df.groupby(pd.Grouper(freq=resample_unit)).agg(agg_dict)

        resampled_data[METRIC_AVG_TOUCHES] = (resampled_data[METRIC_TOTAL_CALLS] / resampled_data[METRIC_TOTAL_CASES]).fillna(0)
        resampled_data[METRIC_AVG_CONNECTED_TOUCHES] = (resampled_data[METRIC_TOTAL_CONNECTED_CALLS] / resampled_data[METRIC_TOTAL_CASES]).fillna(0)
        resampled_data[METRIC_EFFICIENCY_PER_CALL] = (resampled_data[METRIC_TOTAL_COLLECTIONS] / resampled_data[METRIC_TOTAL_CONNECTED_CALLS]).replace([np.inf, -np.inf], 0).fillna(0)
        resampled_data[METRIC_EFFICIENCY_PER_HOUR] = (resampled_data[METRIC_TOTAL_COLLECTIONS] / (resampled_data[METRIC_TOTAL_TALK_TIME] / 3600)).replace([np.inf, -np.inf], 0).fillna(0)
        
        trend_data = resampled_data[metric_col]

        if show_per_capita and is_total_metric:
            group_size = group_sizes.get(group, 1)
            if group_size > 0:
                trend_data = trend_data / group_size

        fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data.values, mode='lines+markers', name=group))
        
    fig.update_layout(
        title_text=f'團隊績效趨勢分析 ({time_unit}視角): {metric_display_name}',
        xaxis_title='日期',
        yaxis_title=yaxis_title,
        legend_title=COL_GROUP,
        font=dict(family="Arial, sans-serif", size=14)
    )
    return fig


def create_group_comparison_chart(df, metric_col):
    if df.empty or COL_GROUP not in df.columns or metric_col not in df.columns:
        return go.Figure().update_layout(title_text='沒有足夠的數據進行組別比較')
    metric_display_name = REVERSE_METRICS_TRANSLATION.get(metric_col, metric_col)
    group_performance = df.groupby(COL_GROUP)[metric_col].mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=group_performance.index, y=group_performance.values, text=group_performance.apply(lambda x: f'{x:,.2f}'), textposition='auto', marker_color='royalblue'))
    fig.update_layout(title_text=f'各組別平均「{metric_display_name}」比較', xaxis_title=COL_GROUP, yaxis_title=f'平均 {metric_display_name}', font=dict(family="Arial, sans-serif", size=14))
    return fig


def create_performance_matrix_chart(df, x_col, y_col):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure().update_layout(title_text='沒有數據可供顯示')
    
    if df[x_col].nunique() < 2 or df[y_col].nunique() < 2:
        return go.Figure().update_layout(title_text=f"指標缺乏足夠變異性，無法生成績效矩陣", xaxis_title=REVERSE_METRICS_TRANSLATION.get(x_col, x_col), yaxis_title=REVERSE_METRICS_TRANSLATION.get(y_col, y_col))

    x_display = REVERSE_METRICS_TRANSLATION.get(x_col, x_col)
    y_display = REVERSE_METRICS_TRANSLATION.get(y_col, y_col)
    title = f"績效矩陣分析：{y_display} vs. {x_display}"
    x_mean = df[x_col].mean()
    y_mean = df[y_col].mean()
    fig = go.Figure()
    for group in df[COL_GROUP].unique():
        group_df = df[df[COL_GROUP] == group]
        fig.add_trace(go.Scatter(x=group_df[x_col], y=group_df[y_col], mode='markers', name=group, customdata=group_df[[COL_AGENT_ID, COL_AGENT_NAME]], hovertemplate=("<b>催員:</b> %{customdata[1]} (%{customdata[0]})<br>" + f"<b>{x_display}:</b> %{{x:,.2f}}<br>" + f"<b>{y_display}:</b> %{{y:,.2f}}<extra></extra>"), marker=dict(size=12, line=dict(width=1), opacity=0.8)))
    fig.add_shape(type="line", x0=df[x_col].min(), y0=y_mean, x1=df[x_col].max(), y1=y_mean, line=dict(color="grey", width=2, dash="dash"), name="Y 平均")
    fig.add_shape(type="line", x0=x_mean, y0=df[y_col].min(), x1=x_mean, y1=df[y_col].max(), line=dict(color="grey", width=2, dash="dash"), name="X 平均")
    is_y_cost_metric = y_col in [] 
    if is_y_cost_metric:
        q1_name, q2_name, q3_name, q4_name = "重點輔導區", "勤奮探索區", "潛力種子區", "高效明星區"
    else:
        q1_name, q2_name, q3_name, q4_name = "潛力種子區", "高效明星區", "重點輔導區", "勤奮探索區"
    fig.add_annotation(x=x_mean, y=df[y_col].max(), xref="x", yref="y", text=q2_name, showarrow=False, xanchor='left', yanchor='top', font=dict(color="grey", size=12), align="left")
    fig.add_annotation(x=x_mean, y=df[y_col].min(), xref="x", yref="y", text=q3_name, showarrow=False, xanchor='left', yanchor='bottom', font=dict(color="grey", size=12), align="left")
    fig.add_annotation(x=df[x_col].min(), y=y_mean, xref="x", yref="y", text=q1_name, showarrow=False, xanchor='left', yanchor='top', font=dict(color="grey", size=12), align="left")
    fig.add_annotation(x=df[x_col].max(), y=y_mean, xref="x", yref="y", text=q4_name, showarrow=False, xanchor='right', yanchor='top', font=dict(color="grey", size=12), align="right")
    fig.update_layout(title_text=title, xaxis_title=x_display, yaxis_title=y_display, legend_title=COL_GROUP, font=dict(family="Arial, sans-serif", size=14), height=600)
    return fig

def create_coaching_radar_chart(df, coached_agent_id, coached_agent_display_name, benchmark_agents, scaling_method='rank'):
    metrics = [m for m in METRICS_TRANSLATION.values() if m in df.columns]
    df_normalized = df.copy()
    
    if scaling_method == 'rank':
        for col in metrics:
            df_normalized[col] = df[col].rank(pct=True)
    else:
        for col in metrics:
            df_normalized[f'{col}_log'] = np.log1p(df[col])
            min_val = df_normalized[f'{col}_log'].min()
            max_val = df_normalized[f'{col}_log'].max()
            if (max_val - min_val) > 0:
                df_normalized[col] = (df_normalized[f'{col}_log'] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 0.5
    
    theta_labels = [REVERSE_METRICS_TRANSLATION.get(m, m) for m in metrics]
    fig = go.Figure()

    coached_raw_data = df[df[COL_AGENT_ID] == coached_agent_id][metrics].iloc[0]
    coached_normalized_data = df_normalized[df_normalized[COL_AGENT_ID] == coached_agent_id][metrics].iloc[0]
    
    def format_hover_text(metric_key, raw_value, rank_val):
        metric_name = REVERSE_METRICS_TRANSLATION[metric_key]
        if metric_key in [METRIC_TOTAL_COLLECTIONS, METRIC_EFFICIENCY_PER_CALL, METRIC_EFFICIENCY_PER_HOUR]:
            formatted_value = f"${raw_value:,.0f}"
        elif metric_key in [METRIC_TOTAL_CALLS, METRIC_TOTAL_CONNECTED_CALLS, METRIC_TOTAL_CASES, METRIC_TOTAL_TALK_TIME]:
            formatted_value = f"{raw_value:,.0f}"
        else:
            formatted_value = f"{raw_value:,.2f}"
        return f"{metric_name}: {formatted_value}<br>(團隊排名: Top {rank_val:.0%})"

    hovertemplate_coached = [format_hover_text(label, raw_value, coached_normalized_data[label]) for label, raw_value in coached_raw_data.items()]

    r_coached = list(coached_normalized_data.values)
    r_coached.append(r_coached[0])
    theta_labels_closed = list(theta_labels)
    theta_labels_closed.append(theta_labels_closed[0])
    hovertemplate_coached.append(hovertemplate_coached[0])
    
    fig.add_trace(go.Scatterpolar(
        r=r_coached, theta=theta_labels_closed, fill='toself', name=f'受輔導人員: {coached_agent_display_name}',
        line=dict(color='deepskyblue'), fillcolor='rgba(0, 191, 255, 0.5)',
        customdata=hovertemplate_coached, hovertemplate='%{customdata}<extra></extra>'
    ))

    benchmark_df_raw = df[df[COL_AGENT_ID].isin(benchmark_agents)]
    benchmark_avg_raw_data = benchmark_df_raw[metrics].mean()
    benchmark_df_normalized = df_normalized[df_normalized[COL_AGENT_ID].isin(benchmark_agents)]
    benchmark_avg_normalized_data = benchmark_df_normalized[metrics].mean()
    
    hovertemplate_benchmark = [format_hover_text(label, raw_value, benchmark_avg_normalized_data[label]) for label, raw_value in benchmark_avg_raw_data.items()]

    r_benchmark = list(benchmark_avg_normalized_data.values)
    r_benchmark.append(r_benchmark[0])
    hovertemplate_benchmark.append(hovertemplate_benchmark[0])

    fig.add_trace(go.Scatterpolar(
        r=r_benchmark, theta=theta_labels_closed, fill='none', name='標竿群組平均',
        line=dict(color='red', dash='dash'), customdata=hovertemplate_benchmark,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title=dict(text=f'{coached_agent_display_name} vs. 標竿群組績效剖析', font=dict(size=20)), height=600)
    return fig, coached_raw_data, benchmark_avg_raw_data

# --- REFACTORED/NEW FUNCTION ---
def create_agent_deep_dive_distribution_chart(raw_df, agent_id, benchmark_ids, agent_display_name, threshold, metric_to_analyze):
    """
    建立一個分佈圖，用於比較個人與標竿群組在不同指標（通話次數、通話時長）上的工作模式。
    """
    fig = go.Figure()
    y_max = 0

    # 根據選擇的指標，設定對應的欄位名稱和顯示名稱
    if metric_to_analyze == '總撥打次數':
        metric_col = COL_CALLS_IN_WINDOW
        metric_display = '總撥打次數'
        xaxis_title = f'單一案件所需{metric_display}'
    elif metric_to_analyze == '總接通次數':
        metric_col = COL_CONNECTED_CALLS_IN_WINDOW
        metric_display = '總接通次數'
        xaxis_title = f'單一案件所需{metric_display}'
    elif metric_to_analyze == '通話時長':
        metric_col = COL_TALK_TIME_IN_WINDOW
        metric_display = '通話時長'
        xaxis_title = f'單一案件{metric_display} (秒)'
    else:
        return go.Figure().update_layout(title_text='無效的分析指標')

    # 過濾掉超過閾值的數據
    filtered_raw_df = raw_df[raw_df[metric_col] <= threshold]
    
    # 處理被分析人員的數據
    agent_df = filtered_raw_df[filtered_raw_df[COL_AGENT_ID] == agent_id]
    agent_avg_metric = 0
    if not agent_df.empty and agent_df[metric_col].sum() > 0:
        if metric_to_analyze in ['總撥打次數', '總接通次數']:
            agent_counts = agent_df[agent_df[metric_col] > 0][metric_col].value_counts().sort_index()
        else: # 通話時長分箱處理
            bins = np.arange(0, threshold + 31, 30)
            labels = [f'{i}-{i+30}' for i in bins[:-1]]
            agent_df['time_bin'] = pd.cut(agent_df[metric_col], bins=bins, labels=labels, right=False)
            agent_counts = agent_df['time_bin'].value_counts().sort_index()

        if not agent_counts.empty:
            y_max = agent_counts.values.max()
            fig.add_trace(go.Bar(x=agent_counts.index.astype(str), y=agent_counts.values, name=f'{agent_display_name} (案件數)', text=agent_counts.values, textposition='auto', marker_color='darkcyan'))
            agent_avg_metric = agent_df[agent_df[metric_col] > 0][metric_col].mean()

    # 處理標竿群組的數據
    benchmark_avg_metric = 0
    if benchmark_ids:
        benchmark_df = filtered_raw_df[filtered_raw_df[COL_AGENT_ID].isin(benchmark_ids)]
        if not benchmark_df.empty and benchmark_df[metric_col].sum() > 0:
            if metric_to_analyze in ['總撥打次數', '總接通次數']:
                benchmark_counts = benchmark_df[benchmark_df[metric_col] > 0][metric_col].value_counts()
            else: # 通話時長分箱處理
                bins = np.arange(0, threshold + 31, 30)
                labels = [f'{i}-{i+30}' for i in bins[:-1]]
                benchmark_df['time_bin'] = pd.cut(benchmark_df[metric_col], bins=bins, labels=labels, right=False)
                benchmark_counts = benchmark_df['time_bin'].value_counts()
            
            num_benchmark_agents = len(benchmark_ids)
            avg_benchmark_counts = (benchmark_counts / num_benchmark_agents).sort_index()
            
            if not avg_benchmark_counts.empty:
                y_max = max(y_max, avg_benchmark_counts.values.max())
                fig.add_trace(go.Scatter(x=avg_benchmark_counts.index.astype(str), y=avg_benchmark_counts.values, name='標竿群組 (平均案件數/人)', mode='lines+markers', line=dict(color='red', dash='dash')))
            benchmark_avg_metric = benchmark_df[benchmark_df[metric_col] > 0][metric_col].mean()

    # 添加平均值標示線
    if agent_avg_metric > 0:
        avg_text = f"個人平均: {agent_avg_metric:.2f}"
        if metric_to_analyze == '通話時長':
             avg_text += "s"
        fig.add_shape(type="line", x0=-0.5, y0=agent_avg_metric if metric_to_analyze != '通話時長' else None, x1=len(agent_counts)-0.5 if 'agent_counts' in locals() and not agent_counts.empty else 0, y1=agent_avg_metric if metric_to_analyze != '通話時長' else None, line=dict(color="deepskyblue", width=2, dash="dot"), yref='y' if metric_to_analyze != '通話時長' else 'paper')
        # The annotation logic needs to be smarter for binned data
        # For simplicity, we skip annotating average line for binned data for now.

    if benchmark_avg_metric > 0:
        avg_text = f"標竿平均: {benchmark_avg_metric:.2f}"
        if metric_to_analyze == '通話時長':
             avg_text += "s"
        # Similar annotation complexity for binned data.
        
    if y_max == 0:
        return go.Figure().update_layout(title_text=f'在此條件下無有效結案案件')
        
    fig.update_layout(
        title_text=f'<b>{agent_display_name}</b> vs. 標竿群組工作模式比較 ({metric_display})', 
        xaxis_title=xaxis_title, 
        yaxis_title='案件數量', 
        font=dict(family="Arial, sans-serif", size=14), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
        height=500, 
        barmode='group'
    )
    return fig


def create_payment_distribution_chart(raw_df, agent_id, benchmark_ids, agent_display_name):
    """
    建立一個箱型圖來比較個人與標竿群組的案件價值分佈。
    """
    fig = go.Figure()

    # 取得被分析人員的數據
    agent_df = raw_df[raw_df[COL_AGENT_ID] == agent_id]
    if not agent_df.empty:
        fig.add_trace(go.Box(
            y=agent_df[COL_PAYMENT_AMOUNT],
            name=agent_display_name,
            marker_color='darkcyan',
            boxpoints='all', # 顯示所有數據點
            jitter=0.3,
            pointpos=-1.8
        ))

    # 取得標竿群組的數據
    if benchmark_ids:
        benchmark_df = raw_df[raw_df[COL_AGENT_ID].isin(benchmark_ids)]
        if not benchmark_df.empty:
            fig.add_trace(go.Box(
                y=benchmark_df[COL_PAYMENT_AMOUNT],
                name='標竿群組',
                marker_color='tomato',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

    if agent_df.empty and (not benchmark_ids or benchmark_df.empty):
        return go.Figure().update_layout(title_text='沒有足夠的案件數據可供分析價值分佈')

    fig.update_layout(
        title_text=f'<b>{agent_display_name}</b> vs. 標竿群組案件價值分佈比較',
        yaxis_title='單案催回金額 ($)',
        font=dict(family="Arial, sans-serif", size=14),
        showlegend=False # 箱型圖的名稱會顯示在X軸上
    )
    return fig

# --- 主應用程式介面 (Main App Interface) ---

st.title("📊 電催績效追蹤歸因分析儀表板 (雲端版)")
st.caption("一個整合了時間序列分析與質效評估的決策支援平台")

try:
    df_raw = load_and_prep_data()
except Exception as e:
    st.error(f"儀表板啟動失敗：{e}")
    df_raw = None

if df_raw is not None:
    st.sidebar.header("⚙️ 控制面板")
    
    st.sidebar.subheader("選擇分析日期區間")
    min_date = df_raw[COL_PAYMENT_DATE].min().date()
    max_date = df_raw[COL_PAYMENT_DATE].max().date()
    start_date = st.sidebar.date_input("開始日期", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("結束日期", max_date, min_value=start_date, max_value=max_date)

    date_filtered_raw_df = df_raw[(df_raw[COL_PAYMENT_DATE].dt.date >= start_date) & (df_raw[COL_PAYMENT_DATE].dt.date <= end_date)].copy()

    agent_summary_df = pd.DataFrame()
    daily_summary_df = pd.DataFrame()
    group_sizes = {}

    if not date_filtered_raw_df.empty:
        rename_dict = {
            COL_PAYMENT_AMOUNT: METRIC_TOTAL_COLLECTIONS,
            COL_CALLS_IN_WINDOW: METRIC_TOTAL_CALLS,
            COL_CONNECTED_CALLS_IN_WINDOW: METRIC_TOTAL_CONNECTED_CALLS,
            COL_TALK_TIME_IN_WINDOW: METRIC_TOTAL_TALK_TIME,
            COL_CASE_NO: METRIC_TOTAL_CASES
        }
        
        daily_agg_funcs = {
            COL_PAYMENT_AMOUNT: 'sum', COL_CALLS_IN_WINDOW: 'sum', 
            COL_CONNECTED_CALLS_IN_WINDOW: 'sum', COL_TALK_TIME_IN_WINDOW: 'sum', 
            COL_CASE_NO: 'nunique'
        }
        daily_summary_df = date_filtered_raw_df.groupby([COL_AGENT_ID, COL_AGENT_NAME, COL_GROUP, pd.Grouper(key=COL_PAYMENT_DATE, freq='D')]).agg(daily_agg_funcs).reset_index()
        daily_summary_df.rename(columns=rename_dict, inplace=True)
        
        agent_agg_funcs = {
            COL_PAYMENT_AMOUNT: 'sum', COL_CALLS_IN_WINDOW: 'sum', 
            COL_CONNECTED_CALLS_IN_WINDOW: 'sum', COL_TALK_TIME_IN_WINDOW: 'sum', 
            COL_CASE_NO: 'nunique'
        }
        agent_summary_df = date_filtered_raw_df.groupby([COL_AGENT_ID, COL_AGENT_NAME, COL_GROUP]).agg(agent_agg_funcs).reset_index()
        agent_summary_df.rename(columns=rename_dict, inplace=True)
        
        agent_summary_df[METRIC_AVG_TOUCHES] = (agent_summary_df[METRIC_TOTAL_CALLS] / agent_summary_df[METRIC_TOTAL_CASES]).fillna(0)
        agent_summary_df[METRIC_AVG_CONNECTED_TOUCHES] = (agent_summary_df[METRIC_TOTAL_CONNECTED_CALLS] / agent_summary_df[METRIC_TOTAL_CASES]).fillna(0)
        agent_summary_df[METRIC_EFFICIENCY_PER_CALL] = (agent_summary_df[METRIC_TOTAL_COLLECTIONS] / agent_summary_df[METRIC_TOTAL_CONNECTED_CALLS]).replace([np.inf, -np.inf], 0).fillna(0)
        agent_summary_df[METRIC_EFFICIENCY_PER_HOUR] = (agent_summary_df[METRIC_TOTAL_COLLECTIONS] / (agent_summary_df[METRIC_TOTAL_TALK_TIME] / 3600)).replace([np.inf, -np.inf], 0).fillna(0)
        
        group_sizes = agent_summary_df.groupby(COL_GROUP)[COL_AGENT_ID].nunique()

    st.sidebar.subheader("團隊與人員篩選")
    
    if not agent_summary_df.empty:
        all_groups = sorted(agent_summary_df[COL_GROUP].unique())
        selected_groups = st.sidebar.multiselect('選擇比較組別 (可多選)', options=all_groups, default=all_groups)
    else:
        selected_groups = []

    if selected_groups:
        filtered_df = agent_summary_df[agent_summary_df[COL_GROUP].isin(selected_groups)]
        filtered_daily_df = daily_summary_df[daily_summary_df[COL_GROUP].isin(selected_groups)]
    else:
        filtered_df = pd.DataFrame(columns=agent_summary_df.columns)
        filtered_daily_df = pd.DataFrame(columns=daily_summary_df.columns)

    if not filtered_df.empty:
        filtered_df[COL_DISPLAY_NAME] = filtered_df[COL_AGENT_ID] + " (" + filtered_df[COL_AGENT_NAME] + ")"
        agent_display_list = ['全體'] + sorted(filtered_df[COL_DISPLAY_NAME].unique())
    else:
        agent_display_list = ['全體']
    selected_display_name = st.sidebar.selectbox('選擇催收員', agent_display_list)

    if selected_display_name != '全體':
        final_filtered_df = filtered_df[filtered_df[COL_DISPLAY_NAME] == selected_display_name]
    else:
        final_filtered_df = filtered_df

    tab1, tab2, tab3, tab4 = st.tabs(["績效總覽", "行為歸因矩陣", "績效賦能", "催員深度剖析"])

    with tab1:
        st.header("績效總覽 (KPI Overview)")
        if not final_filtered_df.empty:
            st.subheader("關鍵績效指標 (KPIs)")
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("總結案催回金額", f"${final_filtered_df[METRIC_TOTAL_COLLECTIONS].sum():,.0f}")
            kpi_cols[1].metric("總處理案件數", f"{final_filtered_df[METRIC_TOTAL_CASES].sum():,.0f}")
            kpi_cols[2].metric("平均撥打次數/案", f"{final_filtered_df[METRIC_AVG_TOUCHES].mean():.2f}")
            if st.session_state.get('has_connected_calls_data'):
                kpi_cols[3].metric("效率 (金額/有效通話)", f"${final_filtered_df[METRIC_EFFICIENCY_PER_CALL].mean():,.0f}")
            kpi_cols[4].metric("效率 (金額/小時)", f"${final_filtered_df[METRIC_EFFICIENCY_PER_HOUR].mean():,.0f}")
            st.markdown("---")
            
            if len(selected_groups) > 1 and selected_display_name == '全體':
                st.subheader("團隊績效比較")
                comparison_metric_options = [m for m in METRICS_TRANSLATION.keys() if METRICS_TRANSLATION[m] in filtered_df.columns]
                comparison_metric_display_name = st.selectbox("選擇比較指標", options=comparison_metric_options, index=0)
                comparison_metric_col_name = METRICS_TRANSLATION[comparison_metric_display_name]
                st.plotly_chart(create_group_comparison_chart(filtered_df, comparison_metric_col_name), use_container_width=True)

            st.markdown("---")
            st.subheader("績效趨勢分析")
            col1, col2 = st.columns([1, 3])
            with col1:
                trend_metric_display = st.selectbox("選擇趨勢指標", options=[k for k, v in METRICS_TRANSLATION.items() if v in agent_summary_df.columns])
                trend_metric_col = METRICS_TRANSLATION[trend_metric_display]
                time_unit_selection = st.radio("選擇時間顆粒度", ('日', '週'), horizontal=True)
                show_per_capita = st.checkbox("以人均值顯示趨勢", value=True)
            with col2:
                st.plotly_chart(create_performance_trend_chart(filtered_daily_df, trend_metric_col, time_unit_selection, group_sizes, show_per_capita), use_container_width=True)
        else:
            st.info("在選定的條件下沒有數據可顯示。")

    with tab2:
        st.header("行為歸因分析 (Behavioral Analysis)")
        if not filtered_df.empty:
            st.write("透過互動式績效矩陣，探索不同行為指標與成果指標之間的關聯性，並自動識別人員分佈象限。")
            metrics_options = [m for m in METRICS_TRANSLATION.values() if m in filtered_df.columns]
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("選擇 X 軸指標", options=metrics_options, index=metrics_options.index(METRIC_TOTAL_CALLS) if METRIC_TOTAL_CALLS in metrics_options else 0, key="ax_matrix", format_func=lambda x: REVERSE_METRICS_TRANSLATION.get(x, x))
            with col2:
                y_axis_default_index = metrics_options.index(METRIC_AVG_TOUCHES) if METRIC_AVG_TOUCHES in metrics_options else 0
                y_axis = st.selectbox("選擇 Y 軸指標", options=metrics_options, index=y_axis_default_index, key="ay_matrix", format_func=lambda x: REVERSE_METRICS_TRANSLATION.get(x, x))
            st.plotly_chart(create_performance_matrix_chart(filtered_df, x_axis, y_axis), use_container_width=True)
        else:
            st.info("在選定的條件下沒有數據可顯示。")

    with tab3:
        st.header("績效賦能 (Coaching Module)")
        st.write("選擇一位受輔導人員，並與一個或多個標竿人員組成的群組平均績效進行比較。")
        if not filtered_df.empty and len(filtered_df[COL_AGENT_ID].unique()) > 1:
            agent_select_df = filtered_df[[COL_AGENT_ID, COL_AGENT_NAME]].drop_duplicates()
            agent_select_df[COL_DISPLAY_NAME] = agent_select_df[COL_AGENT_ID] + " (" + agent_select_df[COL_AGENT_NAME] + ")"
            agent_id_map = agent_select_df.set_index(COL_DISPLAY_NAME)[COL_AGENT_ID]
            display_name_list = sorted(agent_select_df[COL_DISPLAY_NAME].unique())
            
            if 'benchmark_select' not in st.session_state:
                st.session_state.benchmark_select = []

            col1, col2 = st.columns(2)
            with col1:
                coached_display_name = st.selectbox("選擇受輔導人員", options=display_name_list, key="coach_select")
                coached_agent_id = agent_id_map[coached_display_name]
            with col2:
                benchmark_options = [name for name in display_name_list if name != coached_display_name]
                st.session_state.benchmark_select = [agent for agent in st.session_state.benchmark_select if agent in benchmark_options]
                
                benchmark_display_names = st.multiselect("選擇標竿群組 (可多選)", options=benchmark_options, key="benchmark_select")
                benchmark_agent_ids = [agent_id_map[name] for name in benchmark_display_names]
            
            scaling_method_display = st.radio(
                "雷達圖縮放方式",
                ('排名正規化 (相對表現)', '對數正規化 (絕對量體)'),
                index=0,
                horizontal=True,
                help="選擇雷達圖上各指標的縮放方式。\n\n- **排名正規化**: 根據人員在該指標的團隊排名進行顯示，專注於相對強弱項比較。\n- **對數正規化**: 根據指標的實際數值大小進行顯示，能反映量體上的差異。"
            )
            scaling_method = 'rank' if '排名' in scaling_method_display else 'log'

            if coached_agent_id and benchmark_agent_ids:
                fig, coached_raw, benchmark_avg_raw = create_coaching_radar_chart(
                    agent_summary_df, 
                    coached_agent_id, 
                    coached_display_name, 
                    benchmark_agent_ids,
                    scaling_method=scaling_method
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("量化比較矩陣")
                
                comparison_df_raw = pd.DataFrame({
                    "績效指標": [REVERSE_METRICS_TRANSLATION.get(i, i) for i in coached_raw.index],
                    f"受輔導人員 ({coached_display_name})": coached_raw.values,
                    "標竿群組平均": benchmark_avg_raw.values
                })
                comparison_df_raw['差距'] = comparison_df_raw.iloc[:, 1] - comparison_df_raw.iloc[:, 2]
                
                display_df = comparison_df_raw.copy()
                
                time_metric_name = REVERSE_METRICS_TRANSLATION[METRIC_TOTAL_TALK_TIME]
                if time_metric_name in display_df['績效指標'].values:
                    time_idx = display_df.index[display_df['績效指標'] == time_metric_name][0]
                    for col in [f"受輔導人員 ({coached_display_name})", "標竿群組平均", "差距"]:
                         display_df.loc[time_idx, col] = format_seconds_to_hms(display_df.loc[time_idx, col])
                
                def calculate_gap_colors(df_raw, df_display):
                    colors = {}
                    cost_metrics_display = [] 
                    for index, row in df_raw.iterrows():
                        metric = row['績效指標']
                        gap = row['差距']
                        color = ''
                        if pd.notna(gap):
                            if metric in cost_metrics_display:
                                color = 'green' if gap < 0 else 'red'
                            else:
                                color = 'green' if gap > 0 else 'red'
                        colors[df_display.index[index]] = f'color: {color}'
                    return colors
                
                gap_colors = calculate_gap_colors(comparison_df_raw, display_df)
                
                st.dataframe(
                    display_df.style.apply(lambda s: s.map(gap_colors), subset=['差距'])
                                  .format({
                                      f"受輔導人員 ({coached_display_name})": conditional_format,
                                      "標竿群組平均": conditional_format,
                                      "差距": conditional_format,
                                  }, na_rep='-'),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("請選擇一位受輔導人員和至少一位標竿組成員以生成圖表。")
        else:
            st.info("在選定的條件下，需要至少兩位催收員的數據才能進行比較。")

    with tab4:
        st.header("催員深度剖析 (Agent Deep Dive)")
        st.write("此模組深入分析單一催收員的工作模式與產出特徵，並與標竿群組進行比較，以利於識別其工作型態與效率。")

        if not filtered_df.empty:
            agent_select_df_deep_dive = filtered_df[[COL_AGENT_ID, COL_AGENT_NAME]].drop_duplicates()
            agent_select_df_deep_dive[COL_DISPLAY_NAME] = agent_select_df_deep_dive[COL_AGENT_ID] + " (" + agent_select_df_deep_dive[COL_AGENT_NAME] + ")"
            agent_id_map_deep_dive = agent_select_df_deep_dive.set_index(COL_DISPLAY_NAME)[COL_AGENT_ID]
            display_name_list_deep_dive = sorted(agent_select_df_deep_dive[COL_DISPLAY_NAME].unique())

            if display_name_list_deep_dive:
                # --- CODE MODIFICATION START ---
                
                # 動態建立分析指標選項
                analysis_metric_options = ['總撥打次數']
                if st.session_state.get('has_connected_calls_data'):
                    analysis_metric_options.append('總接通次數')
                if st.session_state.get('has_talk_time_data'):
                    analysis_metric_options.append('通話時長')
                
                metric_to_analyze = st.radio("選擇分析指標", options=analysis_metric_options, horizontal=True, key="deep_dive_radio")
                
                # 根據選擇的指標，動態設定閾值滑桿
                if metric_to_analyze == '總撥打次數':
                    max_val_col = COL_CALLS_IN_WINDOW
                    slider_label = f"設定納入分析的單案最大{metric_to_analyze}"
                    default_val = 40
                elif metric_to_analyze == '總接通次數':
                    max_val_col = COL_CONNECTED_CALLS_IN_WINDOW
                    slider_label = f"設定納入分析的單案最大{metric_to_analyze}"
                    default_val = 20
                else: # 通話時長
                    max_val_col = COL_TALK_TIME_IN_WINDOW
                    slider_label = f"設定納入分析的單案最大{metric_to_analyze}(秒)"
                    default_val = 600 # 預設10分鐘
                
                max_val = int(date_filtered_raw_df[max_val_col].max())
                threshold = st.slider(slider_label, min_value=1, max_value=max_val if max_val > 0 else 1, value=min(default_val, max_val) if max_val > 0 else 1, step=1, disabled=(max_val == 0))

                if 'deep_dive_benchmark' not in st.session_state:
                    st.session_state.deep_dive_benchmark = []

                col1, col2 = st.columns(2)
                with col1:
                    selected_agent_display_name = st.selectbox("選擇要深度剖析的催員", options=display_name_list_deep_dive, key="deep_dive_agent")
                    selected_agent_id = agent_id_map_deep_dive[selected_agent_display_name]
                with col2:
                    benchmark_options = [name for name in display_name_list_deep_dive if name != selected_agent_display_name]
                    st.session_state.deep_dive_benchmark = [agent for agent in st.session_state.deep_dive_benchmark if agent in benchmark_options]

                    benchmark_display_names = st.multiselect("選擇比較標竿群組 (可多選)", options=benchmark_options, key="deep_dive_benchmark")
                    benchmark_ids = [agent_id_map_deep_dive[name] for name in benchmark_display_names]

                # 建立一個分頁或欄位來並排顯示圖表
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # 使用重構後的函式
                    st.plotly_chart(create_agent_deep_dive_distribution_chart(date_filtered_raw_df, selected_agent_id, benchmark_ids, selected_agent_display_name, threshold, metric_to_analyze), use_container_width=True)
                
                with chart_col2:
                    st.plotly_chart(create_payment_distribution_chart(date_filtered_raw_df, selected_agent_id, benchmark_ids, selected_agent_display_name), use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("績效總結比較")

                # 1. 取得並顯示被分析人員的數據
                agent_summary_data = agent_summary_df[agent_summary_df[COL_AGENT_ID] == selected_agent_id]
                
                if not agent_summary_data.empty:
                    st.markdown(f"**分析人員: {selected_agent_display_name}**")
                    kpi_cols = st.columns(3)
                    kpi_cols[0].metric("期間催回總額", f"${agent_summary_data[METRIC_TOTAL_COLLECTIONS].iloc[0]:,.0f}")
                    kpi_cols[1].metric("期間處理案件數", f"{agent_summary_data[METRIC_TOTAL_CASES].iloc[0]:,.0f}")
                    
                    # 顯示與分析指標相關的平均值
                    if st.session_state.get('has_talk_time_data'):
                         avg_talk_time = (agent_summary_data[METRIC_TOTAL_TALK_TIME].iloc[0] / agent_summary_data[METRIC_TOTAL_CASES].iloc[0]) if agent_summary_data[METRIC_TOTAL_CASES].iloc[0] > 0 else 0
                         kpi_cols[2].metric("平均通話時長/案", f"{avg_talk_time:.2f} 秒")
                    else:
                         kpi_cols[2].metric("平均撥打次數/案", f"{agent_summary_data[METRIC_AVG_TOUCHES].iloc[0]:.2f}")

                else:
                    st.warning("找不到該人員的績效摘要數據。")

                # 2. 如果有選擇標竿群組，則計算並顯示其平均數據
                if benchmark_ids:
                    benchmark_summary_df = agent_summary_df[agent_summary_df[COL_AGENT_ID].isin(benchmark_ids)]
                    
                    if not benchmark_summary_df.empty:
                        st.markdown("---") # 視覺分隔線
                        st.markdown("**標竿群組平均績效**")
                        
                        # 計算標竿群組的人均指標
                        avg_collections = benchmark_summary_df[METRIC_TOTAL_COLLECTIONS].mean()
                        avg_cases = benchmark_summary_df[METRIC_TOTAL_CASES].mean()
                        avg_touches = benchmark_summary_df[METRIC_AVG_TOUCHES].mean()
                        avg_connected_touches = benchmark_summary_df[METRIC_AVG_CONNECTED_TOUCHES].mean()
                        avg_talk_time_per_case = (benchmark_summary_df[METRIC_TOTAL_TALK_TIME].sum() / benchmark_summary_df[METRIC_TOTAL_CASES].sum()) if benchmark_summary_df[METRIC_TOTAL_CASES].sum() > 0 else 0


                        # 顯示指標
                        kpi_cols_bench = st.columns(3)
                        kpi_cols_bench[0].metric("人均催回總額", f"${avg_collections:,.0f}")
                        kpi_cols_bench[1].metric("人均處理案件數", f"{avg_cases:.2f}")

                        if st.session_state.get('has_talk_time_data'):
                            kpi_cols_bench[2].metric("人均通話時長/案", f"{avg_talk_time_per_case:.2f} 秒")
                        else:
                            kpi_cols_bench[2].metric("人均撥打次數/案", f"{avg_touches:.2f}")

                # --- CODE MODIFICATION END ---
            else:
                st.info("請在側邊欄選擇包含催員的組別以進行深度剖析。")
        else:
            st.info("在選定的條件下沒有數據可顯示。")
else:
    st.error("數據載入失敗，儀表板無法啟動。請檢查主控台中的錯誤訊息或 `secrets.toml` 設定。")
