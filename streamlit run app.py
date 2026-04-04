import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ページ設定
st.set_page_config(page_title="⚾ NPB データ分析ダッシュボード", layout="wide")

# =====================================================================
# セッションステート（お気に入り機能と入力状態の保存）
# =====================================================================
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'input_pitcher' not in st.session_state:
    st.session_state.input_pitcher = ""
if 'input_batter' not in st.session_state:
    st.session_state.input_batter = ""

# =====================================================================
# データ前処理・クレンジング関数
# =====================================================================
def add_advanced_stats(df):
    """結果のテキストからフラグを生成し、打球方向などを高速で分類する"""
    df = df.copy()
    res = df['結果'].fillna('')
    
    # 1. 基本的なフラグの作成（int8型を使ってメモリを極限まで節約）
    df['is_PA'] = np.int8(1)
    df['is_BB_HBP'] = res.str.contains('四球|死球|敬遠', na=False)
    df['is_SAC'] = res.str.contains('犠打|犠牲バント|犠飛|犠牲フライ', na=False)
    df['is_SF'] = res.str.contains('犠飛|犠牲フライ', na=False)
    df['is_SO'] = res.str.contains('三振', na=False)
    
    df['is_H'] = res.str.contains('安打|ヒット|二塁打|ツーベース|三塁打|スリーベース|本塁打|ホームラン', na=False)
    df['is_2B'] = res.str.contains('二塁打|ツーベース', na=False)
    df['is_3B'] = res.str.contains('三塁打|スリーベース', na=False)
    df['is_HR'] = res.str.contains('本塁打|ホームラン', na=False)
    df['is_1B'] = df['is_H'] & ~df['is_2B'] & ~df['is_3B'] & ~df['is_HR']
    
    df['is_AB'] = df['is_PA'] - df['is_BB_HBP'].astype('int8') - df['is_SAC'].astype('int8')
    df['TB'] = df['is_1B'].astype('int8')*1 + df['is_2B'].astype('int8')*2 + df['is_3B'].astype('int8')*3 + df['is_HR'].astype('int8')*4
    
    # 2. 純粋な結果（方向抜きの詳細結果）
    res_clean = np.where(res.str.contains('三振'), '三振',
                np.where(res.str.contains('四球|死球|敬遠'), '四死球',
                np.where(res.str.contains('犠打|犠牲バント'), '犠打',
                np.where(res.str.contains('犠飛|犠牲フライ'), '犠飛',
                np.where(res.str.contains('本塁打|ホームラン'), '本塁打',
                np.where(res.str.contains('三塁打|スリーベース'), '三塁打',
                np.where(res.str.contains('二塁打|ツーベース'), '二塁打',
                np.where(res.str.contains('安打|ヒット'), '単打',
                np.where(res.str.contains('併殺'), '併殺打',
                np.where(res.str.contains('ゴロ'), 'ゴロ',
                np.where(res.str.contains('飛|フライ'), 'フライ',
                np.where(res.str.contains('直|ライナー'), 'ライナー',
                np.where(res.str.contains('邪飛'), 'ファウルフライ', 'その他')))))))))))))
    df['純粋な結果'] = res_clean
    
    # 3. 扇形グラウンド用 30度ごとの方向判定
    is_foul = res.str.contains('邪|ファウル', na=False)
    is_left = res.str.contains('左|三|遊|レフト|サード|ショート', na=False)
    is_right = res.str.contains('右|一|二|ライト|ファースト|セカンド', na=False)
    is_center = res.str.contains('投|捕|ピッチャー|キャッチャー', na=False) | (res.str.contains('中|センター', na=False) & ~res.str.contains('左中間|右中間', na=False))
    
    d_cat = np.where(is_foul & is_left, '左ファウル',
            np.where(is_foul & is_right, '右ファウル',
            np.where(is_foul, '後ろファウル',
            np.where(~is_foul & is_left, '左方向',
            np.where(~is_foul & is_right, '右方向',
            np.where(~is_foul & is_center, 'センター', '不明'))))))
            
    df['打球方向_30'] = d_cat
    
    if '打点' not in df.columns:
        df['打点'] = 0
        
    return df

# 🌟メモリ節約のためキャッシュは最新の1つだけ保持する
@st.cache_data(max_entries=1)
def load_and_clean_data():
    try:
        df = pd.read_parquet("all_matchup_data.parquet")
    except FileNotFoundError:
        return None, "データファイル (all_matchup_data.parquet) が見つかりません。"

    if '年度' not in df.columns:
        df['年度'] = pd.to_datetime(df['日付']).dt.year

    aliases_file = "npb_official_aliases.json"
    if os.path.exists(aliases_file):
        with open(aliases_file, "r", encoding="utf-8") as f:
            player_aliases = json.load(f)
        df['投手'] = df['投手'].replace(player_aliases)
        df['打者'] = df['打者'].replace(player_aliases)
        
        manual_aliases = {"Ｔ－岡田": "T-岡田", "Ｔ-岡田": "T-岡田", "Ｃ．Ｃ．メルセデス": "C.C.メルセデス", "Ｒ．マルティネス": "R.マルティネス", "Ｌ．モイネロ": "L.モイネロ"}
        df['投手'] = df['投手'].replace(manual_aliases)
        df['打者'] = df['打者'].replace(manual_aliases)

    df.loc[(df['投手'] == '拓也') & (df['年度'] <= 2016) & (df['投手チーム'] == 'ソフトバンク'), '投手'] = '甲斐拓也'
    df.loc[(df['打者'] == '拓也') & (df['年度'] <= 2016) & (df['打者チーム'] == 'ソフトバンク'), '打者'] = '甲斐拓也'
    df.loc[(df['投手'] == '拓也') & (df['年度'] >= 2026) & (df['投手チーム'] == 'ヤクルト'), '投手'] = '矢崎拓也'
    df.loc[(df['打者'] == '拓也') & (df['年度'] >= 2026) & (df['打者チーム'] == 'ヤクルト'), '打者'] = '矢崎拓也'

    df = add_advanced_stats(df)
    
    # 🌟超重要：オブジェクト（文字列）をカテゴリ型に変換し、RAM消費量を約1/10に削減
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    return df, None

@st.cache_data(max_entries=1)
def load_directories():
    files = glob.glob("player_directory_*.csv")
    df_list = []
    for f in files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if df_list:
        return pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['チーム', '選手名'])
    return pd.DataFrame()

def format_rate(val):
    if val >= 1.0: return f"{val:.3f}"
    return f".{str(int(val * 1000)).zfill(3)}"

# =====================================================================
# UI 構築とサイドバー
# =====================================================================
st.title("⚾ NPB 超高度データ分析ダッシュボード")

with st.spinner("データを準備しています..."):
    df, error_msg = load_and_clean_data()
    df_dir = load_directories()

if error_msg:
    st.error(error_msg)
    st.stop()

st.sidebar.header("🔍 分析設定")

team_options = ["全球団", "巨人", "阪神", "中日", "DeNA", "広島", "ヤクルト", "オリックス", "ロッテ", "ソフトバンク", "楽天", "西武", "日本ハム"]
selected_team = st.sidebar.selectbox("球団で絞り込み", team_options)

if selected_team != "全球団" and not df_dir.empty:
    team_players = df_dir[df_dir['チーム'] == selected_team]['選手名'].unique()
    with st.sidebar.expander(f"💡 {selected_team} の所属選手一覧"):
