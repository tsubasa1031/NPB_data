import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.graph_objects as go
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
    """結果のテキストから打数や安打などのフラグを生成し、打球方向を30度ずつに分類する"""
    df = df.copy()
    # 1. 基本的なフラグの作成
    df['is_PA'] = 1
    df['is_BB_HBP'] = df['結果'].str.contains('四球|死球|敬遠', na=False)
    df['is_SAC'] = df['結果'].str.contains('犠打|犠牲バント|犠飛|犠牲フライ', na=False)
    df['is_SF'] = df['結果'].str.contains('犠飛|犠牲フライ', na=False)
    df['is_SO'] = df['結果'].str.contains('三振', na=False)
    
    df['is_H'] = df['結果'].str.contains('安打|ヒット|二塁打|ツーベース|三塁打|スリーベース|本塁打|ホームラン', na=False)
    df['is_2B'] = df['結果'].str.contains('二塁打|ツーベース', na=False)
    df['is_3B'] = df['結果'].str.contains('三塁打|スリーベース', na=False)
    df['is_HR'] = df['結果'].str.contains('本塁打|ホームラン', na=False)
    df['is_1B'] = df['is_H'] & ~df['is_2B'] & ~df['is_3B'] & ~df['is_HR']
    
    df['is_AB'] = df['is_PA'] - df['is_BB_HBP'].astype(int) - df['is_SAC'].astype(int)
    df['TB'] = df['is_1B'].astype(int)*1 + df['is_2B'].astype(int)*2 + df['is_3B'].astype(int)*3 + df['is_HR'].astype(int)*4
    
    # 2. 打球方向と内訳用の分類
    def get_dir_and_res(res):
        res = str(res)
        # 結果カテゴリの内訳
        if '本塁打' in res or 'ホームラン' in res: r_cat = '本塁打'
        elif any(w in res for w in ['二塁打', '三塁打', 'ツーベース', 'スリーベース']): r_cat = '長打(二・三塁打)'
        elif '安打' in res or 'ヒット' in res: r_cat = '単打'
        elif '犠' in res: r_cat = '犠打/犠飛'
        elif '併殺' in res: r_cat = '併殺打'
        elif '三振' in res or any(w in res for w in ['四球', '死球', '敬遠']): r_cat = '非打球(三振/四死球)'
        else: r_cat = '凡打/アウト'
        
        # 扇形グラウンド用 30度ごとの方向判定
        if '邪' in res: 
            if any(w in res for w in ['左', '三']): d_cat = '左ファウル'
            elif any(w in res for w in ['右', '一']): d_cat = '右ファウル'
            else: d_cat = '後ろファウル'
        else:
            if any(w in res for w in ['左', '三', '遊']): d_cat = '左方向'
            elif any(w in res for w in ['中', '投', '捕']): d_cat = 'センター'
            elif any(w in res for w in ['右', '一', '二']): d_cat = '右方向'
            else: d_cat = '不明'
            
        return pd.Series([d_cat, r_cat])
        
    df[['打球方向_30', '打球結果_詳細']] = df['結果'].apply(get_dir_and_res)
    if '打点' not in df.columns:
        df['打点'] = 0
        
    return df

@st.cache_data
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
    return df, None

@st.cache_data
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
    """打率などを .333 のようなフォーマットにする"""
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

# --- 1. 球団絞り込み ---
team_options = ["全球団", "巨人", "阪神", "中日", "DeNA", "広島", "ヤクルト", "オリックス", "ロッテ", "ソフトバンク", "楽天", "西武", "日本ハム"]
selected_team = st.sidebar.selectbox("球団で絞り込み", team_options)

if selected_team != "全球団" and not df_dir.empty:
    team_players = df_dir[df_dir['チーム'] == selected_team]['選手名'].unique()
    with st.sidebar.expander(f"💡 {selected_team} の所属選手一覧"):
        st.write("、".join(sorted(team_players)))

filtered_df = df.copy()
if selected_team != "全球団":
    filtered_df = filtered_df[(filtered_df['投手チーム'] == selected_team) | (filtered_df['打者チーム'] == selected_team)]

years = sorted(filtered_df['年度'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("表示する年度", ["全年度"] + list(years))
if selected_year != "全年度":
    filtered_df = filtered_df[filtered_df['年度'] == selected_year]

# --- 2. 予測変換（インクリメンタルサーチ）付きの選手選択 ---
all_pitchers = sorted(filtered_df['投手'].dropna().unique())
all_batters = sorted(filtered_df['打者'].dropna().unique())
all_players = sorted(list(set(all_pitchers) | set(all_batters)))
player_options = [""] + all_players

analysis_mode = st.sidebar.radio("分析モード", ["投手視点で分析", "打者視点で分析", "特定の対戦 (投手 vs 打者)"])

if analysis_mode == "投手視点で分析":
    idx = player_options.index(st.session_state.input_pitcher) if st.session_state.input_pitcher in player_options else 0
    selected_pitcher = st.sidebar.selectbox("投手を選択 (タイピングで予測変換)", player_options, index=idx)
    st.session_state.input_pitcher = selected_pitcher
elif analysis_mode == "打者視点で分析":
    idx = player_options.index(st.session_state.input_batter) if st.session_state.input_batter in player_options else 0
    selected_batter = st.sidebar.selectbox("打者を選択 (タイピングで予測変換)", player_options, index=idx)
    st.session_state.input_batter = selected_batter
else:
    st.sidebar.caption("※タイピングで予測変換されます")
    idx_p = player_options.index(st.session_state.input_pitcher) if st.session_state.input_pitcher in player_options else 0
    idx_b = player_options.index(st.session_state.input_batter) if st.session_state.input_batter in player_options else 0
    selected_pitcher = st.sidebar.selectbox("投手", player_options, index=idx_p)
    selected_batter = st.sidebar.selectbox("打者", player_options, index=idx_b)
    st.session_state.input_pitcher = selected_pitcher
    st.session_state.input_batter = selected_batter

# --- 3. お気に入り機能 ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ お気に入り選手")
new_fav = st.sidebar.text_input("お気に入りに登録", placeholder="選手名を入力してEnter")

if st.sidebar.button("追加する"):
    if new_fav and new_fav not in st.session_state.favorites:
        st.session_state.favorites.append(new_fav)
        st.rerun()

if st.session_state.favorites:
    st.sidebar.caption("↓ クリックで入力欄に自動セット")
    for fav in st.session_state.favorites:
        cols = st.sidebar.columns([3, 1])
        if cols[0].button(f"👤 {fav}", key=f"set_{fav}"):
            if analysis_mode == "投手視点で分析": st.session_state.input_pitcher = fav
            elif analysis_mode == "打者視点で分析": st.session_state.input_batter = fav
            else:
                if not st.session_state.input_pitcher: st.session_state.input_pitcher = fav
                else: st.session_state.input_batter = fav
            st.rerun()
        if cols[1].button("✖", key=f"del_{fav}"):
            st.session_state.favorites.remove(fav)
            st.rerun()

# =====================================================================
# データ抽出とスタッツ表示
# =====================================================================
if analysis_mode == "投手視点で分析":
    if not selected_pitcher: st.stop()
    target_df = filtered_df[filtered_df['投手'] == selected_pitcher]
    title_text = f"📊 {selected_pitcher} 投手のデータ"
elif analysis_mode == "打者視点で分析":
    if not selected_batter: st.stop()
    target_df = filtered_df[filtered_df['打者'] == selected_batter]
    title_text = f"📊 {selected_batter} 打者のデータ"
else:
    if not selected_pitcher or not selected_batter: st.stop()
    target_df = filtered_df[(filtered_df['投手'] == selected_pitcher) & (filtered_df['打者'] == selected_batter)]
    title_text = f"⚔️ {selected_pitcher} vs {selected_batter}"

st.subheader(f"{title_text} ({selected_year})")

if target_df.empty:
    st.warning("該当する対戦データがありません。")
else:
    # --- 📈 全基本スタッツの計算 ---
    stats = {
        'PA': target_df['is_PA'].sum(),
        'AB': target_df['is_AB'].sum(),
        'H': target_df['is_H'].sum(),
        'HR': target_df['is_HR'].sum(),
        'RBI': target_df['打点'].sum(),
        'SO': target_df['is_SO'].sum(),
        'BB': target_df['is_BB_HBP'].sum(),
    }
    ab, h, bb, sf = stats['AB'], stats['H'], stats['BB'], target_df['is_SF'].sum()
    tb = target_df['TB'].sum()
    
    stats['AVG'] = h / ab if ab > 0 else 0.0
    stats['OBP'] = (h + bb) / (ab + bb + sf) if (ab + bb + sf) > 0 else 0.0
    stats['SLG'] = tb / ab if ab > 0 else 0.0
    stats['OPS'] = stats['OBP'] + stats['SLG']

    # --- 📈 スタッツの表示 ---
    st.markdown("#### 📈 基本指標")
    
    # カスタムCSSでMetricsを見やすくする
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("打席", stats['PA'])
    c2.metric("打数", stats['AB'])
    c3.metric("安打", stats['H'])
    c4.metric("本塁打", stats['HR'])
    c5.metric("打点", stats['RBI'])
    c6.metric("三振", stats['SO'])
    c7.metric("四死球", stats['BB'])

    st.markdown("<br>", unsafe_allow_html=True)
    c8, c9, c10, c11 = st.columns(4)
    c8.metric("打率 (AVG)", format_rate(stats['AVG']))
    c9.metric("出塁率 (OBP)", format_rate(stats['OBP']))
    c10.metric("長打率 (SLG)", format_rate(stats['SLG']))
    c11.metric("OPS", format_rate(stats['OPS']))

    st.markdown("---")

    # --- 🏟️ 打球方向と内訳の扇形チャート (Plotly Barpolar) ---
    st.markdown("#### 🏟️ 打球方向と結果の内訳 (30度ごとの扇形分割)")
    
    dir_order = ['左ファウル', '左方向', 'センター', '右方向', '右ファウル']
    theta_vals = [150, 120, 90, 60, 30] # 15度〜165度の中で各30度の中心角
    
    plot_df = target_df[~target_df['打球方向_30'].isin(['不明', '後ろファウル']) & (target_df['打球結果_詳細'] != '非打球(三振/四死球)')]
    
    if plot_df.empty:
        st.info("※ この条件ではグラウンドに飛んだ打球のデータがありません（三振や四死球のみなど）")
    else:
        agg_df = plot_df.groupby(['打球方向_30', '打球結果_詳細']).size().reset_index(name='回数')
        fig = go.Figure()
        
        # 内訳の順番とカラー設定（下から積み上げ）
        res_order = ['凡打/アウト', '併殺打', '犠打/犠飛', '単打', '長打(二・三塁打)', '本塁打']
        colors = ['#CFD8DC', '#90A4AE', '#81C784', '#4FC3F7', '#FF9800', '#E53935']
        
        for r_type, color in zip(res_order, colors):
            sub_df = agg_df[agg_df['打球結果_詳細'] == r_type]
            r_vals, texts = [], []
            for d in dir_order:
                val = sub_df[sub_df['打球方向_30'] == d]['回数'].sum()
                r_vals.append(val)
                # 0の場合はテキスト非表示にしてスッキリさせる
                texts.append(f"{val}" if val > 0 else "")
                
            fig.add_trace(go.Barpolar(
                r=r_vals,
                theta=theta_vals,
                width=[30]*5,
                name=r_type,
                marker_color=color,
                marker_line_color='white',
                marker_line_width=1,
                text=texts,
                textposition='inside',
                textfont=dict(color='white' if r_type in ['本塁打', '長打(二・三塁打)', '併殺打'] else 'black', size=14),
                hoverinfo="name+r"
            ))

        fig.update_layout(
            polar=dict(
                sector=[15, 165], # フェアゾーン90度＋両ファウル30度ずつの合計150度
                hole=0.2,         # ホームベース付近を空ける
                radialaxis=dict(showticklabels=False, ticks=''),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=theta_vals,
                    ticktext=['左ファウル<br>(135-165°)', '左方向<br>(105-135°)', 'センター<br>(75-105°)', '右方向<br>(45-75°)', '右ファウル<br>(15-45°)'],
                    direction='counterclockwise',
                    tickfont=dict(size=13, weight="bold")
                )
            ),
            barmode='stack', # 積み上げモード
            height=500,
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=14))
        )
        st.plotly_chart(fig, use_container_width=True)
