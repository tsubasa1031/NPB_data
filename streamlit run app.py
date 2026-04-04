import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# ページ設定
st.set_page_config(page_title="⚾ NPB データ分析ダッシュボード", layout="wide")

# =====================================================================
# 1. データの読み込みと前処理（キャッシュして爆速化）
# =====================================================================
@st.cache_data
def load_and_clean_data():
    # 1. メインデータの読み込み
    try:
        df = pd.read_parquet("all_matchup_data.parquet")
    except FileNotFoundError:
        return None, "データファイル (all_matchup_data.parquet) が見つかりません。"

    # 年度列がない場合は作成
    if '年度' not in df.columns:
        df['年度'] = pd.to_datetime(df['日付']).dt.year

    # 2. 自動生成した名寄せ辞書の読み込みと適用
    aliases_file = "npb_official_aliases.json"
    if os.path.exists(aliases_file):
        with open(aliases_file, "r", encoding="utf-8") as f:
            player_aliases = json.load(f)
            
        # 辞書を使って一括置換
        df['投手'] = df['投手'].replace(player_aliases)
        df['打者'] = df['打者'].replace(player_aliases)
        
        # 手動で追加したいよくある表記ゆれ（必要に応じて追加）
        manual_aliases = {
            "Ｔ－岡田": "T-岡田", "Ｔ-岡田": "T-岡田",
            "Ｃ．Ｃ．メルセデス": "C.C.メルセデス", "Ｒ．マルティネス": "R.マルティネス",
            "Ｌ．モイネロ": "L.モイネロ"
        }
        df['投手'] = df['投手'].replace(manual_aliases)
        df['打者'] = df['打者'].replace(manual_aliases)

    # 3. ⚠️ 特殊な名寄せ処理（同名別人の「拓也」問題などを手動で分離）
    # パターンA：2016年以前で、チームがソフトバンクの「拓也」は「甲斐拓也」
    df.loc[(df['投手'] == '拓也') & (df['年度'] <= 2016) & (df['投手チーム'] == 'ソフトバンク'), '投手'] = '甲斐拓也'
    df.loc[(df['打者'] == '拓也') & (df['年度'] <= 2016) & (df['打者チーム'] == 'ソフトバンク'), '打者'] = '甲斐拓也'
    
    # パターンB：2026年以降で、チームがヤクルトの「拓也」は「矢崎拓也」
    df.loc[(df['投手'] == '拓也') & (df['年度'] >= 2026) & (df['投手チーム'] == 'ヤクルト'), '投手'] = '矢崎拓也'
    df.loc[(df['打者'] == '拓也') & (df['年度'] >= 2026) & (df['打者チーム'] == 'ヤクルト'), '打者'] = '矢崎拓也'

    return df, None

# =====================================================================
# UI 構築
# =====================================================================
st.title("⚾ NPB 投手 vs 打者 分析ダッシュボード")

with st.spinner("データを読み込み、名寄せ処理を実行中..."):
    df, error_msg = load_and_clean_data()

if error_msg:
    st.error(error_msg)
    st.stop()

# ---------------------------------------------------------
# サイドバーの設定
# ---------------------------------------------------------
st.sidebar.header("🔍 分析設定")

# モード選択
analysis_mode = st.sidebar.radio(
    "分析モードを選択してください",
    ["投手視点で分析", "打者視点で分析", "特定の対戦 (投手 vs 打者)"]
)

# 年度フィルター
years = sorted(df['年度'].unique(), reverse=True)
year_options = ["全年度"] + list(years)
selected_year = st.sidebar.selectbox("表示する年度", year_options)

# データ絞り込み（年度）
filtered_df = df.copy()
if selected_year != "全年度":
    filtered_df = filtered_df[filtered_df['年度'] == selected_year]

# リストの作成
all_pitchers = sorted(filtered_df['投手'].dropna().unique())
all_batters = sorted(filtered_df['打者'].dropna().unique())

st.sidebar.markdown("---")

# ---------------------------------------------------------
# メイン画面の表示
# ---------------------------------------------------------
if analysis_mode == "投手視点で分析":
    selected_pitcher = st.sidebar.selectbox("投手を選択", all_pitchers)
    target_df = filtered_df[filtered_df['投手'] == selected_pitcher]
    st.subheader(f"📊 {selected_pitcher} 投手のデータ ({selected_year})")

elif analysis_mode == "打者視点で分析":
    selected_batter = st.sidebar.selectbox("打者を選択", all_batters)
    target_df = filtered_df[filtered_df['打者'] == selected_batter]
    st.subheader(f"📊 {selected_batter} 打者のデータ ({selected_year})")

else: # 特定の対戦
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_pitcher = st.selectbox("投手", all_pitchers)
    with col2:
        # その投手と対戦経験のある打者だけに絞る
        opponents = sorted(filtered_df[filtered_df['投手'] == selected_pitcher]['打者'].dropna().unique())
        if not opponents:
            opponents = ["データなし"]
        selected_batter = st.selectbox("打者", opponents)
        
    target_df = filtered_df[(filtered_df['投手'] == selected_pitcher) & (filtered_df['打者'] == selected_batter)]
    st.subheader(f"⚔️ {selected_pitcher} vs {selected_batter} ({selected_year})")


# --- 結果の表示 ---
if target_df.empty:
    st.warning("該当するデータがありません。条件を変更してください。")
else:
    # 1. 基本指標の表示
    total_pa = len(target_df) # 打席数
    
    # 安打数と打数の計算（簡易的な計算）
    hits = len(target_df[target_df['打席結果'] == '安打'])
    # 凡打・三振など打数がカウントされるもの
    at_bats = len(target_df[target_df['打席結果'].isin(['安打', '凡打', '三振'])])
    
    avg = hits / at_bats if at_bats > 0 else 0.000
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総対戦打席数", f"{total_pa} 打席")
    col2.metric("打数", f"{at_bats} 打数")
    col3.metric("安打数", f"{hits} 本")
    col4.metric("打率", f".{str(int(avg * 1000)).zfill(3)}")

    st.markdown("---")

    # 2. グラフの表示
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 🎯 打席結果の内訳")
        result_counts = target_df['打席結果'].value_counts().reset_index()
        result_counts.columns = ['打席結果', '回数']
        
        # Plotlyを使った綺麗な円グラフ
        fig1 = px.pie(result_counts, values='回数', names='打席結果', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        st.markdown("#### ⚾ 詳細な結果 (三振、ゴロ、フライなど)")
        detail_counts = target_df['結果'].value_counts().head(10).reset_index()
        detail_counts.columns = ['結果', '回数']
        
        # Plotlyを使った横向き棒グラフ
        fig2 = px.bar(detail_counts, x='回数', y='結果', orientation='h', color='回数', color_continuous_scale='Blues')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    # 3. 生データのテーブル表示
    st.markdown("#### 📝 対戦履歴データ")
    
    # 表示する列を整理
    display_cols = ['日付', '投手チーム', '投手', '打者チーム', '打者', 'イニング数', '表裏', '結果']
    available_cols = [col for col in display_cols if col in target_df.columns]
    
    st.dataframe(target_df[available_cols].sort_values('日付', ascending=False), use_container_width=True)
