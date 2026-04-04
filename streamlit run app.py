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
        try:
            df_list.append(pd.read_csv(f))
        except:
            pass
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
        st.write("、".join(sorted(team_players)))

filtered_df = df.copy()
if selected_team != "全球団":
    filtered_df = filtered_df[(filtered_df['投手チーム'] == selected_team) | (filtered_df['打者チーム'] == selected_team)]

years = sorted(filtered_df['年度'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("表示する年度", ["全年度"] + list(years))
if selected_year != "全年度":
    filtered_df = filtered_df[filtered_df['年度'] == selected_year]

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
            if analysis_mode == "投手視点で分析":
                st.session_state.input_pitcher = fav
            elif analysis_mode == "打者視点で分析":
                st.session_state.input_batter = fav
            else:
                if not st.session_state.input_pitcher:
                    st.session_state.input_pitcher = fav
                else:
                    st.session_state.input_batter = fav
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

    st.markdown("#### 📈 基本指標")
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

    # --- 📊 円グラフと詳細棒グラフ（復活） ---
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 🎯 打席結果の割合")
        result_counts = target_df['打席結果'].value_counts().reset_index()
        result_counts.columns = ['打席結果', '回数']
        fig1 = px.pie(result_counts, values='回数', names='打席結果', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        st.markdown("#### ⚾ 詳細な結果 (方向なし)")
        detail_counts = target_df['純粋な結果'].value_counts().head(10).reset_index()
        detail_counts.columns = ['結果', '回数']
        fig2 = px.bar(detail_counts, x='回数', y='結果', orientation='h', color='回数', color_continuous_scale='Blues')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- 🏟️ 打球方向の扇形チャート (グラウンド型・バブル配置) ---
    st.markdown("#### 🏟️ 飛びやすい打球方向 (30度ごとの扇形分割)")
    
    dir_counts = target_df['打球方向_30'].value_counts()
    d_left_foul = dir_counts.get('左ファウル', 0)
    d_left = dir_counts.get('左方向', 0)
    d_center = dir_counts.get('センター', 0)
    d_right = dir_counts.get('右方向', 0)
    d_right_foul = dir_counts.get('右ファウル', 0)

    total_dir = d_left_foul + d_left + d_center + d_right + d_right_foul
    
    if total_dir > 0:
        fig_dir = go.Figure()

        # 1. 外野芝生（扇形 15度〜165度）
        theta_outer = np.linspace(15, 165, 100) * np.pi / 180
        x_outer = 1.7 * np.cos(theta_outer)
        y_outer = 1.7 * np.sin(theta_outer)
        fig_dir.add_trace(go.Scatter(
            x=[0] + list(x_outer) + [0], 
            y=[0] + list(y_outer) + [0],
            mode='lines', fill='toself', fillcolor='#e0f2e0',
            line=dict(color='white', width=3), hoverinfo='skip'
        ))
        
        # 2. 内野土（扇形）
        theta_inner = np.linspace(15, 165, 100) * np.pi / 180
        x_inner = 0.85 * np.cos(theta_inner)
        y_inner = 0.85 * np.sin(theta_inner)
        fig_dir.add_trace(go.Scatter(
            x=[0] + list(x_inner) + [0], 
            y=[0] + list(y_inner) + [0],
            mode='lines', fill='toself', fillcolor='#f2e6d9',
            line=dict(color='white', width=2), hoverinfo='skip'
        ))

        # 3. 扇形の区切り線（45度, 75度, 105度, 135度）
        for angle in [45, 75, 105, 135]:
            rad = angle * np.pi / 180
            # ファウルライン(45, 135)は実線、中間線(75, 105)は点線
            dash_style = 'dot' if angle in [75, 105] else 'solid'
            fig_dir.add_trace(go.Scatter(
                x=[0, 1.7 * np.cos(rad)],
                y=[0, 1.7 * np.sin(rad)],
                mode='lines',
                line=dict(color='white', width=2, dash=dash_style),
                hoverinfo='skip'
            ))

        # 4. バブルとテキストの配置
        sizes = [d_left_foul, d_left, d_center, d_right, d_right_foul]
        max_size = max(sizes) if max(sizes) > 0 else 1
        marker_sizes = [(s/max_size)*45 + 30 for s in sizes]

        angles_center = [150, 120, 90, 60, 30]
        labels = ["左ファウル", "左方向", "センター", "右方向", "右ファウル"]
        colors = ['#CFD8DC', '#FF9999', '#99CCFF', '#FFCC99', '#CFD8DC']
        
        x_centers = []
        y_centers = []
        text_labels = []
        
        for cnt, angle, label in zip(sizes, angles_center, labels):
            rad = angle * np.pi / 180
            r_pos = 1.3 # テキストを配置する半径
            x_centers.append(r_pos * np.cos(rad))
            y_centers.append(r_pos * np.sin(rad))
            pct = (cnt / total_dir) * 100
            # 0本の場合は表示をスッキリさせる
            if cnt > 0:
                text_labels.append(f"{label}<br>{cnt}本<br>({pct:.1f}%)")
            else:
                text_labels.append("")

        fig_dir.add_trace(go.Scatter(
            x=x_centers,
            y=y_centers,
            mode='markers+text',
            marker=dict(
                size=marker_sizes,
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=text_labels,
            textposition='middle center',
            textfont=dict(color='black', size=13, weight='bold'),
            hoverinfo='none'
        ))

        fig_dir.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.8, 1.8]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.9]),
            plot_bgcolor='white',
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            height=450
        )
        
        st.plotly_chart(fig_dir, use_container_width=True)
    else:
        st.info("※ この条件ではグラウンドに飛んだ打球のデータがありません（三振や四死球のみなど）")
