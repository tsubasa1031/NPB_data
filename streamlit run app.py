import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go

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
def clean_detail_result(text):
    """詳細結果から「左」「右」などの方向を排除し、純粋な結果に変換する関数"""
    if pd.isna(text): return text
    text = str(text)
    if '三振' in text: return '三振'
    if any(w in text for w in ['四球', '死球', '敬遠']): return '四死球'
    if '犠打' in text or '犠牲バント' in text: return '犠打'
    if '犠飛' in text: return '犠飛'
    if '本塁打' in text or 'ホームラン' in text: return '本塁打'
    if '三塁打' in text: return '三塁打'
    if '二塁打' in text: return '二塁打'
    if '安打' in text or 'ヒット' in text: return '単打'
    if '併殺' in text: return '併殺打'
    if 'ゴロ' in text: return 'ゴロ'
    if '飛' in text or 'フライ' in text: return 'フライ'
    if '直' in text or 'ライナー' in text: return 'ライナー'
    if '邪飛' in text: return 'ファウルフライ'
    return 'その他'

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_parquet("all_matchup_data.parquet")
    except FileNotFoundError:
        return None, "データファイル (all_matchup_data.parquet) が見つかりません。"

    if '年度' not in df.columns:
        df['年度'] = pd.to_datetime(df['日付']).dt.year

    # 名寄せ処理
    aliases_file = "npb_official_aliases.json"
    if os.path.exists(aliases_file):
        with open(aliases_file, "r", encoding="utf-8") as f:
            player_aliases = json.load(f)
        df['投手'] = df['投手'].replace(player_aliases)
        df['打者'] = df['打者'].replace(player_aliases)
        
        manual_aliases = {
            "Ｔ－岡田": "T-岡田", "Ｔ-岡田": "T-岡田",
            "Ｃ．Ｃ．メルセデス": "C.C.メルセデス", "Ｒ．マルティネス": "R.マルティネス",
            "Ｌ．モイネロ": "L.モイネロ"
        }
        df['投手'] = df['投手'].replace(manual_aliases)
        df['打者'] = df['打者'].replace(manual_aliases)

    # 同姓同名・改名の例外処理
    df.loc[(df['投手'] == '拓也') & (df['年度'] <= 2016) & (df['投手チーム'] == 'ソフトバンク'), '投手'] = '甲斐拓也'
    df.loc[(df['打者'] == '拓也') & (df['年度'] <= 2016) & (df['打者チーム'] == 'ソフトバンク'), '打者'] = '甲斐拓也'
    df.loc[(df['投手'] == '拓也') & (df['年度'] >= 2026) & (df['投手チーム'] == 'ヤクルト'), '投手'] = '矢崎拓也'
    df.loc[(df['打者'] == '拓也') & (df['年度'] >= 2026) & (df['打者チーム'] == 'ヤクルト'), '打者'] = '矢崎拓也'

    # 方向を排除した詳細結果の列を作成
    df['純粋な結果'] = df['結果'].apply(clean_detail_result)

    return df, None

@st.cache_data
def load_directories():
    """Gitにアップされた全年度の選手名鑑を結合してマスター化する"""
    files = glob.glob("player_directory_*.csv")
    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except:
            pass
    if df_list:
        df_dir = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['チーム', '選手名'])
        return df_dir
    return pd.DataFrame()

# =====================================================================
# UI 構築とサイドバー
# =====================================================================
st.title("⚾ NPB 投手 vs 打者 分析ダッシュボード")

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

# 球団が選ばれている場合、名鑑からその球団の選手一覧をヒントとして表示
if selected_team != "全球団" and not df_dir.empty:
    team_players = df_dir[df_dir['チーム'] == selected_team]['選手名'].unique()
    with st.sidebar.expander(f"💡 {selected_team} の所属選手一覧"):
        st.write("、".join(sorted(team_players)))

# データ絞り込み（球団）
filtered_df = df.copy()
if selected_team != "全球団":
    # 投手か打者どちらかがその球団である試合に絞る
    filtered_df = filtered_df[(filtered_df['投手チーム'] == selected_team) | (filtered_df['打者チーム'] == selected_team)]

# --- 2. モード選択と記入式入力 ---
analysis_mode = st.sidebar.radio(
    "分析モード",
    ["投手視点で分析", "打者視点で分析", "特定の対戦 (投手 vs 打者)"]
)

if analysis_mode == "投手視点で分析":
    selected_pitcher = st.sidebar.text_input("投手名を入力 (フルネーム)", value=st.session_state.input_pitcher)
    st.session_state.input_pitcher = selected_pitcher
elif analysis_mode == "打者視点で分析":
    selected_batter = st.sidebar.text_input("打者名を入力 (フルネーム)", value=st.session_state.input_batter)
    st.session_state.input_batter = selected_batter
else:
    col_p, col_b = st.sidebar.columns(2)
    selected_pitcher = col_p.text_input("投手名", value=st.session_state.input_pitcher)
    selected_batter = col_b.text_input("打者名", value=st.session_state.input_batter)
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
    st.sidebar.caption("↓ クリックで入力欄にセット")
    for fav in st.session_state.favorites:
        cols = st.sidebar.columns([3, 1])
        if cols[0].button(f"👤 {fav}", key=f"set_{fav}"):
            # クリックされたらモードに応じて入力状態を上書きして再描画
            if analysis_mode == "投手視点で分析":
                st.session_state.input_pitcher = fav
            elif analysis_mode == "打者視点で分析":
                st.session_state.input_batter = fav
            else:
                # 特定対戦モードの場合は、空いている方に入れる（簡易的）
                if not st.session_state.input_pitcher:
                    st.session_state.input_pitcher = fav
                else:
                    st.session_state.input_batter = fav
            st.rerun()
            
        if cols[1].button("✖", key=f"del_{fav}"):
            st.session_state.favorites.remove(fav)
            st.rerun()

st.sidebar.markdown("---")
years = sorted(filtered_df['年度'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("表示する年度", ["全年度"] + list(years))

# =====================================================================
# データ抽出と表示
# =====================================================================
if selected_year != "全年度":
    filtered_df = filtered_df[filtered_df['年度'] == selected_year]

# モードに応じたデータの決定
if analysis_mode == "投手視点で分析":
    if not selected_pitcher:
        st.info("👈 サイドバーから投手名を入力してください。")
        st.stop()
    target_df = filtered_df[filtered_df['投手'] == selected_pitcher]
    title_text = f"📊 {selected_pitcher} 投手のデータ"

elif analysis_mode == "打者視点で分析":
    if not selected_batter:
        st.info("👈 サイドバーから打者名を入力してください。")
        st.stop()
    target_df = filtered_df[filtered_df['打者'] == selected_batter]
    title_text = f"📊 {selected_batter} 打者のデータ"

else:
    if not selected_pitcher or not selected_batter:
        st.info("👈 サイドバーから投手名と打者名の両方を入力してください。")
        st.stop()
    target_df = filtered_df[(filtered_df['投手'] == selected_pitcher) & (filtered_df['打者'] == selected_batter)]
    title_text = f"⚔️ {selected_pitcher} vs {selected_batter}"

# 年度の表示追加
st.subheader(f"{title_text} ({selected_year})")

if target_df.empty:
    st.warning("該当する対戦データがありません。名前が正しいか、球団や年度の絞り込みを解除してお試しください。")
else:
    # --- 指標計算 ---
    total_pa = len(target_df)
    hits = len(target_df[target_df['打席結果'] == '安打'])
    at_bats = len(target_df[target_df['打席結果'].isin(['安打', '凡打', '三振'])])
    avg = hits / at_bats if at_bats > 0 else 0.000
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総対戦打席数", f"{total_pa} 打席")
    col2.metric("打数", f"{at_bats} 打数")
    col3.metric("安打数", f"{hits} 本")
    col4.metric("打率", f".{str(int(avg * 1000)).zfill(3)}")

    st.markdown("---")

    # --- グラフ描画（方向抜き詳細 ＆ 割合） ---
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### 🎯 打席結果の内訳")
        result_counts = target_df['打席結果'].value_counts().reset_index()
        result_counts.columns = ['打席結果', '回数']
        fig1 = px.pie(result_counts, values='回数', names='打席結果', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        st.markdown("#### ⚾ 詳細な結果 (方向なし)")
        # 🌟 ここで「左飛」などが「フライ」に変換された「純粋な結果」を使用します
        detail_counts = target_df['純粋な結果'].value_counts().head(10).reset_index()
        detail_counts.columns = ['結果', '回数']
        fig2 = px.bar(detail_counts, x='回数', y='結果', orientation='h', color='回数', color_continuous_scale='Blues')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # --- 🌟 打球方向の図解可視化 ---
    st.markdown("#### 🏟️ 飛びやすい打球方向")
    
    dir_counts = target_df['打球方向'].value_counts()
    left = dir_counts.get('左', 0)
    center = dir_counts.get('中', 0)
    right = dir_counts.get('右', 0)
    total_dir = left + center + right
    
    if total_dir > 0:
        left_pct = (left / total_dir) * 100
        center_pct = (center / total_dir) * 100
        right_pct = (right / total_dir) * 100
        
        fig_dir = go.Figure()

        # グラウンドの外野芝生（扇形）
        fig_dir.add_trace(go.Scatter(
            x=[0, -1.2, 0, 1.2, 0], 
            y=[0, 1.2, 1.7, 1.2, 0],
            mode='lines', fill='toself', fillcolor='#e0f2e0',
            line=dict(color='white', width=3), hoverinfo='skip'
        ))
        
        # グラウンドの内野土
        fig_dir.add_trace(go.Scatter(
            x=[0, -0.6, 0, 0.6, 0], 
            y=[0, 0.6, 0.85, 0.6, 0],
            mode='lines', fill='toself', fillcolor='#f2e6d9',
            line=dict(color='white', width=2), hoverinfo='skip'
        ))

        # バブルの大きさを件数に合わせて動的に変える（最小30〜最大80）
        sizes = [left, center, right]
        max_size = max(sizes) if max(sizes) > 0 else 1
        marker_sizes = [(s/max_size)*50 + 30 for s in sizes]

        # 左・中・右にバブルとテキストを配置
        fig_dir.add_trace(go.Scatter(
            x=[-0.7, 0, 0.7],
            y=[1.1, 1.4, 1.1],
            mode='markers+text',
            marker=dict(
                size=marker_sizes,
                color=['#FF9999', '#99CCFF', '#FFCC99'],
                line=dict(color='white', width=2)
            ),
            text=[f"左方向<br>{left}本<br>({left_pct:.1f}%)", 
                  f"センター<br>{center}本<br>({center_pct:.1f}%)", 
                  f"右方向<br>{right}本<br>({right_pct:.1f}%)"],
            textposition='middle center',
            textfont=dict(color='black', size=14, weight='bold'),
            hoverinfo='none'
        ))

        fig_dir.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.9]),
            plot_bgcolor='white',
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            height=350
        )
        
        st.plotly_chart(fig_dir, use_container_width=True)
    else:
        st.info("この条件では打球方向が記録されたデータ（安打や凡打など）がありません。（三振や四死球のみなど）")
