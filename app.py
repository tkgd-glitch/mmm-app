import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ページのタイトル
st.title("簡易版 MMMダッシュボード")
st.write("過去のデータから広告の貢献度をAIが分析し、予算のシミュレーションを行います。")

# CSVのアップロード機能
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    # データの読み込み
    df = pd.read_csv(uploaded_file)
    df_clean = df.fillna(0) # 空のデータを0で埋める
    
    st.success("データの読み込みに成功しました！")
    
    # ----------------------------------------
    # 1. データの可視化
    # ----------------------------------------
    st.write("---")
    st.subheader("📈 データの可視化（トレンド確認）")
    selected_column = st.selectbox("グラフに表示する指標を選んでください:", df_clean.columns)
    st.line_chart(df_clean[selected_column])

    # ----------------------------------------
    # 2. AIによるMMM分析（精度と貢献度）
    # ----------------------------------------
    st.write("---")
    st.subheader("🤖 AIによる広告の貢献度分析")
    
    # 分析する変数の設定
    target_col = 'sales'
    media_cols = ['tv_spend', 'search_spend', 'display_spend', 'social_spend', 'video_spend']
    
    # AIの学習（モデル構築）
    X = df_clean[media_cols]
    y = df_clean[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # ① モデル精度の算出と表示 (R2スコア)
    r2_score = model.score(X, y)
    st.metric(label="AIの予測精度（R²スコア）", value=f"{r2_score:.2f} / 1.00", delta="1.0に近いほど高精度")
    
    # ② 貢献度（係数）のグラフ化
    coef_df = pd.DataFrame({'メディア': media_cols, '貢献度（係数）': model.coef_})
    coef_df['貢献度（係数）'] = coef_df['貢献度（係数）'].apply(lambda x: max(0, x)) # マイナスは0にする
    coef_df = coef_df.sort_values(by='貢献度（係数）', ascending=False).set_index('メディア')
    
    st.bar_chart(coef_df)
    st.write("※ グラフの棒が長いほど、1円投資した時の売上リターン（投資対効果）が高いことを示します。")

    # ----------------------------------------
    # 3. 予算配分シミュレーター
    # ----------------------------------------
    st.write("---")
    st.subheader("💰 予算配分シミュレーター")
    st.write("各メディアの予算を動かして、「この配分なら売上がどうなるか」を予測してみましょう！")
    
    # スライダーで予算を入力させるUIを作成
    col1, col2 = st.columns(2)
    with col1:
        sim_tv = st.slider("TVCM予算", 0, 5000000, 2000000, step=100000)
        sim_search = st.slider("検索広告予算", 0, 5000000, 1000000, step=100000)
        sim_display = st.slider("ディスプレイ予算", 0, 5000000, 1000000, step=100000)
    with col2:
        sim_social = st.slider("SNS広告予算", 0, 5000000, 1000000, step=100000)
        sim_video = st.slider("動画広告予算", 0, 5000000, 500000, step=100000)
        
    # AIでシミュレーション予測
    sim_data = pd.DataFrame({
        'tv_spend': [sim_tv], 'search_spend': [sim_search], 
        'display_spend': [sim_display], 'social_spend': [sim_social], 'video_spend': [sim_video]
    })
    
    predicted_sales = model.predict(sim_data)[0]
    
    # 結果を大きく表示
    st.success("この予算配分での予測結果")
    st.metric(label="予測される売上（Sales）", value=f"{int(predicted_sales):,} 円")