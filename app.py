import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ページのタイトル
st.title("本格版 MMMダッシュボード")
st.write("過去のデータからベース売上と広告効果を分解し、投資対効果を分析します。")

# CSVのアップロード機能
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    # データの読み込み
    df = pd.read_csv(uploaded_file)
    df_clean = df.fillna(0) # 空のデータを0で埋める
    
    st.success("データの読み込みに成功しました！")
    
    # ----------------------------------------
    # 1. 変数の動的選択（マッピング）
    # ----------------------------------------
    st.write("---")
    st.subheader("⚙️ 分析する変数の設定")
    st.write("アップロードしたデータのどの列を分析に使うか指定してください。")
    
    # 目的変数の選択（単一選択）
    target_col = st.selectbox("🎯 目的変数（売上やCVなど最大化したい指標）:", df_clean.columns)
    
    # 選ばれた目的変数以外のカラムを候補にする
    remaining_cols = [col for col in df_clean.columns if col != target_col]
    
    # メディア変数（広告費など）の選択（複数選択）
    media_cols = st.multiselect("📺 メディア変数（TVCM、WEB広告などの投下量・費用）:", remaining_cols)
    
    # さらに残ったカラムを候補にする
    remaining_cols_for_control = [col for col in remaining_cols if col not in media_cols]
    
    # コントロール変数（外部要因）の選択（複数選択）
    control_cols = st.multiselect("🌤️ コントロール変数（季節性、天候、値引きなど外部要因）※任意:", remaining_cols_for_control)
    
    # ----------------------------------------
    # 分析実行ボタン
    # ----------------------------------------
    if st.button("AIでMMM分析を実行する"):
        if not media_cols:
            st.error("メディア変数を1つ以上選択してください。")
        else:
            with st.spinner('AIが計算中です...'):
                # ----------------------------------------
                # 2. AIによるMMM分析（重回帰）
                # ----------------------------------------
                # 説明変数（メディア + コントロール）
                X_cols = media_cols + control_cols
                X = df_clean[X_cols]
                y = df_clean[target_col]
                
                # AIの学習
                model = LinearRegression()
                model.fit(X, y)
                
                st.write("---")
                st.subheader("📊 分析結果")
                
                # モデル精度の算出と表示 (R2スコア)
                r2_score = model.score(X, y)
                st.metric(label="AIの予測精度（R²スコア）", value=f"{r2_score:.2f} / 1.00", delta="1.0に近いほど高精度")
                
                # ----------------------------------------
                # 3. ベース売上と各要素の貢献度の算出・可視化
                # ----------------------------------------
                # 各変数の係数と切片（ベース）を取得
                coefficients = dict(zip(X_cols, model.coef_))
                intercept = model.intercept_
                
                # 時系列ごとの貢献売上を計算する用のデータフレーム
                contribution_df = pd.DataFrame(index=df_clean.index)
                
                # もしデータに 'date' や 'week' があれば横軸にする
                if 'date' in df_clean.columns:
                    contribution_df.index = df_clean['date']
                
                # ① ベース売上（切片）
                contribution_df['Base_Sales (ベース売上)'] = intercept
                
                # ② コントロール変数の貢献分
                for col in control_cols:
                    contribution_df[f'{col} (外部要因)'] = df_clean[col].values * coefficients[col]
                    
                # ③ メディア変数の貢献分
                for col in media_cols:
                    contribution_df[f'{col} (広告効果)'] = df_clean[col].values * coefficients[col]
                
                # マイナスの貢献度が出た場合の簡易補正（グラフ描画を綺麗にするため0を下限にする）
                contribution_df = contribution_df.clip(lower=0)
                
                st.write("#### 📈 売上要因の分解（Decomposition）")
                st.write("売上が「何によって作られたか」を時系列で積み上げグラフにして表示します。")
                
                # 積み上げ面グラフで表示
                st.area_chart(contribution_df)
                
                # ----------------------------------------
                # 4. メディア別の投資対効果（係数）
                # ----------------------------------------
                st.write("#### 💰 メディア別の投資対効果（1単位あたりの貢献度）")
                coef_df = pd.DataFrame