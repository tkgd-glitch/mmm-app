import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from scipy.optimize import minimize
import altair as alt

# 1. ページ設定
st.set_page_config(page_title="OptiMix MMM | Dashboard", layout="wide", initial_sidebar_state="expanded")

# 追加: 画面遷移時にスクロールを一番上にリセットするJavaScriptコンポーネント (強化版)
if st.session_state.get('scroll_to_top', False):
    st.components.v1.html(
        """
        <script>
            function forceScrollTop() {
                const doc = window.parent.document;
                // Streamlitの各バージョンでスクロールバーを持つ可能性のあるコンテナを全て指定
                const scrollableElements = doc.querySelectorAll('.main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], [data-testid="stMain"]');
                scrollableElements.forEach(el => {
                    el.scrollTop = 0;
                    if (typeof el.scrollTo === 'function') {
                        el.scrollTo(0, 0);
                    }
                });
                window.parent.scrollTo(0, 0);
            }
            
            // 画面の再描画タイミングのズレをカバーするため、時間差で複数回強制スクロールを実行
            forceScrollTop();
            setTimeout(forceScrollTop, 50);
            setTimeout(forceScrollTop, 200);
            setTimeout(forceScrollTop, 500);
        </script>
        """,
        height=0,
        width=0
    )
    st.session_state.scroll_to_top = False

# --- UI/UX: モダン・グラスモーフィズム パレット ---
COLORS = [
    "#3B82F6", # Blue 500
    "#8B5CF6", # Violet 500
    "#0EA5E9", # Sky 500
    "#10B981", # Emerald 500
    "#F43F5E", # Rose 500
    "#F59E0B", # Amber 500
    "#64748B", # Slate 500
]
COLOR_BASELINE = "#CBD5E1" 
COLOR_TREND = "#94A3B8"
COLOR_SEASON = "#FDBA74"
COLOR_OUTLIER = "#F87171"
COLOR_INTER = "#818CF8"
COLOR_SYNERGY = "#D946EF"  

COLOR_PRIMARY = "#2563EB"
COLOR_SECONDARY = "#059669"
COLOR_ROSE = "#E11D48"  
COLOR_GRAY = "#64748B"
COLOR_TEXT_MAIN = "#0F172A"
COLOR_LIGHT_BG = "#F8FAFC"  

st.markdown(f"""
    <style>
    .stApp {{
        background-color: #f1f5f9;
        background-image: 
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(236, 72, 153, 0.08) 0px, transparent 50%);
        background-attachment: fixed;
    }}
    .main {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    h1, h2, h3, h4, h5, h6 {{ color: {COLOR_TEXT_MAIN}; font-weight: 700; letter-spacing: -0.02em; }}
    p {{ line-height: 1.6; color: #475569; }}
    
    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(32px) saturate(120%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(120%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.7);
        box-shadow: 1px 0 20px rgba(0, 0, 0, 0.02);
    }}
    [data-testid="stSidebarHeader"] {{ background: transparent !important; }}

    div[data-testid="stRadio"] > div {{
        flex-direction: row; 
        background: rgba(255, 255, 255, 0.5); 
        backdrop-filter: blur(24px) saturate(120%); 
        -webkit-backdrop-filter: blur(24px) saturate(120%); 
        padding: 0.375rem; 
        border-radius: 1.25rem; 
        gap: 0.25rem; 
        display: inline-flex; 
        border: 1px solid rgba(255, 255, 255, 0.8); 
        box-shadow: inset 0 1px 2px rgba(255,255,255,0.8), 0 2px 8px rgba(0, 0, 0, 0.03);
    }}
    div[data-testid="stRadio"] > div > label {{
        background-color: transparent; padding: 0.6rem 1.5rem; border-radius: 1rem; cursor: pointer; color: {COLOR_GRAY}; transition: all 0.3s ease; font-weight: 600; font-size: 0.9rem; border: 1px solid transparent;
    }}
    div[data-testid="stRadio"] > div > label:hover {{ color: {COLOR_TEXT_MAIN}; background-color: rgba(255, 255, 255, 0.6); }}
    div[data-testid="stRadio"] > div > label[data-checked="true"] {{
        background: rgba(255, 255, 255, 0.95); color: {COLOR_PRIMARY}; box-shadow: 0 4px 10px rgba(0,0,0,0.06); border: 1px solid rgba(255, 255, 255, 1);
    }}

    .insight-box {{
        background: rgba(255, 255, 255, 0.65); 
        backdrop-filter: blur(32px) saturate(120%); 
        -webkit-backdrop-filter: blur(32px) saturate(120%); 
        border-left: 4px solid {COLOR_PRIMARY}; 
        border-top: 1px solid rgba(255,255,255,1); 
        border-right: 1px solid rgba(255,255,255,0.6); 
        border-bottom: 1px solid rgba(255,255,255,0.6); 
        padding: 1.5rem; 
        border-radius: 1.25rem; 
        margin-bottom: 2rem; 
        display: flex; 
        align-items: flex-start; 
        gap: 1.25rem; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.04); 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .insight-box:hover {{ transform: translateY(-2px); box-shadow: 0 12px 24px -3px rgba(0, 0, 0, 0.06); }}
    .insight-icon {{ background: rgba(255,255,255,0.9); padding: 0.75rem; border-radius: 1rem; box-shadow: 0 4px 10px rgba(0,0,0,0.05); display: flex; align-items: center; justify-content: center; color: {COLOR_PRIMARY}; }}
    .insight-content h4 {{ margin: 0 0 0.5rem 0; color: {COLOR_TEXT_MAIN}; font-size: 1.1rem; }}
    .insight-content p {{ margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.6; }}
    
    .section-title {{ font-size: 1.7rem; color: {COLOR_TEXT_MAIN}; border-bottom: 1px solid rgba(0, 0, 0, 0.05); padding-bottom: 0.75rem; margin-bottom: 2rem; font-weight: 800; letter-spacing: -0.03em; }}

    button[kind="primary"], button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.95) 0%, rgba(29, 78, 216, 0.95) 100%) !important;
        backdrop-filter: blur(12px) saturate(120%) !important;
        -webkit-backdrop-filter: blur(12px) saturate(120%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    button[kind="primary"] *, button[data-testid="baseButton-primary"] * {{ color: #ffffff !important; }}
    button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg, rgba(30, 64, 175, 1) 0%, rgba(30, 58, 138, 1) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
    }}
    
    div[data-testid="stExpander"] {{ 
        background: rgba(255, 255, 255, 0.5); 
        backdrop-filter: blur(24px) saturate(120%); 
        border-radius: 1rem; 
        border: 1px solid rgba(255, 255, 255, 0.8); 
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255,255,255,1); 
        transition: all 0.3s ease; 
    }}
    
    .hover-lift {{ transition: transform 0.3s ease, box-shadow 0.3s ease; }}
    .hover-lift:hover {{ transform: translateY(-3px); box-shadow: 0 16px 30px -8px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255,255,255,1) !important; }}
    </style>
    """, unsafe_allow_html=True)

# SVG Icons
ICON_DOLLAR = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
ICON_USERS = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
ICON_PIE = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>'
ICON_ZAP = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
ICON_TARGET = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
ICON_LIGHTBULB = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1.3.5 2.6 1.5 3.5.8.8 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>'
ICON_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'

def render_kpi_card(title, value, change="", trend="neutral", svg_icon="", reverse_trend=False):
    if reverse_trend:
        trend_color = COLOR_ROSE if trend == "up" else COLOR_SECONDARY if trend == "down" else COLOR_GRAY
    else:
        trend_color = COLOR_SECONDARY if trend == "up" else COLOR_ROSE if trend == "down" else COLOR_GRAY
    trend_icon = "↑" if trend == "up" else "↓" if trend == "down" else "→"
    change_html = f'<div style="color: {trend_color}; font-size: 0.8rem; font-weight: 700; display: flex; align-items: center; gap: 4px; background: {trend_color}15; padding: 4px 8px; border-radius: 20px; border: 1px solid {trend_color}30;"><span>{trend_icon}</span> {change}</div>' if change else ""
    return f'<div class="hover-lift" style="background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(24px) saturate(120%); padding: 1.5rem; border-radius: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.9); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255, 255, 255, 1); height: 100%; display: flex; flex-direction: column; justify-content: space-between;"><div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem;"><div style="background: rgba(255,255,255,0.9); color: {COLOR_PRIMARY}; border-radius: 0.8rem; padding: 0.6rem; width: 3rem; height: 3rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,1);">{svg_icon}</div>{change_html}</div><div><h3 style="color: {COLOR_GRAY}; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.25rem 0;">{title}</h3><div style="color: {COLOR_TEXT_MAIN}; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;">{value}</div></div></div>'

def render_sim_summary(opt_sales, current_sales, is_cv, total_budget):
    uplift = opt_sales - current_sales
    if is_cv:
        title_opt = "最適化後の予測CV数"
        val_opt = f"{int(opt_sales):,} 件"
        val_up = f"増加幅: + {int(uplift):,} 件"
        title_curr = "現状 (過去平均) のCV予測"
        val_curr = f"{int(current_sales):,} 件"
        cpa_opt = total_budget / opt_sales if opt_sales > 0 else 0
        val_up += f" <span style='opacity:0.8; font-weight: normal; margin-left: 6px;'>(見込CPA: ¥{int(cpa_opt):,})</span>"
    else:
        title_opt = "最適化後の予測売上"
        val_opt = f"¥{int(opt_sales):,}"
        val_up = f"改善幅: + ¥{int(uplift):,}"
        title_curr = "現状 (過去平均ペース) での予測"
        val_curr = f"¥{int(current_sales):,}"

    return f'<div style="display: flex; gap: 1.5rem; margin-bottom: 2rem;"><div class="hover-lift" style="flex: 1; background: rgba(37, 99, 235, 0.9); backdrop-filter: blur(30px); padding: 1.75rem; border-radius: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.3); color: white; box-shadow: 0 12px 30px -5px rgba(37, 99, 235, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.4); position: relative; overflow: hidden;"><div style="font-size: 0.9rem; font-weight: 700; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.05em;">{title_opt}</div><div style="font-size: 2.75rem; font-weight: 800; margin: 0.75rem 0; letter-spacing: -0.03em; text-shadow: 0 2px 4px rgba(0,0,0,0.15);">{val_opt}</div><div style="background: rgba(255,255,255,0.2); backdrop-filter: blur(12px); display: inline-flex; align-items: center; gap: 8px; padding: 0.4rem 1rem; border-radius: 2rem; font-size: 0.9rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.4);">{val_up}</div><div style="position: absolute; right: -10%; top: -20%; width: 250px; height: 250px; background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%); border-radius: 50%;"></div></div><div class="hover-lift" style="flex: 1; background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(24px) saturate(120%); padding: 1.75rem; border-radius: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.9); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255, 255, 255, 1); display: flex; flex-direction: column; justify-content: center;"><div style="font-size: 0.9rem; font-weight: 700; color: {COLOR_GRAY}; text-transform: uppercase; letter-spacing: 0.05em;">{title_curr}</div><div style="font-size: 2.25rem; font-weight: 800; color: {COLOR_TEXT_MAIN}; margin-top: 0.75rem; letter-spacing: -0.02em;">{val_curr}</div></div></div>'

def render_path_diagram(direct_sum, indirect_sum, inter_name, target_name, media_details):
    total = direct_sum + indirect_sum
    direct_pct = (direct_sum / total * 100) if total > 0 else 0
    indirect_pct = (indirect_sum / total * 100) if total > 0 else 0
    
    breakdown_html = f"<div style='width: 100%; max-width: 850px; margin-top: 2rem; border-top: 1px dashed rgba(0,0,0,0.15); padding-top: 1.5rem;'><h6 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1rem; font-size: 0.95rem;'>各メディアの貢献内訳 (直接 vs 間接)</h6><div style='display: flex; flex-direction: column; gap: 8px;'>"
    for m in media_details:
        if m['total'] <= 0: continue
        d_pct = m['direct'] / m['total'] * 100
        i_pct = m['indirect'] / m['total'] * 100
        i_text = f"{i_pct:.0f}%" if i_pct > 8 else ""
        d_text = f"{d_pct:.0f}%" if d_pct > 8 else ""
        breakdown_html += f"<div style='display: flex; align-items: center;'><div style='width: 140px; font-size: 0.85rem; color: {COLOR_GRAY}; font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;'>{m['name']}</div><div style='flex: 1; height: 18px; display: flex; border-radius: 9px; overflow: hidden; background: rgba(0,0,0,0.05); margin: 0 15px;'><div style='width: {i_pct}%; background: {COLOR_SECONDARY}; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; color: white; font-weight: bold;' title='間接効果'>{i_text}</div><div style='width: {d_pct}%; background: {COLOR_PRIMARY}; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; color: white; font-weight: bold;' title='直接効果'>{d_text}</div></div><div style='width: 50px; text-align: right; font-size: 0.8rem; color: {COLOR_TEXT_MAIN}; font-weight: 700;'>100%</div></div>"
    breakdown_html += "</div></div>"
    
    html = f'<div style="background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(24px) saturate(120%); border-radius: 1.25rem; padding: 2.5rem 2rem 2rem 2rem; border: 1px solid rgba(255, 255, 255, 0.9); box-shadow: 0 8px 24px rgba(0,0,0,0.03); margin-bottom: 3rem;"><h5 style="color: {COLOR_TEXT_MAIN}; margin-top: 0; margin-bottom: 2.5rem; font-weight: 800; text-align: left;">階層型モデルの因果パス (直接効果 vs 間接効果)</h5><div style="position: relative; width: 100%; max-width: 850px; height: 260px; margin: 0 auto;"><div style="position: absolute; top: 200px; left: 0; transform: translateY(-50%); z-index: 3; background: linear-gradient(135deg, {COLOR_PRIMARY}, #1E40AF); color: white; padding: 1.2rem; border-radius: 1rem; width: 25%; text-align: center; font-weight: 700; box-shadow: 0 8px 16px rgba(37, 99, 235, 0.3), inset 0 1px 1px rgba(255,255,255,0.4);">マーケティング投資<br><span style="font-size: 0.75rem; font-weight: 400; opacity: 0.8; margin-top: 4px; display: block;">(各メディア)</span></div><div style="position: absolute; top: 200px; right: 0; transform: translateY(-50%); z-index: 3; background: linear-gradient(135deg, {COLOR_TEXT_MAIN}, #334155); color: white; padding: 1.2rem; border-radius: 1rem; width: 25%; text-align: center; font-weight: 700; box-shadow: 0 8px 16px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.2);">事業成果<br><span style="font-size: 0.75rem; font-weight: 400; opacity: 0.8; margin-top: 4px; display: block;">({target_name})</span></div><div style="position: absolute; top: 40px; left: 12.5%; width: 75%; height: 160px; border-top: 3px dashed {COLOR_SECONDARY}; border-left: 3px dashed {COLOR_SECONDARY}; border-right: 3px dashed {COLOR_SECONDARY}; border-top-left-radius: 20px; border-top-right-radius: 20px; z-index: 1;"><div style="position: absolute; bottom: -6px; right: -6px; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 10px solid {COLOR_SECONDARY};"></div></div><div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); z-index: 4; background: white; color: {COLOR_SECONDARY}; border: 2px solid {COLOR_SECONDARY}; padding: 1rem; border-radius: 1rem; width: 30%; min-width: 180px; text-align: center; font-weight: 700; box-shadow: 0 6px 12px rgba(0,0,0,0.06);"><div style="font-size: 0.9rem; color: {COLOR_SECONDARY}; margin-bottom: 6px; border-bottom: 1px dashed {COLOR_SECONDARY}; padding-bottom: 6px;">間接効果: {indirect_pct:.1f}%</div>中間変数の押し上げ<br><span style="font-size: 0.75rem; font-weight: 400; opacity: 0.8; margin-top: 4px; display: block;">({inter_name})</span></div><div style="position: absolute; top: 200px; left: 25%; width: 50%; border-bottom: 4px solid {COLOR_PRIMARY}; z-index: 1;"><div style="position: absolute; top: -14px; left: 50%; transform: translateX(-50%); background: white; padding: 4px 14px; color: {COLOR_PRIMARY}; font-weight: 800; border-radius: 20px; font-size: 0.85rem; border: 1px solid {COLOR_PRIMARY}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">直接効果: {direct_pct:.1f}%</div><div style="position: absolute; top: -4px; right: -2px; width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 12px solid {COLOR_PRIMARY};"></div></div></div><p style="color: {COLOR_GRAY}; font-size: 0.85rem; margin-top: 1.5rem; text-align: center;">※ メディア投資による総貢献量のうち、「直接成果に繋がった割合」と「{inter_name} の増加を経由して成果に繋がった割合」</p>{breakdown_html}</div>'
    return html

# 2. 初期化と共通関数
if 'init_done' not in st.session_state:
    st.session_state.init_done = False
    st.session_state.analysis_results = None
    st.session_state.active_tab = "モデルチューニング"

if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []
if 'last_sim_result' not in st.session_state:
    st.session_state.last_sim_result = None

def save_current_scenario(total_bud, opt_sal, spends_dict, is_cv):
    name = st.session_state.get("input_scen_name", "").strip()
    if not name:
        name = f"シナリオ {len(st.session_state.get('saved_scenarios', [])) + 1}"
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []
    st.session_state.saved_scenarios.append({
        "id": str(time.time()),
        "name": name,
        "total_budget": total_bud,
        "predicted_result": opt_sal,
        "media_budgets": spends_dict,
        "is_cv": is_cv
    })

def sync_tab(): 
    st.session_state.active_tab = st.session_state._radio_tab
    st.session_state.scroll_to_top = True

def go_to_tab(tab_name): 
    st.session_state.active_tab = tab_name
    st.session_state.scroll_to_top = True

def apply_lag(series, lag):
    lag = int(round(lag))
    if lag <= 0:
        return series
    shifted = np.zeros(len(series))
    shifted[lag:] = series[:-lag]
    return shifted

def apply_adstock(series, decay):
    adstocked = np.zeros(len(series))
    for i in range(len(series)):
        adstocked[i] = series[i] if i == 0 else series[i] + decay * adstocked[i-1]
    return adstocked

def apply_hill_saturation(x, K, S): 
    x_safe = np.maximum(x, 1e-6)
    K_safe = np.maximum(K, 1e-6)
    return (np.power(x_safe, S) / (np.power(K_safe, S) + np.power(x_safe, S))) + (1e-5 * x_safe / K_safe)

def calculate_mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100

def calculate_durbin_watson(y_true, y_pred):
    residuals = y_true - y_pred
    diff_residuals = np.diff(residuals)
    return np.sum(diff_residuals**2) / np.sum(residuals**2)

# 追加: 予算シミュレーター用の係数取得関数をトップレベルに移動してレポート側でも使えるようにする
def get_sim_coef(result_dict, media_name):
    if result_dict.get('intermediate_results'): return max(0, result_dict['intermediate_results']['total_coefs'][media_name])
    return max(0, result_dict['coefficients'].get(media_name, 0))

# --- 修正: レポート出力（HTML）ジェネレーターの引数と内容を整理 ---
def generate_html_report(res, df_clean, is_cv, margin_r, last_sim_result=None):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    total_sales = max(1, res['y_true'].sum())
    marketing_sales = 0
    synergy_sales_total = 0
    if res.get('synergy_col_name') and res.get('synergy_col_name') in res['coefficients']:
        synergy_sales_total = np.sum(res['X_transformed'][res['synergy_col_name']].values * res['coefficients'][res['synergy_col_name']])
        marketing_sales += synergy_sales_total
        
    for m in res['media_cols']:
        if res.get('intermediate_results'):
            marketing_sales += max(0, np.sum(res['X_transformed'][m].values * res['intermediate_results']['total_coefs'][m]))
        else:
            marketing_sales += max(0, np.sum(res['X_transformed'][m].values * res['coefficients'].get(m, 0)))
            
    marketing_pct = (marketing_sales / total_sales) * 100 if total_sales > 0 else 0
    total_invest = sum([df_clean[m].sum() for m in res['media_cols']])
    
    if is_cv:
        kpi_1_label = "対象期間の総CV数"
        kpi_1_val = f"{int(total_sales):,} 件"
        kpi_2_label = "平均 獲得単価 (CPA)"
        avg_cpa = total_invest / max(1, marketing_sales)
        kpi_2_val = f"¥ {int(avg_cpa):,}"
    else:
        kpi_1_label = "対象期間の総売上"
        kpi_1_val = f"¥ {int(total_sales):,}"
        kpi_2_label = "平均 ROI (利益ベース)" if margin_r < 1.0 else "平均 投資対効果 (ROAS)"
        avg_eff = (marketing_sales * margin_r) / max(1, total_invest)
        kpi_2_val = f"{avg_eff * 100:.1f}%"
        
    r2_val = f"{res['r2']:.2f}"
    
    media_rows = ""
    for m in res['media_cols']:
        invest = df_clean[m].sum()
        if invest <= 0: continue
        
        if res.get('intermediate_results'):
            contrib = max(0, np.sum(res['X_transformed'][m].values * res['intermediate_results']['total_coefs'][m]))
        else:
            contrib = max(0, np.sum(res['X_transformed'][m].values * res['coefficients'].get(m, 0)))
            
        syn_contrib = 0
        if synergy_sales_total > 0 and m in [res.get('synergy_m1'), res.get('synergy_m2')]:
            inv_m1 = df_clean[res.get('synergy_m1')].sum() if res.get('synergy_m1') else 0
            inv_m2 = df_clean[res.get('synergy_m2')].sum() if res.get('synergy_m2') else 0
            sum_inv = inv_m1 + inv_m2
            if sum_inv > 0: syn_contrib = synergy_sales_total * (invest / sum_inv)
            
        total_contrib = contrib + syn_contrib
        
        if is_cv:
            eff = (total_contrib / invest) * 1000000 if invest > 0 else 0
            eff_str = f"{eff:,.1f} 件"
            cpa = invest / total_contrib if total_contrib > 0 else 0
            cpa_str = f"¥ {int(cpa):,}"
            media_rows += f"<tr><td>{m}</td><td>¥ {int(invest):,}</td><td>{int(total_contrib):,} 件</td><td>{eff_str}</td><td>{cpa_str}</td></tr>"
        else:
            eff = (total_contrib * margin_r) / invest if invest > 0 else 0
            eff_str = f"{eff * 100:.1f}%"
            media_rows += f"<tr><td>{m}</td><td>¥ {int(invest):,}</td><td>¥ {int(total_contrib):,}</td><td>{eff_str}</td><td>-</td></tr>"

    th_eff = "100万円あたりCV数" if is_cv else ("ROI" if margin_r < 1.0 else "ROAS")
    th_cpa = "CPA" if is_cv else "-"
    th_contrib = "CV貢献" if is_cv else "売上貢献"

    # 追加: 最新のシミュレーション結果をレポートに組み込む
    sim_html = ""
    if last_sim_result:
        sim_res = last_sim_result
        n_periods = sim_res['n_periods']
        unit_str = "件" if is_cv else "¥"
        val_fmt = lambda x: f"{int(x):,} {unit_str}" if is_cv else f"¥ {int(x):,}"
        
        sim_html += f"<h2>3. 最新の予算アロケーション・シミュレーション</h2>"
        sim_html += f"<p style='color: #475569;'>対象期間の総予算枠: <strong>¥ {int(sim_res['total_budget']):,}</strong></p>"
        
        # 修正: 売上かCVかでヘッダーの文言を分かりやすく分岐
        th_pred_curr = "現状予測 (CV数)" if is_cv else "現状予測 (売上)"
        th_pred_opt = "最適化後予測 (CV数)" if is_cv else "最適化後予測 (売上)"
        sim_html += f"<table><tr><th>メディア</th><th>現状予算</th><th>最適化予算</th><th>{th_pred_curr}</th><th>{th_pred_opt}</th></tr>"
        
        for idx, m in enumerate(res['media_cols']):
            b_curr = df_clean[m].mean() * n_periods
            b_opt = sim_res['opt_spends'][idx]
            
            c = get_sim_coef(res, m)
            steady_curr = df_clean[m].mean() / (1 - res['decay_rates'][m])
            s_curr = apply_hill_saturation(steady_curr, res['hill_K_values'][m], res['hill_S_values'][m]) * c * n_periods
            
            steady_opt = (b_opt/n_periods) / (1 - res['decay_rates'][m])
            s_opt = apply_hill_saturation(steady_opt, res['hill_K_values'][m], res['hill_S_values'][m]) * c * n_periods
            
            sim_html += f"<tr><td>{m}</td><td>¥ {int(b_curr):,}</td><td class='highlight'>¥ {int(b_opt):,}</td><td>{val_fmt(s_curr)}</td><td class='highlight'>{val_fmt(s_opt)}</td></tr>"
            
        if res.get('synergy_col_name') and res['synergy_col_name'] in res['coefficients']:
            m1, m2 = res['synergy_m1'], res['synergy_m2']
            idx1, idx2 = res['media_cols'].index(m1), res['media_cols'].index(m2)
            b_opt_1, b_opt_2 = sim_res['opt_spends'][idx1], sim_res['opt_spends'][idx2]
            
            steady_opt_1 = (b_opt_1/n_periods) / (1 - res['decay_rates'][m1])
            steady_opt_2 = (b_opt_2/n_periods) / (1 - res['decay_rates'][m2])
            
            syn_sat = apply_hill_saturation(steady_opt_1, res['hill_K_values'][m1], res['hill_S_values'][m1]) * apply_hill_saturation(steady_opt_2, res['hill_K_values'][m2], res['hill_S_values'][m2])
            syn_opt_sales = syn_sat * res['coefficients'][res['synergy_col_name']] * n_periods
            
            synergy_sales_total_historical = np.sum(res['X_transformed'][res['synergy_col_name']].values * res['coefficients'][res['synergy_col_name']])
            s_curr_syn = synergy_sales_total_historical * (n_periods / len(df_clean))
            
            sim_html += f"<tr><td>相乗効果 ({m1} × {m2})</td><td>-</td><td>-</td><td>{val_fmt(s_curr_syn)}</td><td class='highlight'>{val_fmt(syn_opt_sales)}</td></tr>"
            
        sim_html += "</table>"

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>OptiMix MMM エグゼクティブ・サマリー</title>
<style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #1e293b; line-height: 1.6; background-color: #f8fafc; padding: 40px 20px; }}
    .container {{ max-width: 1000px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
    .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end; }}
    h1 {{ color: #2563eb; margin: 0; font-size: 28px; letter-spacing: -0.5px; }}
    .date {{ color: #64748b; font-size: 14px; margin: 0; }}
    h2 {{ color: #0f172a; font-size: 20px; margin-top: 40px; border-left: 4px solid #2563eb; padding-left: 10px; }}
    .kpi-board {{ display: flex; gap: 20px; margin-bottom: 30px; }}
    .kpi-card {{ flex: 1; background: #f1f5f9; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0; }}
    .kpi-value {{ font-size: 24px; font-weight: 800; color: #0f172a; margin-top: 8px; }}
    .kpi-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
    th, td {{ padding: 14px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; }}
    th {{ background-color: #f8fafc; color: #475569; font-weight: 600; text-align: center; border-bottom: 2px solid #cbd5e1; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .highlight {{ color: #059669; font-weight: bold; }}
    .footer {{ margin-top: 50px; text-align: center; color: #94a3b8; font-size: 12px; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OptiMix MMM エグゼクティブ・サマリー</h1>
            <p class="date">出力日時: {now_str}</p>
        </div>
        
        <h2>1. モデル精度と全体KPI</h2>
        <div class="kpi-board">
            <div class="kpi-card">
                <div class="kpi-label">{kpi_1_label}</div>
                <div class="kpi-value">{kpi_1_val}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">マーケティング貢献比率</div>
                <div class="kpi-value">{marketing_pct:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">{kpi_2_label}</div>
                <div class="kpi-value" style="color: #2563eb;">{kpi_2_val}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">モデル予測精度 (R²)</div>
                <div class="kpi-value">{r2_val}</div>
            </div>
        </div>

        <h2>2. チャネル別 投資効率</h2>
        <table>
            <tr>
                <th>メディア</th>
                <th>投資額</th>
                <th>{th_contrib}</th>
                <th>{th_eff}</th>
                <th>{th_cpa}</th>
            </tr>
            {media_rows}
        </table>

        {sim_html}
        
        <div class="footer">
            Generated by OptiMix MMM Dashboard
        </div>
    </div>
</body>
</html>
"""
    return html.encode('utf-8-sig')

class NonNegativeMediaModel:
    def __init__(self, media_cols, scaling_method="平均値割 (Mean)", estimator_type="ベイズ推定 (Bayesian Ridge)"):
        self.media_cols = media_cols
        self.scaling_method = scaling_method
        self.estimator_type = estimator_type
        self.scalers = {}
        self.coef_ = None
        self.intercept_ = None
        self.features = None

    def _fit_scale(self, X):
        X_scaled = X.copy()
        for col in X.columns:
            if self.scaling_method == "平均値割 (Mean)":
                m = X[col].mean()
                m_val = m if abs(m) > 1e-5 else 1.0 
                self.scalers[col] = {'type': 'mean', 'val': m_val}
                X_scaled[col] = X[col] / m_val
            elif self.scaling_method == "Min-Max":
                mi, ma = X[col].min(), X[col].max()
                diff = ma - mi if abs(ma - mi) > 1e-5 else 1.0
                self.scalers[col] = {'type': 'minmax', 'min': mi, 'diff': diff}
                X_scaled[col] = (X[col] - mi) / diff
            elif self.scaling_method == "標準化 (Z-score)":
                m, s = X[col].mean(), X[col].std()
                s_val = s if abs(s) > 1e-5 else 1.0
                self.scalers[col] = {'type': 'zscore', 'mean': m, 'std': s_val}
                X_scaled[col] = (X[col] - m) / s_val
            else:
                self.scalers[col] = {'type': 'none'}
        return X_scaled

    def _get_estimator(self):
        if self.estimator_type == "ベイズ推定 (Bayesian Ridge)":
            return BayesianRidge()
        else:
            return Ridge(alpha=1.0)

    def fit(self, X, y):
        self.features = X.columns
        X_scaled = self._fit_scale(X)
        
        self.y_mean_ = np.mean(y) if abs(np.mean(y)) > 1e-5 else 1.0
        y_scaled = y / self.y_mean_
        
        model = self._get_estimator().fit(X_scaled, y_scaled)
        coefs = pd.Series(model.coef_, index=X_scaled.columns)
        
        negative_medias = [m for m in self.media_cols if m in X_scaled.columns and coefs[m] < 0]
        
        if negative_medias:
            X_sub = X_scaled.drop(columns=negative_medias)
            model_sub = self._get_estimator().fit(X_sub, y_scaled)
        else:
            model_sub = model
            X_sub = X_scaled

        self.coef_ = np.zeros(len(self.features))
        self.intercept_ = model_sub.intercept_ * self.y_mean_
        for i, col in enumerate(self.features):
            if col in X_sub.columns:
                c = model_sub.coef_[X_sub.columns.get_loc(col)] * self.y_mean_
                sc = self.scalers[col]
                if sc['type'] == 'mean':
                    self.coef_[i] = c / sc['val']
                elif sc['type'] == 'minmax':
                    self.coef_[i] = c / sc['diff']
                    self.intercept_ -= (sc['min'] * c) / sc['diff']
                elif sc['type'] == 'zscore':
                    self.coef_[i] = c / sc['std']
                    self.intercept_ -= (sc['mean'] * c) / sc['std']
                else:
                    self.coef_[i] = c
            else:
                self.coef_[i] = 0.0
        return self

    def predict(self, X):
        return np.dot(X[self.features].values, self.coef_) + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        if ss_tot == 0: return 0.0
        return 1 - (ss_res / ss_tot)

def calculate_vif(X_df, exclude_cols=None):
    if exclude_cols is None: exclude_cols = []
    vif_data = []
    
    X_vif = X_df.drop(columns=[c for c in exclude_cols if c in X_df.columns]).copy()
    X_vif['Intercept'] = 1.0 
    cols_to_check = [c for c in X_vif.columns if not c.startswith('outlier_event_') and c != 'Intercept']
    for col in cols_to_check:
        X_other = X_vif.drop(columns=[col])
        y_i = X_vif[col]
        model = LinearRegression(fit_intercept=False).fit(X_other, y_i)
        vif = 1 / (1 - model.score(X_other, y_i) + 1e-10)
        vif_data.append({'変数': col, 'VIFスコア': round(vif, 2)})
    return pd.DataFrame(vif_data)

def get_time_features(df_len, granularity, use_trend, use_seasonality):
    features = pd.DataFrame(index=range(df_len))
    t = np.arange(df_len)
    if use_trend: features['trend'] = t / (df_len - 1 if df_len > 1 else 1)
    if use_seasonality:
        p = 52.14 if "週" in granularity else 12 if "月" in granularity else 365.25
        features['seasonality_sin'] = np.sin(2 * np.pi * t / p)
        features['seasonality_cos'] = np.cos(2 * np.pi * t / p)
    return features

def get_outlier_features(y, threshold, window=5):
    if len(y) < window: return pd.DataFrame(index=range(len(y))), []
    y_series = pd.Series(y)
    rolling_median = y_series.rolling(window=window, center=True, min_periods=1).median()
    mad = np.median(np.abs(y - rolling_median))
    if mad == 0: mad = 1e-10
    mod_z_scores = np.abs(y - rolling_median) / (mad / 0.6745)
    outlier_indices = np.where(mod_z_scores > threshold)[0]
    outlier_df = pd.DataFrame(index=range(len(y)))
    outlier_cols = []
    for idx in outlier_indices:
        col_name = f"outlier_event_{idx}"
        outlier_df[col_name] = 0
        outlier_df.loc[idx, col_name] = 1 
        outlier_cols.append(col_name)
    return outlier_df, outlier_cols

@st.cache_data
def convert_df_to_csv(df): return df.to_csv(index=True).encode('utf-8-sig')
@st.cache_data
def convert_df_to_csv_no_index(df): return df.to_csv(index=False).encode('utf-8-sig')

# 3. サイドバー
with st.sidebar:
    st.markdown(f'<div style="font-size: 1.4rem; font-weight: 800; color: {COLOR_TEXT_MAIN}; margin-bottom: 2rem; display: flex; align-items: center; letter-spacing: -0.03em;"><div style="background: rgba(37, 99, 235, 0.15); color: {COLOR_PRIMARY}; width: 36px; height: 36px; border-radius: 10px; border: 1px solid rgba(37, 99, 235, 0.2); display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: inset 0 2px 4px rgba(255,255,255,0.6);"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg></div>OptiMix MMM</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("データをアップロード (CSV)", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        data = pd.read_csv(file)
        datetime_cols = []
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_datetime(data[col][:10])
                    datetime_cols.append(col)
                except:
                    pass
        num_cols = data.select_dtypes(include=['number']).columns.tolist()
        return data, num_cols, datetime_cols

    df, numeric_cols, datetime_cols = load_data(uploaded_file)
    df_clean = df.copy()
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    if not st.session_state.init_done:
        st.session_state.target_selector = 'sales' if 'sales' in numeric_cols else numeric_cols[0]
        st.session_state.media_selector = [c for c in numeric_cols if 'spend' in c.lower() and c != st.session_state.target_selector]
        st.session_state.init_done = True

    st.sidebar.markdown(f'<p style="font-size: 0.75rem; font-weight: 700; color: {COLOR_GRAY}; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 1.5rem; border-bottom: 1px solid rgba(0, 0, 0, 0.05); padding-bottom: 0.5rem;">1. Variables</p>', unsafe_allow_html=True)
    target_col = st.sidebar.selectbox("目的変数 (KGI)", numeric_cols, key='target_selector')
    
    target_type = st.sidebar.radio("目的変数のタイプ", ["売上金額 (Currency)", "コンバージョン件数 (Count)"])
    margin_rate = 1.0
    if target_type == "売上金額 (Currency)":
        margin_rate = st.sidebar.slider("平均粗利率 (%) - ROI計算用", min_value=1, max_value=100, value=100) / 100.0

    feature_options = [c for c in numeric_cols if c != target_col]
    
    media_cols = st.sidebar.multiselect("メディア変数 (投資)", feature_options, key='media_selector')
    control_cols_raw = st.sidebar.multiselect("外部要因 (コントロール)", feature_options)
    
    st.sidebar.markdown(f'<p style="font-size: 0.75rem; font-weight: 700; color: {COLOR_GRAY}; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 1.5rem; border-bottom: 1px solid rgba(0, 0, 0, 0.05); padding-bottom: 0.5rem;">2. Time Settings</p>', unsafe_allow_html=True)
    data_granularity = st.sidebar.selectbox("データの粒度", ["週次データ (Weekly)", "日次データ (Daily)", "月次データ (Monthly)"])
    date_col = st.sidebar.selectbox("日付カラム (ソート用)", ["指定なし"] + datetime_cols)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    with st.sidebar.expander("データクレンジング (イベント追加)", expanded=False):
        st.markdown("<p style='font-size: 0.8rem; color: #64748B; margin-bottom: 1rem;'>特需や障害などの異常値を手動でフラグ化（ダミー変数化）し、モデルに学習させます。</p>", unsafe_allow_html=True)
        
        if 'custom_events' not in st.session_state:
            st.session_state.custom_events = []
            
        with st.form("add_event_form"):
            ev_name = st.text_input("イベント名 (例: 年末セール)")
            if date_col != "指定なし":
                temp_date_series = pd.to_datetime(df_clean[date_col], errors='coerce').dropna()
                min_date = temp_date_series.min().date() if len(temp_date_series) > 0 else pd.Timestamp('2020-01-01').date()
                max_date = temp_date_series.max().date() if len(temp_date_series) > 0 else pd.Timestamp('2020-12-31').date()
                ev_dates = st.date_input("期間 (開始日〜終了日)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                ev_dates = st.slider("行番号範囲", 0, len(df_clean)-1, (0, 0))
            
            submitted = st.form_submit_button("＋ 追加")
            if submitted and ev_name:
                st.session_state.custom_events.append({
                    'name': ev_name,
                    'dates': ev_dates
                })
                st.rerun()
                
        if st.session_state.custom_events:
            st.markdown("<hr style='margin: 1rem 0; border-color: rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
            for i, ev in enumerate(st.session_state.custom_events):
                cols = st.columns([4, 1])
                cols[0].markdown(f"<div style='font-size: 0.85rem; color:{COLOR_TEXT_MAIN}; line-height: 1.3;'><b>{ev['name']}</b><br><span style='color:{COLOR_GRAY}; font-size: 0.75rem;'>{ev['dates']}</span></div>", unsafe_allow_html=True)
                if cols[1].button("削除", key=f"del_ev_{i}"):
                    st.session_state.custom_events.pop(i)
                    st.rerun()

    with st.sidebar.expander("高度なモデリング設定", expanded=False):
        estimator_type = st.selectbox("推定アルゴリズム", ["ベイズ推定 (Bayesian Ridge)", "Ridge回帰 (L2正則化)"])
        
        st.markdown("""
        <div style='background: rgba(241, 245, 249, 0.6); border-left: 3px solid #3B82F6; border-radius: 6px; padding: 12px; margin-top: 4px; margin-bottom: 12px; font-size: 0.8rem; color: #475569; line-height: 1.5;'>
            <b>アルゴリズムの使い分け</b><br>
            <span style='color: #2563EB; font-weight: 600;'>ベイズ推定:</span><br>
            データが少ない場合や相関が強い場合でも、過学習を防ぎ現実的で安定した結果を出します。<b>本番の分析（最終決定）に推奨</b>されます。<br>
            <span style='color: #059669; font-weight: 600; margin-top: 4px; display: inline-block;'>Ridge回帰:</span><br>
            計算が非常に高速です。変数を頻繁に入れ替えて、<b>素早く全体の傾向（アタリ）を掴みたい初期の試行錯誤</b>に向いています。
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        intermediate_col = st.selectbox("パス解析用 中間変数", ["使用しない"] + feature_options)
        exclude_from_inter = []
        if intermediate_col != "使用しない":
            st.markdown("<div style='background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; padding: 10px; margin-top: 10px; margin-bottom: 10px; border-radius: 4px;'><p style='font-size: 0.8rem; color: #475569; margin: 0;'><b>パス解析における留意点</b><br>検索広告のような「刈り取り系メディア」は、検索ボリュームを押し上げる原因ではないため、以下のリストで予測モデルから除外指定することを推奨します。</p></div>", unsafe_allow_html=True)
            exclude_from_inter = st.multiselect("中間変数の予測から除外するメディア", media_cols)
            
        st.divider()
        
        st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;'>相乗効果 (Synergy) の分析</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; color: #64748B; margin-bottom: 1rem;'>同時に出稿することで「1+1=3」のボーナス効果を生むメディアのペアを指定します。</p>", unsafe_allow_html=True)
        syn_col1, syn_col2 = st.columns(2)
        with syn_col1: synergy_m1 = st.selectbox("メディアA", ["なし"] + media_cols)
        with syn_col2: synergy_m2 = st.selectbox("メディアB", ["なし"] + media_cols)
        
        synergy_col_name = None
        if synergy_m1 != "なし" and synergy_m2 != "なし" and synergy_m1 != synergy_m2:
            synergy_col_name = f"Synergy ({synergy_m1} × {synergy_m2})"
        
        st.divider()
        
        scaling_method = st.selectbox("変数の正規化手法 (スケーリング)", ["平均値割 (Mean)", "Min-Max", "標準化 (Z-score)", "正規化しない"])
        
        st.markdown("""
        <div style='background: rgba(241, 245, 249, 0.6); border-left: 3px solid #F59E0B; border-radius: 6px; padding: 12px; margin-top: 4px; margin-bottom: 12px; font-size: 0.8rem; color: #475569; line-height: 1.5;'>
            <b>正規化（スケーリング）手法について</b><br>
            <span style='color: #D97706; font-weight: 600;'>平均値割 (Mean) 【MMM推奨】:</span><br>
            変数の平均を「1」に揃えます。値がマイナスにならないため、MMM特有のS字カーブや非負制約と最も相性が良く、実務の定石です。<br>
            <span style='color: #0F172A; font-weight: 600; margin-top: 4px; display: inline-block;'>Min-Max:</span><br>
            最小値を「0」、最大値を「1」にします。各変数のスケールを完全に同一の箱に収めたい場合に有効です。<br>
            <span style='color: #0F172A; font-weight: 600; margin-top: 4px; display: inline-block;'>標準化 (Z-score):</span><br>
            平均を「0」、標準偏差を「1」にします。一般的な統計の標準ですが、値がマイナス化するため、広告の残存効果などの計算でエラーや解釈の難しさを生むことがあります。
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        use_trend = st.checkbox("長期トレンドを含める", value=True)
        use_seasonality = st.checkbox("季節性を含める", value=True)
        st.divider()
        use_outlier_detection = st.checkbox("外れ値を自動隔離", value=False)
        if use_outlier_detection:
            outlier_threshold = st.slider("検知感度", 1.5, 5.0, 3.0, step=0.1)
        else:
            outlier_threshold = 3.0

    if date_col != "指定なし":
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.sort_values(date_col).reset_index(drop=True)

    model_media_cols = media_cols.copy()
    if synergy_col_name:
        model_media_cols.append(synergy_col_name)

    custom_event_cols = []
    if st.session_state.get('custom_events'):
        for i, ev in enumerate(st.session_state.custom_events):
            col_name = f"custom_event_{i}_{ev['name']}"
            df_clean[col_name] = 0
            if date_col != "指定なし" and isinstance(ev['dates'], tuple) and len(ev['dates']) == 2:
                s_date, e_date = ev['dates']
                mask = (df_clean[date_col].dt.date >= s_date) & (df_clean[date_col].dt.date <= e_date)
                df_clean.loc[mask, col_name] = 1
                custom_event_cols.append(col_name)
            elif date_col == "指定なし":
                s_idx, e_idx = ev['dates']
                df_clean.loc[s_idx:e_idx, col_name] = 1
                custom_event_cols.append(col_name)

    control_cols = [c for c in control_cols_raw if c not in media_cols and c != intermediate_col]
    control_cols.extend(custom_event_cols)

    media_color_map = {m: COLORS[i % len(COLORS)] for i, m in enumerate(media_cols)}

    tab_options = ["モデルチューニング", "分析オーバービュー", "予算シミュレーター"]
    
    if st.session_state.active_tab not in tab_options:
        st.session_state.active_tab = tab_options[0]

    st.markdown('<div style="padding-top: 0.5rem; padding-bottom: 2rem;">', unsafe_allow_html=True)
    st.radio("画面選択", tab_options, index=tab_options.index(st.session_state.active_tab),
             horizontal=True, label_visibility="collapsed", key="_radio_tab", on_change=sync_tab)
    st.markdown('</div>', unsafe_allow_html=True)

    current_tab = st.session_state.active_tab

    # ==========================================
    # TAB 1: パラメータチューニング画面
    # ==========================================
    if current_tab == "モデルチューニング":
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.markdown("<h2 class='section-title' style='margin-bottom: 0.5rem;'>モデル・チューニング</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 1rem;'>各メディアの遅延、残存効果、飽和度を最適化し、精度の高い統計モデルを構築します。</p>", unsafe_allow_html=True)
        with col_t2:
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                if st.button("分析を確定して次へ", type="primary", use_container_width=True, key="btn_run_top"):
                    st.session_state.run_analysis_flag = True
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.65); backdrop-filter: blur(32px) saturate(120%); -webkit-backdrop-filter: blur(32px) saturate(120%); border-left: 4px solid #10B981; border-top: 1px solid rgba(255,255,255,1); border-right: 1px solid rgba(255,255,255,0.6); border-bottom: 1px solid rgba(255,255,255,0.6); padding: 1.5rem; border-radius: 1.25rem; margin-bottom: 2rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.04);">
            <h4 style="margin: 0 0 0.5rem 0; color: #0F172A; font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l3-9 5 18 3-9h5"></path></svg>
                メディアごとのパラメータ設定の目安
            </h4>
            <p style="margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.7;">
                メディアの特性に応じて、ある程度のパラメータの「アタリ」をつけることで、より現実に即したモデルになります。<br><br>
                <strong style="color: #0F172A;">・遅延 (Lag):</strong> 雑誌広告や展示会など、出稿・実施から顧客のアクション（検索や購買）までに物理的なタイムラグがある場合に設定します。デジタル広告は通常 0 です。<br>
                <strong style="color: #0F172A;">・残存 (Decay):</strong> テレビや動画などの認知メディアは高く（0.3〜0.8）、検索やSNS静止画などの獲得メディアは低く（0.0〜0.3）なりやすいです。<br>
                <strong style="color: #0F172A;">・S字の強さ (Slope):</strong> 立ち上がりが遅い認知系メディア（テレビ等）はS字曲線になるため1.0以上（1.5〜3.0）が目安です。SNSなどで低予算時に効果が出にくい場合も大きめに設定します。逆に最初から効果が出る獲得系メディア（検索等）は凹曲線になるため1.0以下（0.5〜0.9）が目安です。<br>
                <strong style="color: #0F172A;">・半飽和点 (K ratio):</strong> 検索広告のように検索ボリュームに上限がありすぐ飽和するメディアは低め（0.1〜0.3）に、テレビのように予算をかければスケールするメディアは高め（0.5〜0.8）に設定します。
            </p>
            <hr style="margin: 1rem 0; border-color: rgba(0,0,0,0.05);">
            <h4 style="margin: 0 0 0.5rem 0; color: #0F172A; font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>
                階層型モデル（パス解析）を利用する際の設定
            </h4>
            <p style="margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.7;">
                「検索ボリューム」を中間変数として利用する場合、サイドバーの高度なモデリング設定にある<b>「中間変数の予測から除外するメディア」</b>に「検索広告費」を指定してください。
            </p>
        </div>
        """, unsafe_allow_html=True)

        if media_cols:
            with st.expander("アルゴリズム自動最適化 (Auto-Fit) の探索範囲設定", expanded=False):
                st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 0.9rem; margin-bottom: 1.5rem;'>最適化アルゴリズムが非現実的なパラメータを出力しないよう、探索範囲の制約を設定します。</p>", unsafe_allow_html=True)
                auto_fit_bounds = {}
                
                for idx, m in enumerate(media_cols):
                    if idx % 2 == 0: cols_b = st.columns(2, gap="large")
                    with cols_b[idx % 2]:
                        st.markdown(f"<div style='font-weight: 700; color: {COLOR_TEXT_MAIN}; padding-bottom: 0.5rem; border-bottom: 2px solid {media_color_map[m]}; margin-bottom: 1rem; display: inline-block;'>{m}</div>", unsafe_allow_html=True)
                        
                        is_tv = 'tv' in m.lower() or 'テレビ' in m.lower()
                        is_search = 'search' in m.lower() or '検索' in m.lower()
                        is_sns = 'social' in m.lower() or 'sns' in m.lower() or 'facebook' in m.lower()
                        
                        def_d_min, def_d_max = (0.3, 0.8) if is_tv else (0.0, 0.3) if is_search else (0.1, 0.6)
                        def_k_min, def_k_max = (0.1, 0.4) if is_search else (0.4, 0.9) if is_tv else (0.2, 0.7)
                        def_s_min, def_s_max = (0.5, 1.0) if is_search else (1.5, 3.0) if is_tv or is_sns else (0.8, 2.0)
                        
                        if f"bound_d_{m}" not in st.session_state: st.session_state[f"bound_d_{m}"] = (def_d_min, def_d_max)
                        if f"bound_k_{m}" not in st.session_state: st.session_state[f"bound_k_{m}"] = (def_k_min, def_k_max)
                        if f"bound_s_{m}" not in st.session_state: st.session_state[f"bound_s_{m}"] = (def_s_min, def_s_max)
                        
                        d_range = st.slider("減衰率 (Decay)", 0.0, 0.9, key=f"bound_d_{m}", step=0.05)
                        k_range = st.slider("半飽和点 (K ratio)", 0.1, 1.0, key=f"bound_k_{m}", step=0.05)
                        s_range = st.slider("S字の強さ (Slope)", 0.5, 3.0, key=f"bound_s_{m}", step=0.1)
                        auto_fit_bounds[m] = {'decay': d_range, 'k_ratio': k_range, 'slope': s_range}
                        st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_auto, col_desc = st.columns([1, 3], gap="large")
            with col_auto:
                if st.button("制約付きで最適化アルゴリズムを実行", use_container_width=True):
                    with st.spinner("アルゴリズムが最適なパラメータを探索中..."):
                        def objective_r2(params):
                            temp_X = pd.DataFrame()
                            for i, m in enumerate(media_cols):
                                current_lag = st.session_state.get(f"lag_m_{m}", 0)
                                lagged_val = apply_lag(df_clean[m].values, current_lag)
                                
                                adstocked = apply_adstock(lagged_val, params[3*i])
                                k_val = np.max(adstocked) * params[3*i+1] if np.max(adstocked) > 0 else 1.0
                                temp_X[m] = apply_hill_saturation(adstocked, k_val, params[3*i+2])
                                
                            if synergy_col_name:
                                temp_X[synergy_col_name] = temp_X[synergy_m1] * temp_X[synergy_m2]
                                
                            if intermediate_col != "使用しない": temp_X[intermediate_col] = df_clean[intermediate_col].values - df_clean[intermediate_col].mean()
                            for c in control_cols: temp_X[c] = df_clean[c].values - df_clean[c].mean()
                            time_feats = get_time_features(len(df_clean), data_granularity, use_trend, use_seasonality)
                            for c in time_feats.columns: temp_X[c] = time_feats[c].values - time_feats[c].mean()
                            if use_outlier_detection:
                                outlier_df, outlier_cols = get_outlier_features(df_clean[target_col].values, outlier_threshold)
                                for c in outlier_cols: temp_X[c] = outlier_df[c].values - outlier_df[c].mean()
                                
                            y = df_clean[target_col].values
                            
                            dummy_model = NonNegativeMediaModel(model_media_cols, scaling_method=scaling_method, estimator_type=estimator_type)
                            X_scaled = dummy_model._fit_scale(temp_X)
                            
                            y_mean_val = np.mean(y) if abs(np.mean(y)) > 1e-5 else 1.0
                            y_scaled = y / y_mean_val
                            
                            model = dummy_model._get_estimator().fit(X_scaled, y_scaled)
                            
                            penalty = 0
                            coef_dict = dict(zip(X_scaled.columns, model.coef_))
                            for m in model_media_cols:
                                c = coef_dict.get(m, 0)
                                if c < 0:
                                    penalty += (abs(c) ** 2) * 1000.0  
                                    
                            return -model.score(X_scaled, y_scaled) + penalty

                        initial_guess, bounds = [], []
                        for m in media_cols:
                            d_min, d_max = auto_fit_bounds[m]['decay']
                            k_min, k_max = auto_fit_bounds[m]['k_ratio']
                            s_min, s_max = auto_fit_bounds[m]['slope']
                            bounds.extend([(d_min, d_max), (k_min, k_max), (s_min, s_max)])
                            initial_guess.extend([
                                np.clip(st.session_state.get(f"decay_m_{m}", 0.5), d_min, d_max), 
                                np.clip(st.session_state.get(f"k_m_{m}", 0.5), k_min, k_max),
                                np.clip(st.session_state.get(f"s_m_{m}", 1.0), s_min, s_max)
                            ])
                        
                        result = minimize(objective_r2, initial_guess, method='L-BFGS-B', bounds=bounds)
                        if result.success:
                            for i, m in enumerate(media_cols):
                                st.session_state[f"decay_m_{m}"] = round(result.x[3*i], 2)
                                st.session_state[f"k_m_{m}"] = round(result.x[3*i+1], 2)
                                st.session_state[f"s_m_{m}"] = round(result.x[3*i+2], 2)
                            st.rerun() 
                        else:
                            st.error("最適化に失敗しました。制約範囲や変数を見直してください。")
            with col_desc:
                st.info("設定した制約の範囲内で、最も予測精度が高くなる組み合わせをアルゴリズムが全自動で探索・設定します。")
                st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>※「遅延 (Lag)」は自動最適化の対象外です。下部のスライダーで手動設定したLag値が前提として使われます。</p>", unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 2.5rem 0; border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)

            # --- 手動調整用UI ---
            decay_rates, hill_k_ratios, hill_s_slopes = {}, {}, {}
            for m in media_cols:
                if f"lag_m_{m}" not in st.session_state: st.session_state[f"lag_m_{m}"] = 0
                if f"decay_m_{m}" not in st.session_state: st.session_state[f"decay_m_{m}"] = 0.5
                if f"k_m_{m}" not in st.session_state: st.session_state[f"k_m_{m}"] = 0.5
                if f"s_m_{m}" not in st.session_state: st.session_state[f"s_m_{m}"] = 1.0
                
                with st.container():
                    col_sliders, col_chart = st.columns([1.2, 3], gap="large")
                    with col_sliders:
                        st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 1.25rem; padding-bottom: 0.25rem; border-bottom: 1px solid rgba(0,0,0,0.05);'><div style='width: 8px; height: 16px; border-radius: 4px; background-color: {media_color_map[m]}; margin-right: 10px;'></div><div style='font-weight: 700; font-size: 1.05rem; color: {COLOR_TEXT_MAIN};'>{m}</div></div>", unsafe_allow_html=True)
                        
                        lag_val = st.slider("遅延 (Lag)", 0, 8, key=f"lag_m_{m}", step=1)
                        decay_rates[m] = st.slider("残存 (Decay)", 0.0, 0.9, key=f"decay_m_{m}", step=0.01)
                        hill_k_ratios[m] = st.slider("半飽和点 (K ratio)", 0.1, 1.0, key=f"k_m_{m}", step=0.01)
                        hill_s_slopes[m] = st.slider("S字の強さ (Slope)", 0.5, 3.0, key=f"s_m_{m}", step=0.01)
                    
                    with col_chart:
                        X_temp = pd.DataFrame()
                        time_features = get_time_features(len(df_clean), data_granularity, use_trend, use_seasonality)
                        for col in media_cols:
                            d = decay_rates.get(col, st.session_state.get(f"decay_m_{col}", 0.5))
                            k_ratio = hill_k_ratios.get(col, st.session_state.get(f"k_m_{col}", 0.5))
                            s_slope = hill_s_slopes.get(col, st.session_state.get(f"s_m_{col}", 1.0))
                            current_lag_c = st.session_state.get(f"lag_m_{col}", 0)
                            
                            lagged = apply_lag(df_clean[col].values, current_lag_c)
                            adst = apply_adstock(lagged, d)
                            kval = np.max(adst) * k_ratio if np.max(adst) > 0 else 1.0
                            X_temp[col] = apply_hill_saturation(adst, kval, s_slope)
                            
                        if synergy_col_name:
                            X_temp[synergy_col_name] = X_temp[synergy_m1] * X_temp[synergy_m2]
                            
                        if intermediate_col != "使用しない": X_temp[intermediate_col] = df_clean[intermediate_col].values - df_clean[intermediate_col].mean()
                        for col in control_cols: X_temp[col] = df_clean[col].values - df_clean[col].mean()
                        for col in time_features.columns: X_temp[col] = time_features[col].values - time_features[col].mean()
                            
                        if use_outlier_detection:
                            temp_outlier_df, temp_outlier_cols = get_outlier_features(df_clean[target_col].values, outlier_threshold)
                            for col in temp_outlier_cols:
                                X_temp[col] = temp_outlier_df[col].values - temp_outlier_df[col].mean()

                        y_true = df_clean[target_col].values
                        temp_model = NonNegativeMediaModel(model_media_cols, scaling_method=scaling_method, estimator_type=estimator_type).fit(X_temp, y_true)
                        coef_temp = dict(zip(X_temp.columns, temp_model.coef_))
                        
                        y_pred_others = np.full(len(y_true), temp_model.intercept_)
                        for c in X_temp.columns:
                            if c != m: y_pred_others += X_temp[c].values * coef_temp.get(c, 0)
                        
                        y_partial = y_true - y_pred_others
                        
                        lagged_m = apply_lag(df_clean[m].values, lag_val)
                        x_ad = apply_adstock(lagged_m, decay_rates[m])
                        
                        df_scatter = pd.DataFrame({'x': x_ad, 'y': y_partial}).dropna()
                        
                        x_max_val = max(x_ad) if len(x_ad) > 0 and max(x_ad) > 0 else 1.0
                        x_smooth = np.linspace(0, x_max_val, 100)
                        K_val = x_max_val * hill_k_ratios[m] if x_max_val > 0 else 1.0
                        y_curve = apply_hill_saturation(x_smooth, K_val, hill_s_slopes[m]) * coef_temp.get(m, 0)
                        df_line = pd.DataFrame({'x': x_smooth, 'y': y_curve}).dropna()
                        
                        x_domain = [0, float(x_max_val * 1.05)]
                        y_min, y_max = float(min(y_partial)), float(max(y_partial))
                        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 100.0
                        y_domain = [y_min - y_margin, y_max + y_margin]
                        
                        scatter = alt.Chart(df_scatter).mark_circle(color=media_color_map[m]).encode(
                            x=alt.X('x:Q', scale=alt.Scale(domain=x_domain), title="", axis=alt.Axis(labels=False, ticks=False, grid=False)), 
                            y=alt.Y('y:Q', scale=alt.Scale(domain=y_domain), title="予測貢献量", axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_TEXT_MAIN, titleFontWeight='normal', titleFontSize=11)),
                            size=alt.value(70),
                            opacity=alt.value(0.7),
                            tooltip=[alt.Tooltip('x:Q', format=',.0f', title='投資額(Adstock後)'), alt.Tooltip('y:Q', format=',.0f', title='予測貢献量')]
                        )
                        line = alt.Chart(df_line).mark_line(color=COLOR_TEXT_MAIN).encode(
                            x=alt.X('x:Q', scale=alt.Scale(domain=x_domain)), 
                            y=alt.Y('y:Q', scale=alt.Scale(domain=y_domain)),
                            strokeWidth=alt.value(2.5)
                        )
                        
                        st.altair_chart(
                            (scatter + line).properties(height=200).configure_view(strokeWidth=0).configure(background='transparent'), 
                            use_container_width=True, 
                            theme="streamlit"
                        )
                
                st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                if st.button("分析を確定して次へ", type="primary", use_container_width=True, key="btn_run_bottom"):
                    st.session_state.run_analysis_flag = True

            if st.session_state.get("run_analysis_flag", False):
                with st.spinner("モデルを構築・評価中..."):
                    time.sleep(0.5)
                    X_analysis = pd.DataFrame()
                    K_values_saved = {}
                    S_values_saved = {}
                    for col in media_cols:
                        current_lag_final = st.session_state.get(f"lag_m_{col}", 0)
                        lagged = apply_lag(df_clean[col].values, current_lag_final)
                        
                        adstocked = apply_adstock(lagged, decay_rates[col])
                        K = np.max(adstocked) * hill_k_ratios[col] if np.max(adstocked) > 0 else 1.0
                        X_analysis[col] = apply_hill_saturation(adstocked, K, hill_s_slopes[col])
                        K_values_saved[col] = K
                        S_values_saved[col] = hill_s_slopes[col]
                    
                    if synergy_col_name:
                        X_analysis[synergy_col_name] = X_analysis[synergy_m1] * X_analysis[synergy_m2]
                        
                    if intermediate_col != "使用しない": X_analysis[intermediate_col] = df_clean[intermediate_col].values - df_clean[intermediate_col].mean()
                    for col in control_cols: X_analysis[col] = df_clean[col].values - df_clean[col].mean()
                    for col in time_features.columns: X_analysis[col] = time_features[col].values - time_features[col].mean()

                    outlier_df, outlier_cols = pd.DataFrame(), []
                    if use_outlier_detection:
                        outlier_df, outlier_cols = get_outlier_features(df_clean[target_col].values, outlier_threshold)
                        for col in outlier_cols: X_analysis[col] = outlier_df[col].values - outlier_df[col].mean()

                    y_analysis = df_clean[target_col].values
                    model = NonNegativeMediaModel(model_media_cols, scaling_method=scaling_method, estimator_type=estimator_type).fit(X_analysis, y_analysis)
                    
                    intermediate_results = None
                    if intermediate_col != "使用しない":
                        X_inter = pd.DataFrame()
                        inter_media_cols = [m for m in media_cols if m not in exclude_from_inter]
                        for col in inter_media_cols: 
                            X_inter[col] = X_analysis[col] 
                        for col in time_features.columns: 
                            X_inter[col] = time_features[col].values - time_features[col].mean()
                        
                        model_inter = NonNegativeMediaModel(inter_media_cols, scaling_method=scaling_method, estimator_type=estimator_type).fit(X_inter, df_clean[intermediate_col].values)
                        inter_coefs = dict(zip(X_inter.columns, model_inter.coef_))
                        target_coefs = dict(zip(X_analysis.columns, model.coef_))
                        e = target_coefs.get(intermediate_col, 0) 
                        total_coefs, direct_coefs, indirect_coefs = {}, {}, {}
                        for m in media_cols:
                            c = target_coefs.get(m, 0) 
                            a = inter_coefs.get(m, 0)  
                            direct_coefs[m] = c
                            indirect_coefs[m] = a * e
                            total_coefs[m] = c + (a * e) 
                        intermediate_results = {'model_inter': model_inter, 'total_coefs': total_coefs, 'direct_coefs': direct_coefs, 'indirect_coefs': indirect_coefs}
                    
                    y_pred = model.predict(X_analysis)
                    
                    exclude_from_vif = []
                    if intermediate_col != "使用しない":
                        exclude_from_vif.append(intermediate_col)
                    if synergy_col_name:
                        exclude_from_vif.append(synergy_col_name)
                        
                    st.session_state.analysis_results = {
                        'model': model, 'media_cols': media_cols, 'control_cols': control_cols, 'intermediate_col': intermediate_col,
                        'intermediate_results': intermediate_results, 'time_features': time_features, 'use_trend': use_trend, 'use_seasonality': use_seasonality,
                        'use_outlier_detection': use_outlier_detection, 'outlier_cols': outlier_cols, 'outlier_df': outlier_df, 'decay_rates': decay_rates,
                        'hill_k_ratios': hill_k_ratios, 'hill_K_values': K_values_saved, 'hill_S_values': S_values_saved, 
                        'synergy_col_name': synergy_col_name, 'synergy_m1': synergy_m1, 'synergy_m2': synergy_m2, 
                        'target_type': target_type, 'margin_rate': margin_rate, 
                        'intercept': model.intercept_, 'coefficients': dict(zip(X_analysis.columns, model.coef_)),
                        'r2': model.score(X_analysis, y_analysis), 'mape': calculate_mape(y_analysis, y_pred), 'dw': calculate_durbin_watson(y_analysis, y_pred), 
                        'y_true': y_analysis, 'y_pred': y_pred, 
                        'vif_df': calculate_vif(X_analysis, exclude_cols=exclude_from_vif), 'X_transformed': X_analysis
                    }
                    
                    st.session_state.last_sim_result = None
                    st.session_state.saved_scenarios = []
                    st.session_state.run_analysis_flag = False
                    st.session_state.active_tab = "分析オーバービュー"
                    st.session_state.scroll_to_top = True
                    st.rerun()

        else:
            st.info("サイドバーでメディア変数を選択してください。")

    # ==========================================
    # TAB 2: オーバービュー (分析結果)
    # ==========================================
    elif current_tab == "分析オーバービュー":
        if st.session_state.analysis_results:
            res = st.session_state.analysis_results
            
            t_type = res.get('target_type', "売上金額 (Currency)")
            is_cv = (t_type != "売上金額 (Currency)")
            margin_r = res.get('margin_rate', 1.0)
            
            if date_col != "指定なし":
                parsed_dates = pd.to_datetime(df_clean[date_col], errors='coerce')
                date_vals = df_clean.index.values if parsed_dates.isna().any() else parsed_dates.values
                x_type = 'Q' if parsed_dates.isna().any() else 'T'
            else:
                date_vals = df_clean.index.values
                x_type = 'Q'

            decomp_df = pd.DataFrame(index=date_vals)
            decomp_df['1_ベース (固定)'] = np.full(len(df_clean), res['intercept'])
            if len(res['control_cols']) > 0:
                control_contrib = np.zeros(len(df_clean))
                for c in res['control_cols']: control_contrib += (df_clean[c].values - df_clean[c].mean()) * res['coefficients'][c]
                decomp_df['2_外部要因'] = control_contrib
            time_feat = res['time_features']
            if res['use_trend'] and 'trend' in res['coefficients']:
                decomp_df['3_トレンド'] = (time_feat['trend'].values - time_feat['trend'].mean()) * res['coefficients']['trend']
            if res['use_seasonality'] and 'seasonality_sin' in res['coefficients']:
                sin_vals = time_feat['seasonality_sin'].values - time_feat['seasonality_sin'].mean()
                cos_vals = time_feat['seasonality_cos'].values - time_feat['seasonality_cos'].mean()
                decomp_df['4_季節性'] = sin_vals * res['coefficients']['seasonality_sin'] + cos_vals * res['coefficients']['seasonality_cos']
            if res.get('use_outlier_detection') and len(res['outlier_cols']) > 0:
                outlier_contrib = np.zeros(len(df_clean))
                for c in res['outlier_cols']:
                    if c in res['coefficients']: outlier_contrib += (res['outlier_df'][c].values - res['outlier_df'][c].mean()) * res['coefficients'][c]
                decomp_df['5_特殊要因 (外れ値)'] = outlier_contrib
            if res.get('intermediate_results'):
                ic = res['intermediate_col']
                decomp_df[f'6_パス ({ic})'] = res['X_transformed'][ic].values * res['coefficients'][ic]
            for m in res['media_cols']:
                decomp_df[f'7_Media ({m})'] = res['X_transformed'][m].values * res['coefficients'][m]
            
            if res.get('synergy_col_name') and res['synergy_col_name'] in res['coefficients']:
                syn_name = res['synergy_col_name']
                decomp_df[f'8_{syn_name}'] = res['X_transformed'][syn_name].values * res['coefficients'][syn_name]

            st.markdown(f'<div style="background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(24px) saturate(120%); border-left: 4px solid {COLOR_PRIMARY}; border-radius: 1rem; padding: 1.25rem 1.5rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255, 255, 255, 1); border-top: 1px solid rgba(255,255,255,0.8); border-right: 1px solid rgba(255,255,255,0.8); border-bottom: 1px solid rgba(255,255,255,0.8);"><h4 style="margin: 0 0 0.25rem 0; color: {COLOR_TEXT_MAIN}; font-size: 1.1rem; font-weight: 700;">ステップ 2: 分析結果の確認</h4><p style="margin: 0; color: {COLOR_GRAY}; font-size: 0.9rem; line-height: 1.6;">構築したモデルによる過去の要因分解と、各メディアの投資対効果を確認します。内容に問題がなければ、次の<strong>「予算シミュレーター」</strong>タブで最適な予算配分を計算してください。</p></div>', unsafe_allow_html=True)

            st.markdown("<h2 class='section-title'>ダッシュボード概要</h2>", unsafe_allow_html=True)
            
            total_sales = max(1, res['y_true'].sum())
            marketing_sales = sum([decomp_df[f'7_Media ({m})'].sum() for m in res['media_cols']])
            if res.get('synergy_col_name') and f"8_{res['synergy_col_name']}" in decomp_df.columns:
                marketing_sales += decomp_df[f"8_{res['synergy_col_name']}"].sum()
                
            marketing_pct = (marketing_sales / total_sales) * 100
            total_invest = sum([df_clean[m].sum() for m in res['media_cols']])
            vif_alert = (res['vif_df']['VIFスコア'].max() >= 10) if not res['vif_df'].empty else False

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="medium")
            if is_cv:
                with kpi_col1: st.markdown(render_kpi_card("対象期間の総CV数", f"{int(total_sales):,} 件", svg_icon=ICON_USERS), unsafe_allow_html=True)
                with kpi_col2: st.markdown(render_kpi_card("マーケティング貢献比率", f"{marketing_pct:.1f}%", change="CVの牽引力", svg_icon=ICON_PIE), unsafe_allow_html=True)
                avg_cpa = total_invest / max(1, marketing_sales)
                with kpi_col3: st.markdown(render_kpi_card("平均 獲得単価 (CPA)", f"¥ {int(avg_cpa):,}", trend="down" if avg_cpa > 0 else "neutral", svg_icon=ICON_ZAP, reverse_trend=True), unsafe_allow_html=True)
            else:
                with kpi_col1: st.markdown(render_kpi_card("対象期間の総売上", f"¥ {int(total_sales):,}", svg_icon=ICON_DOLLAR), unsafe_allow_html=True)
                with kpi_col2: st.markdown(render_kpi_card("マーケティング貢献比率", f"{marketing_pct:.1f}%", change="売上の牽引力", svg_icon=ICON_PIE), unsafe_allow_html=True)
                eff_title = "平均 ROI (利益ベース)" if margin_r < 1.0 else "平均 投資対効果 (ROAS)"
                avg_eff = (marketing_sales * margin_r) / max(1, total_invest)
                with kpi_col3: st.markdown(render_kpi_card(eff_title, f"{avg_eff * 100:.1f}%", trend="up" if avg_eff >= 0 else "down", svg_icon=ICON_ZAP), unsafe_allow_html=True)
                
            with kpi_col4: st.markdown(render_kpi_card("モデル予測精度 (R²)", f"{res['r2']:.2f}", change="警告あり" if vif_alert else "正常", trend="down" if vif_alert else "up", svg_icon=ICON_TARGET), unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)

            roas_list = []
            has_negative_roas = False
            
            synergy_sales_total = 0
            if res.get('synergy_col_name') and res['synergy_col_name'] in res['coefficients']:
                synergy_sales_total = np.sum(res['X_transformed'][res['synergy_col_name']].values * res['coefficients'][res['synergy_col_name']])
            
            for m in res['media_cols']:
                media_invest = df_clean[m].sum()
                if media_invest <= 0:
                    roas_list.append({'メディア': m, '相対的な効率': 0})
                    continue
                
                syn_contrib_for_m = 0
                if synergy_sales_total > 0 and m in [res.get('synergy_m1'), res.get('synergy_m2')]:
                    inv_m1 = df_clean[res['synergy_m1']].sum()
                    inv_m2 = df_clean[res['synergy_m2']].sum()
                    sum_inv = inv_m1 + inv_m2
                    if sum_inv > 0:
                        syn_contrib_for_m = synergy_sales_total * (media_invest / sum_inv)

                if res.get('intermediate_results'):
                    ir = res['intermediate_results']
                    total_contrib = np.sum(res['X_transformed'][m].values * ir['total_coefs'][m]) + syn_contrib_for_m
                else:
                    total_contrib = np.sum(res['X_transformed'][m].values * res['coefficients'][m]) + syn_contrib_for_m
                    
                if is_cv:
                    roas = (total_contrib / media_invest) * 1000000 
                else:
                    roas = (total_contrib * margin_r) / media_invest 
                    
                if roas <= 0: has_negative_roas = True
                roas_list.append({'メディア': m, '相対的な効率': roas})
                
            roas_df = pd.DataFrame(roas_list).set_index('メディア').sort_values('相対的な効率', ascending=False)
            best_media = roas_df.index[0] if not roas_df.empty else "-"

            if has_negative_roas:
                insight_html = f'<div class="insight-box" style="border-left-color: {COLOR_ROSE};"><div class="insight-icon" style="color: {COLOR_ROSE};">{ICON_ALERT}</div><div class="insight-content"><h4>モデル構造の警告</h4><p>一部のメディアの投資対効果（ROAS）が「ゼロ」として推定（無効化）されています。これは、各メディアの出稿タイミングが似ている（多重共線性）か、モデルのパラメータ設定が実態と合っていないことを示唆しています。「モデルチューニング」タブで減衰率や飽和度を手動で調整するか、相関の強い変数を除外して再分析をお勧めします。</p></div></div>'
            else:
                insight_html = f'<div class="insight-box"><div class="insight-icon">{ICON_LIGHTBULB}</div><div class="insight-content"><h4>統計モデリング インサイト</h4><p>対象期間において、最も投資対効果が高いチャネルは <strong style="color: {COLOR_PRIMARY}; background: rgba(255,255,255,0.8); padding: 2px 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,1); box-shadow: 0 2px 4px rgba(0,0,0,0.05);">{best_media}</strong> と推定されます。<br>次フェーズの予算策定では、このチャネルへの投資比率を高めることで、全体の効率をさらに改善できる可能性が高いです。「予算シミュレーター」タブで最適な予算配分を検証してください。</p></div></div>'
            
            st.markdown(insight_html, unsafe_allow_html=True)

            col_chart1, col_chart2 = st.columns([5, 3], gap="large")
            
            x_axis_args = {"grid": False, "labelColor": COLOR_GRAY, "domainColor": "rgba(0,0,0,0.1)", "labelFontSize": 11, "titleColor": COLOR_GRAY, "titleFontWeight": "normal"}
            if x_type == 'T':
                x_axis_args["format"] = "%y/%m/%d"

            with col_chart1:
                y_title_decomp = "CV貢献量 (件)" if is_cv else "売上貢献量 (¥)"
                st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>要因別 貢献推移 (Decomposition)</h4>", unsafe_allow_html=True)
                decomp_df_plot = decomp_df.copy() 
                decomp_df_plot['Date'] = date_vals
                decomp_df_long = decomp_df_plot.melt(id_vars=['Date'], var_name='要因', value_name='貢献量')

                domain = ['1_ベース (固定)', '2_外部要因', '3_トレンド', '4_季節性', '5_特殊要因 (外れ値)']
                range_ = [COLOR_BASELINE, COLOR_GRAY, COLOR_TREND, COLOR_SEASON, COLOR_OUTLIER]
                if res.get('intermediate_results'):
                    domain.append(f'6_パス ({res["intermediate_col"]})')
                    range_.append(COLOR_INTER)
                for m in res['media_cols']:
                    domain.append(f'7_Media ({m})')
                    range_.append(media_color_map[m])
                
                if res.get('synergy_col_name'):
                    domain.append(f"8_{res['synergy_col_name']}")
                    range_.append(COLOR_SYNERGY)

                area_chart = alt.Chart(decomp_df_long).mark_area(opacity=0.9).encode(
                    x=alt.X(f'Date:{x_type}', title=None, axis=alt.Axis(**x_axis_args)),
                    y=alt.Y('貢献量:Q', title=y_title_decomp, stack='zero', axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_GRAY, titleFontWeight='normal', titleFontSize=11)),
                    color=alt.Color('要因:N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(orient='bottom', title=None, labelColor=COLOR_GRAY, labelFontSize=11)),
                    order=alt.Order('要因:N', sort='ascending'),
                    tooltip=[
                        alt.Tooltip(f'Date:{x_type}', title='時間', format='%y/%m/%d' if x_type == 'T' else ''), 
                        alt.Tooltip('要因:N'), 
                        alt.Tooltip('貢献量:Q', format=',.0f')
                    ]
                ).properties(height=360)
                
                zero_line_decomp = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=COLOR_GRAY, strokeDash=[4,4]).encode(y='y:Q')
                st.altair_chart((area_chart + zero_line_decomp).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")

            with col_chart2:
                st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>チャネル別 投資効率比較</h4>", unsafe_allow_html=True)
                
                x_axis_title = "100万円あたりの獲得CV数" if is_cv else ("ROI (利益回収率)" if margin_r < 1.0 else "ROAS (売上回収率)")
                x_axis_format = ',.1f' if is_cv else '.2%'
                x_axis_roas = alt.Axis(format=x_axis_format, labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_GRAY, titleFontWeight='normal', titleFontSize=11)
                
                if res.get('intermediate_results'):
                    ir = res['intermediate_results']
                    roas_data = []
                    for m in res['media_cols']:
                        media_invest = df_clean[m].sum()
                        if media_invest > 0:
                            direct_contrib = np.sum(res['X_transformed'][m].values * ir['direct_coefs'][m])
                            indirect_contrib = np.sum(res['X_transformed'][m].values * ir['indirect_coefs'][m])
                            
                            syn_contrib_for_m = 0
                            if synergy_sales_total > 0 and m in [res.get('synergy_m1'), res.get('synergy_m2')]:
                                inv_m1 = df_clean[res['synergy_m1']].sum()
                                inv_m2 = df_clean[res['synergy_m2']].sum()
                                sum_inv = inv_m1 + inv_m2
                                if sum_inv > 0: syn_contrib_for_m = synergy_sales_total * (media_invest / sum_inv)
                            
                            total_contrib = direct_contrib + indirect_contrib + syn_contrib_for_m
                            direct_pct = (direct_contrib / total_contrib) if total_contrib > 0 else 0
                            indirect_pct = (indirect_contrib / total_contrib) if total_contrib > 0 else 0
                            syn_pct = (syn_contrib_for_m / total_contrib) if total_contrib > 0 else 0
                            
                            if is_cv:
                                eff_direct = (direct_contrib / media_invest) * 1000000
                                eff_indirect = (indirect_contrib / media_invest) * 1000000
                                eff_syn = (syn_contrib_for_m / media_invest) * 1000000
                                total_eff = (total_contrib / media_invest) * 1000000
                                cpa_val = media_invest / total_contrib if total_contrib > 0 else 0
                            else:
                                eff_direct = (direct_contrib * margin_r) / media_invest
                                eff_indirect = (indirect_contrib * margin_r) / media_invest
                                eff_syn = (syn_contrib_for_m * margin_r) / media_invest
                                total_eff = (total_contrib * margin_r) / media_invest
                                cpa_val = 0
                            
                            roas_data.append({"メディア": m, "効果の種類": "直接効果", "効率": eff_direct, "割合": direct_pct, "全体効率": total_eff, "全体CPA": cpa_val})
                            roas_data.append({"メディア": m, "効果の種類": f"間接 ({res['intermediate_col']})", "効率": eff_indirect, "割合": indirect_pct, "全体効率": total_eff, "全体CPA": cpa_val})
                            if syn_contrib_for_m > 0:
                                roas_data.append({"メディア": m, "効果の種類": f"相乗効果", "効率": eff_syn, "割合": syn_pct, "全体効率": total_eff, "全体CPA": cpa_val})
                        else:
                            roas_data.append({"メディア": m, "効果の種類": "直接効果", "効率": 0, "割合": 0, "全体効率": 0, "全体CPA": 0})
                            roas_data.append({"メディア": m, "効果の種類": f"間接 ({res['intermediate_col']})", "効率": 0, "割合": 0, "全体効率": 0, "全体CPA": 0})
                    
                    df_roas_plot = pd.DataFrame(roas_data)
                    
                    base_roas = alt.Chart(df_roas_plot).encode(
                        x=alt.X('効率:Q', title=x_axis_title, axis=x_axis_roas),
                        y=alt.Y('メディア:N', sort=alt.EncodingSortField(field='全体効率', op='max', order='descending'), title=None, axis=alt.Axis(labelColor=COLOR_GRAY, labelFontWeight='normal', labelFontSize=11, ticks=False)),
                        order=alt.Order('効果の種類:N', sort='ascending')
                    )
                    
                    color_domain = ['直接効果', f"間接 ({res['intermediate_col']})", '相乗効果']
                    color_range = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SYNERGY]
                    
                    tooltip_list = [alt.Tooltip('メディア:N'), alt.Tooltip('効果の種類:N'), alt.Tooltip('効率:Q', format=x_axis_format, title='効率スコア'), alt.Tooltip('割合:Q', format='.1%', title='構成比')]
                    if is_cv: tooltip_list.append(alt.Tooltip('全体CPA:Q', format=',.0f', title='平均CPA (¥)'))
                    
                    bars = base_roas.mark_bar(cornerRadiusEnd=4).encode(
                        color=alt.Color('効果の種類:N', scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(orient='bottom', title=None)),
                        tooltip=tooltip_list
                    )
                    
                    text = base_roas.mark_text(align='right', dx=-5, color='white', fontSize=10, fontWeight='bold').encode(
                        x=alt.X('効率:Q', stack='zero'),
                        detail='効果の種類:N',
                        text=alt.Text('割合:Q', format='.0%'),
                        opacity=alt.condition(alt.datum.割合 > 0.08, alt.value(1), alt.value(0))
                    )
                    
                    roas_chart = (bars + text).properties(height=360)
                    
                    zero_rule_roas = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=COLOR_GRAY, strokeDash=[4,4]).encode(x='x:Q')
                    st.altair_chart((roas_chart + zero_rule_roas).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")
                else:
                    roas_data_simple = []
                    for m in res['media_cols']:
                        media_invest = df_clean[m].sum()
                        if media_invest > 0:
                            direct_contrib = np.sum(res['X_transformed'][m].values * res['coefficients'][m])
                            syn_contrib_for_m = 0
                            if synergy_sales_total > 0 and m in [res.get('synergy_m1'), res.get('synergy_m2')]:
                                inv_m1 = df_clean[res['synergy_m1']].sum()
                                inv_m2 = df_clean[res['synergy_m2']].sum()
                                sum_inv = inv_m1 + inv_m2
                                if sum_inv > 0: syn_contrib_for_m = synergy_sales_total * (media_invest / sum_inv)
                            
                            total_contrib = direct_contrib + syn_contrib_for_m
                            
                            if is_cv:
                                eff_direct = (direct_contrib / media_invest) * 1000000
                                eff_syn = (syn_contrib_for_m / media_invest) * 1000000
                                total_eff = (total_contrib / media_invest) * 1000000
                                cpa_val = media_invest / total_contrib if total_contrib > 0 else 0
                            else:
                                eff_direct = (direct_contrib * margin_r) / media_invest
                                eff_syn = (syn_contrib_for_m * margin_r) / media_invest
                                total_eff = (total_contrib * margin_r) / media_invest
                                cpa_val = 0
                                
                            roas_data_simple.append({"メディア": m, "効果の種類": "単独効果", "効率": eff_direct, "全体効率": total_eff, "全体CPA": cpa_val})
                            if syn_contrib_for_m > 0:
                                roas_data_simple.append({"メディア": m, "効果の種類": "相乗効果", "効率": eff_syn, "全体効率": total_eff, "全体CPA": cpa_val})
                        else:
                            roas_data_simple.append({"メディア": m, "効果の種類": "単独効果", "効率": 0, "全体効率": 0, "全体CPA": 0})
                    
                    df_roas_plot_simple = pd.DataFrame(roas_data_simple)
                    
                    base_roas_simple = alt.Chart(df_roas_plot_simple).encode(
                        x=alt.X('効率:Q', title=x_axis_title, axis=x_axis_roas),
                        y=alt.Y('メディア:N', sort=alt.EncodingSortField(field='全体効率', op='max', order='descending'), title=None, axis=alt.Axis(labelColor=COLOR_GRAY, labelFontWeight='normal', labelFontSize=11, ticks=False)),
                        order=alt.Order('効果の種類:N', sort='ascending')
                    )
                    
                    tooltip_list_s = [alt.Tooltip('メディア:N'), alt.Tooltip('効果の種類:N'), alt.Tooltip('効率:Q', format=x_axis_format, title='効率スコア')]
                    if is_cv: tooltip_list_s.append(alt.Tooltip('全体CPA:Q', format=',.0f', title='平均CPA (¥)'))
                    
                    bars_simple = base_roas_simple.mark_bar(cornerRadiusEnd=4).encode(
                        color=alt.Color('効果の種類:N', scale=alt.Scale(domain=['単独効果', '相乗効果'], range=[COLOR_PRIMARY, COLOR_SYNERGY]), legend=alt.Legend(orient='bottom', title=None)),
                        tooltip=tooltip_list_s
                    )
                    
                    roas_chart_simple = bars_simple.properties(height=360)
                    zero_rule_roas = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=COLOR_GRAY, strokeDash=[4,4]).encode(x='x:Q')
                    st.altair_chart((roas_chart_simple + zero_rule_roas).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")

            # --- 因果パス図 ---
            if res.get('intermediate_results'):
                ir = res['intermediate_results']
                total_direct = sum([np.sum(res['X_transformed'][m].values * ir['direct_coefs'][m]) for m in res['media_cols']])
                total_indirect = sum([np.sum(res['X_transformed'][m].values * ir['indirect_coefs'][m]) for m in res['media_cols']])
                
                total_direct += synergy_sales_total
                
                media_details = []
                for m in res['media_cols']:
                    d_val = max(0, np.sum(res['X_transformed'][m].values * ir['direct_coefs'][m]))
                    i_val = max(0, np.sum(res['X_transformed'][m].values * ir['indirect_coefs'][m]))
                    
                    if synergy_sales_total > 0 and m in [res.get('synergy_m1'), res.get('synergy_m2')]:
                        inv_m1 = df_clean[res['synergy_m1']].sum()
                        inv_m2 = df_clean[res['synergy_m2']].sum()
                        sum_inv = inv_m1 + inv_m2
                        if sum_inv > 0: d_val += synergy_sales_total * (df_clean[m].sum() / sum_inv)
                        
                    media_details.append({'name': m, 'direct': d_val, 'indirect': i_val, 'total': d_val + i_val})
                media_details.sort(key=lambda x: x['total'], reverse=True)
                
                t_name_path = "CV成果 (Count)" if is_cv else "売上成果 (Sales)"
                st.markdown(render_path_diagram(total_direct, total_indirect, res['intermediate_col'], t_name_path, media_details), unsafe_allow_html=True)


            st.markdown("<hr style='margin: 3.5rem 0; border-color: rgba(0, 0, 0, 0.05);'>", unsafe_allow_html=True)
            
            with st.expander("詳細なモデル検証レポート (データサイエンティスト向け)", expanded=False):
                st.markdown(f"<h5 style='color: {COLOR_TEXT_MAIN}; margin-top: 1rem; margin-bottom: 1rem;'>モデル精度指標</h5>", unsafe_allow_html=True)
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("決定係数 (R²)", f"{res['r2']:.3f}")
                m_col2.metric("平均絶対誤差率 (MAPE)", f"{res['mape']:.1f}%")
                m_col3.metric("ダービン・ワトソン比 (DW)", f"{res['dw']:.2f}")

                if res['dw'] < 1.5 or res['dw'] > 2.5:
                    st.warning("ダービン・ワトソン比が1.5〜2.5の範囲外です。モデルに捉えきれていない時系列の自己相関（未観測の外部要因）が存在する可能性があります。")
                else:
                    st.success("ダービン・ワトソン比は正常範囲（1.5〜2.5）です。自己相関のリスクは低いです。")
                
                st.markdown(f"<h5 style='color: {COLOR_TEXT_MAIN}; margin-top: 2rem; margin-bottom: 1.5rem;'>実測値 vs 予測値 のトラッキング</h5>", unsafe_allow_html=True)
                fit_df = pd.DataFrame({'Date': date_vals, '実測値 (Actual)': res['y_true'], '予測値 (Predicted)': res['y_pred']}).melt('Date', var_name='種類', value_name='貢献量')
                
                fit_chart = alt.Chart(fit_df).mark_line(size=2).encode(
                    x=alt.X(f'Date:{x_type}', title=None, axis=alt.Axis(**x_axis_args)), 
                    y=alt.Y('貢献量:Q', title=None, axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2])),
                    color=alt.Color('種類:N', scale=alt.Scale(domain=['実測値 (Actual)', '予測値 (Predicted)'], range=[COLOR_BASELINE, COLOR_PRIMARY]), legend=alt.Legend(orient='top')),
                    strokeDash=alt.condition(alt.datum['種類'] == '予測値 (Predicted)', alt.value([4, 4]), alt.value([0])),
                    tooltip=[
                        alt.Tooltip(f'Date:{x_type}', title='時間', format='%y/%m/%d' if x_type == 'T' else ''), 
                        alt.Tooltip('種類:N'), 
                        alt.Tooltip('貢献量:Q', format=',.0f')
                    ]
                ).properties(height=300).configure_view(strokeWidth=0).configure(background='transparent')
                st.altair_chart(fit_chart, use_container_width=True, theme="streamlit")

                st.markdown(f"<h5 style='color: {COLOR_TEXT_MAIN}; margin-top: 2.5rem; margin-bottom: 1.5rem;'>残差分析 (Residuals)</h5>", unsafe_allow_html=True)
                resid_df = pd.DataFrame({'Date': date_vals, '残差': res['y_true'] - res['y_pred']})
                
                resid_chart = alt.Chart(resid_df).mark_line(size=2, color=COLOR_ROSE).encode(
                    x=alt.X(f'Date:{x_type}', title=None, axis=alt.Axis(**x_axis_args)),
                    y=alt.Y('残差:Q', title="残差", axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2])),
                    tooltip=[alt.Tooltip(f'Date:{x_type}', title='時間', format='%y/%m/%d' if x_type == 'T' else ''), alt.Tooltip('残差:Q', format=',.0f')]
                ).properties(height=250)
                
                zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=COLOR_GRAY, strokeDash=[4,4]).encode(y='y:Q')
                
                layered_resid_chart = (resid_chart + zero_line).configure_view(strokeWidth=0).configure(background='transparent')
                st.altair_chart(layered_resid_chart, use_container_width=True, theme="streamlit")

                st.markdown(f"<h5 style='color: {COLOR_TEXT_MAIN}; margin-top: 2.5rem; margin-bottom: 1rem;'>多重共線性 (VIF) レポート</h5>", unsafe_allow_html=True)
                if vif_alert: 
                    st.warning("""
                    **VIFが10を超える変数があります。多重共線性が起きており、一部の係数が不安定になっている可能性があります。**
                    
                    **【よくある原因と対策】**
                    1. **検索ボリュームと検索広告の同時投入:** 「パス解析用 中間変数」の検索ボリュームと、「メディア変数」の検索広告費の相関が極端に強い場合、VIFが高く出ます。係数が安定している（ROASが極端な0になっていない）場合はそのまま進めて問題ありませんが、結果が不自然な場合はどちらかを分析から外してください。
                    2. **出稿タイミングの重複:** TVCMとデジタル動画など、全く同じ期間に同じ波で出稿しているメディア同士が効果を奪い合っています。どちらか一方の変数を外すか、統合して「動画全体」として分析してください。
                    """)
                else:
                    st.success("全ての変数のVIFは10未満であり、多重共線性のリスクは低く抑えられています。")
                
                st.dataframe(
                    res['vif_df'].style.map(
                        lambda x: f"background-color: #FEF2F2; color: {COLOR_ROSE}; font-weight: bold;" if x >= 10 else "", 
                        subset=['VIFスコア']
                    ), 
                    use_container_width=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            col_dl1, col_dl2, btn_col = st.columns([1, 1, 2], gap="medium")
            with col_dl1: st.download_button("要因分解データ(CSV)", data=convert_df_to_csv(decomp_df_plot), file_name='decomposition.csv', mime='text/csv', use_container_width=True)
            with col_dl2: st.download_button("効率スコア(CSV)", data=convert_df_to_csv(roas_df), file_name='efficiency_scores.csv', mime='text/csv', use_container_width=True)

            with btn_col:
                if st.button("予算シミュレーターへ進む", type="primary", use_container_width=True):
                    st.session_state.active_tab = "予算シミュレーター"
                    st.session_state.scroll_to_top = True
                    st.rerun()

        else:
            st.info("「モデルチューニング」タブで分析を実行してください。")

    # ==========================================
    # TAB 3: 予算シミュレーター & 最適化画面
    # ==========================================
    elif current_tab == "予算シミュレーター":
        if st.session_state.analysis_results:
            res = st.session_state.analysis_results
            t_type = res.get('target_type', "売上金額 (Currency)")
            is_cv = (t_type != "売上金額 (Currency)")
            margin_r = res.get('margin_rate', 1.0)
            
            st.markdown("<h2 class='section-title'>予算アロケーション・シミュレーター</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 1rem;'>総予算と制約条件を入力し、最適化アルゴリズムに最も効率が高くなる予算配分を計算させます。</p>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if "週" in data_granularity:
                sim_period = st.selectbox("対象期間", ["1ヶ月 (4週)", "四半期 (13週)", "半年 (26週)", "1年間 (52週)"], index=0)
                periods_map = {"1ヶ月 (4週)": 4, "四半期 (13週)": 13, "半年 (26週)": 26, "1年間 (52週)": 52}
            elif "月" in data_granularity:
                sim_period = st.selectbox("対象期間", ["1ヶ月", "四半期 (3ヶ月)", "半年 (6ヶ月)", "1年間 (12ヶ月)"], index=0)
                periods_map = {"1ヶ月": 1, "四半期 (3ヶ月)": 3, "半年 (6ヶ月)": 6, "1年間 (12ヶ月)": 12}
            else: 
                sim_period = st.selectbox("対象期間", ["1ヶ月 (30日)", "四半期 (90日)", "半年 (180日)", "1年間 (365日)"], index=0)
                periods_map = {"1ヶ月 (30日)": 30, "四半期 (90日)": 90, "半年 (180日)": 180, "1年間 (365日)": 365}
            
            n_periods = periods_map[sim_period]
            
            base_per_period = res['intercept']
            
            if res['use_trend'] and 'trend' in res['coefficients']:
                time_feat = res['time_features']
                base_per_period += (time_feat['trend'].iloc[-1] - time_feat['trend'].mean()) * res['coefficients']['trend']

            # 重複していた def get_sim_coef() を削除 (トップレベルに移動済みのため)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_sim_left, col_sim_right = st.columns([1, 2], gap="large")
            
            with col_sim_left:
                st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 0.5rem; font-size: 1.25rem; display: flex; align-items: center; gap: 10px;'><svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='{COLOR_PRIMARY}' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='3'></circle><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z'></path></svg> 最適化の制約条件</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 0.9rem; margin-bottom: 2rem;'>各チャネルへの投資額の「下限」と「上限」を設定してください。</p>", unsafe_allow_html=True)
                
                current_total = sum([df_clean[m].mean() * n_periods for m in res['media_cols']])
                
                total_budget = st.number_input("対象期間の総予算枠 (¥)", min_value=0, value=int(current_total), step=10000)
                st.markdown(f"<div style='text-align: right; color: {COLOR_PRIMARY}; font-weight: 700; font-size: 0.95rem; margin-top: -10px; margin-bottom: 1rem;'>¥ {total_budget:,.0f}</div>", unsafe_allow_html=True)
                
                bounds_dict = {}
                for m in res['media_cols']:
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <div style='width: 6px; height: 16px; border-radius: 3px; background-color: {media_color_map[m]}; margin-right: 10px;'></div>
                        <div style='font-weight:700; font-size:1.05rem; color:{COLOR_TEXT_MAIN};'>{m}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    max_limit = int(df_clean[m].max() * 3 * n_periods) if df_clean[m].max() > 0 else 1000000 * n_periods
                    
                    col_min, col_max = st.columns(2)
                    with col_min:
                        min_val = st.number_input("下限 (Min)", min_value=0, max_value=max_limit, value=0, step=10000, key=f"bmin_{m}")
                        st.markdown(f"<div style='text-align: right; color: {COLOR_GRAY}; font-weight: 600; font-size: 0.85rem; margin-top: -10px;'>¥ {min_val:,.0f}</div>", unsafe_allow_html=True)
                    with col_max:
                        max_val = st.number_input("上限 (Max)", min_value=0, max_value=max_limit, value=max_limit, step=10000, key=f"bmax_{m}")
                        st.markdown(f"<div style='text-align: right; color: {COLOR_GRAY}; font-weight: 600; font-size: 0.85rem; margin-top: -10px;'>¥ {max_val:,.0f}</div>", unsafe_allow_html=True)
                        
                    if min_val > max_val: max_val = min_val
                    bounds_dict[m] = (min_val, max_val)
                    
                    st.markdown("<hr style='margin: 1rem 0; border-color: rgba(255,255,255,0.7);'>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                btn_col1, btn_col2, btn_col3 = st.columns([1, 4, 1])
                with btn_col2:
                    run_opt = st.button("最適化を実行", type="primary", use_container_width=True)

            with col_sim_right:
                if run_opt:
                    if sum([bounds_dict[m][0] for m in res['media_cols']]) > total_budget:
                        st.error("エラー: 下限の合計額が、総予算を超えています。")
                    else:
                        with st.spinner("最適化アルゴリズムが数千の配分パターンから最適解を計算中..."):
                            
                            safe_total = float(total_budget) if total_budget > 0 else 1.0
                            
                            def objective(x_ratios):
                                x_spends = x_ratios * safe_total
                                sales = base_per_period * n_periods
                                for idx, m in enumerate(res['media_cols']):
                                    steady = (x_spends[idx] / n_periods) / (1 - res['decay_rates'][m])
                                    sat = apply_hill_saturation(steady, res['hill_K_values'][m], res['hill_S_values'][m])
                                    sales += sat * get_sim_coef(res, m) * n_periods
                                return -(sales / safe_total) 
                            
                            cons = {'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)}
                            bnds = [(bounds_dict[m][0]/safe_total, bounds_dict[m][1]/safe_total) for m in res['media_cols']]
                            
                            x0_ratios = np.array([df_clean[m].mean() * n_periods for m in res['media_cols']]) / safe_total
                            x0_ratios = np.clip(x0_ratios, [b[0] for b in bnds], [b[1] for b in bnds])
                            if np.sum(x0_ratios) > 0: 
                                x0_ratios = x0_ratios / np.sum(x0_ratios) 
                            
                            opt_res_1 = minimize(objective, x0_ratios, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 2000, 'eps': 1e-3})
                            
                            coef_weights = np.array([max(0.01, get_sim_coef(res, m)) for m in res['media_cols']])
                            coef_weights = coef_weights / np.sum(coef_weights)
                            
                            x1_ratios = (x0_ratios + coef_weights) / 2.0
                            x1_ratios = np.clip(x1_ratios, [b[0] for b in bnds], [b[1] for b in bnds])
                            if np.sum(x1_ratios) > 0: 
                                x1_ratios = x1_ratios / np.sum(x1_ratios)
                                
                            opt_res_2 = minimize(objective, x1_ratios, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 2000, 'eps': 1e-3})
                            
                            best_res = opt_res_1 if opt_res_1.fun < opt_res_2.fun else opt_res_2
                            
                            if best_res.success or (best_res.fun < objective(x0_ratios)):
                                opt_spends = best_res.x * safe_total
                                opt_sales = -best_res.fun * safe_total
                                
                                current_sales = base_per_period * n_periods
                                for m in res['media_cols']:
                                    steady_curr = df_clean[m].mean() / (1 - res['decay_rates'][m])
                                    current_sales += apply_hill_saturation(steady_curr, res['hill_K_values'][m], res['hill_S_values'][m]) * get_sim_coef(res, m) * n_periods
                                
                                st.session_state.last_sim_result = {
                                    'opt_spends': opt_spends,
                                    'opt_sales': opt_sales,
                                    'current_sales': current_sales,
                                    'total_budget': total_budget,
                                    'media_budgets': {m: opt_spends[idx] for idx, m in enumerate(res['media_cols'])},
                                    'n_periods': n_periods,
                                    'flight_dates': [f"Period {i+1}" for i in range(n_periods)]
                                }
                                # 修正: 計算完了後にトップスクロールフラグを立てて画面をリロード
                                st.session_state.scroll_to_top = True
                                st.rerun()
                            else:
                                st.error("計算が収束しませんでした。制約条件を緩和して再度お試しください。")

                if st.session_state.get('last_sim_result'):
                    sim_res = st.session_state.last_sim_result
                    opt_spends = sim_res['opt_spends']
                    opt_sales = sim_res['opt_sales']
                    current_sales = sim_res['current_sales']
                    
                    st.markdown(render_sim_summary(opt_sales, current_sales, is_cv, sim_res['total_budget']), unsafe_allow_html=True)
                    
                    table_data = []
                    for idx, m in enumerate(res['media_cols']):
                        b_curr = df_clean[m].mean() * n_periods
                        b_opt = opt_spends[idx]
                        c = get_sim_coef(res, m)
                        steady_curr = df_clean[m].mean() / (1 - res['decay_rates'][m])
                        s_curr = apply_hill_saturation(steady_curr, res['hill_K_values'][m], res['hill_S_values'][m]) * c * n_periods
                        
                        steady_opt = (b_opt/n_periods) / (1 - res['decay_rates'][m])
                        s_opt = apply_hill_saturation(steady_opt, res['hill_K_values'][m], res['hill_S_values'][m]) * c * n_periods
                        
                        table_data.append({"メディア": m, "シナリオ": "1. 現状", "予算": b_curr, "予測貢献量": s_curr})
                        table_data.append({"メディア": m, "シナリオ": "2. 最適化", "予算": b_opt, "予測貢献量": s_opt})
                    
                    comp_df = pd.DataFrame(table_data)
                    
                    chart_col1, chart_col2 = st.columns(2, gap="large")
                    
                    with chart_col1:
                        st.markdown(f"<h5 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>予算配分の Before / After</h5>", unsafe_allow_html=True)
                        bar_chart_budget = alt.Chart(comp_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                            y=alt.Y('メディア:N', title=None, axis=alt.Axis(labelColor=COLOR_TEXT_MAIN, labelFontWeight='bold', grid=False, labelFontSize=12)),
                            x=alt.X('予算:Q', title="予算 (¥)", axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_GRAY, titleFontWeight='normal', titleFontSize=11)),
                            yOffset=alt.YOffset('シナリオ:N', sort=['1. 現状', '2. 最適化']),
                            color=alt.Color('シナリオ:N', scale=alt.Scale(domain=['1. 現状', '2. 最適化'], range=[COLOR_BASELINE, COLOR_PRIMARY]), legend=alt.Legend(orient='bottom', title=None, labelFontSize=11, labelColor=COLOR_GRAY)),
                            tooltip=[alt.Tooltip('メディア:N'), alt.Tooltip('シナリオ:N'), alt.Tooltip('予算:Q', format=',.0f')]
                        ).properties(height=350)
                        
                        zero_rule_budget = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=COLOR_GRAY, strokeDash=[4,4]).encode(x='x:Q')
                        st.altair_chart((bar_chart_budget + zero_rule_budget).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")

                    with chart_col2:
                        c_title = "予測CV数" if is_cv else "予測売上貢献"
                        st.markdown(f"<h5 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>{c_title}の Before / After</h5>", unsafe_allow_html=True)
                        bar_chart_sales = alt.Chart(comp_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                            y=alt.Y('メディア:N', title=None, axis=alt.Axis(labelColor=COLOR_TEXT_MAIN, labelFontWeight='bold', grid=False, labelFontSize=12)),
                            x=alt.X('予測貢献量:Q', title=c_title, axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_GRAY, titleFontWeight='normal', titleFontSize=11)),
                            yOffset=alt.YOffset('シナリオ:N', sort=['1. 現状', '2. 最適化']),
                            color=alt.Color('シナリオ:N', scale=alt.Scale(domain=['1. 現状', '2. 最適化'], range=[COLOR_BASELINE, COLOR_SECONDARY]), legend=alt.Legend(orient='bottom', title=None, labelFontSize=11, labelColor=COLOR_GRAY)),
                            tooltip=[alt.Tooltip('メディア:N'), alt.Tooltip('シナリオ:N'), alt.Tooltip('予測貢献量:Q', format=',.0f')]
                        ).properties(height=350)
                        
                        zero_rule_sales = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color=COLOR_GRAY, strokeDash=[4,4]).encode(x='x:Q')
                        st.altair_chart((bar_chart_sales + zero_rule_sales).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")

                    display_table = []
                    for m in res['media_cols']:
                        b_curr = comp_df[(comp_df['メディア']==m)&(comp_df['シナリオ']=='1. 現状')]['予算'].values[0]
                        b_opt = comp_df[(comp_df['メディア']==m)&(comp_df['シナリオ']=='2. 最適化')]['予算'].values[0]
                        s_curr = comp_df[(comp_df['メディア']==m)&(comp_df['シナリオ']=='1. 現状')]['予測貢献量'].values[0]
                        s_opt = comp_df[(comp_df['メディア']==m)&(comp_df['シナリオ']=='2. 最適化')]['予測貢献量'].values[0]
                        
                        display_table.append({
                            "メディア": m, 
                            "現状予算": b_curr,
                            "最適化予算": b_opt, 
                            "予算増減": b_opt - b_curr, 
                            "現状予測": s_curr,
                            "最適化後予測": s_opt, 
                            "予測増減": s_opt - s_curr
                        })
                    
                    if res.get('synergy_col_name') and res['synergy_col_name'] in res['coefficients']:
                        m1, m2 = res['synergy_m1'], res['synergy_m2']
                        idx1, idx2 = res['media_cols'].index(m1), res['media_cols'].index(m2)
                        b_opt_1, b_opt_2 = opt_spends[idx1], opt_spends[idx2]
                        
                        steady_opt_1 = (b_opt_1/n_periods) / (1 - res['decay_rates'][m1])
                        steady_opt_2 = (b_opt_2/n_periods) / (1 - res['decay_rates'][m2])
                        
                        syn_sat = apply_hill_saturation(steady_opt_1, res['hill_K_values'][m1], res['hill_S_values'][m1]) * apply_hill_saturation(steady_opt_2, res['hill_K_values'][m2], res['hill_S_values'][m2])
                        syn_opt_sales = syn_sat * res['coefficients'][res['synergy_col_name']] * n_periods
                        
                        synergy_sales_total_historical = np.sum(res['X_transformed'][res['synergy_col_name']].values * res['coefficients'][res['synergy_col_name']])
                        
                        display_table.append({
                            "メディア": f"相乗効果 ({m1} × {m2})",
                            "現状予算": 0, "最適化予算": 0, "予算増減": 0,
                            "現状予測": synergy_sales_total_historical * (n_periods / len(df_clean)),
                            "最適化後予測": syn_opt_sales,
                            "予測増減": syn_opt_sales - (synergy_sales_total_historical * (n_periods / len(df_clean)))
                        })

                    st.markdown("<hr style='margin: 3rem 0; border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.25rem;'>メディア別 アロケーション結果</h4>", unsafe_allow_html=True)
                    
                    df_display = pd.DataFrame(display_table)
                    format_dict = {
                        "現状予算": "¥ {:,.0f}", "最適化予算": "¥ {:,.0f}", "予算増減": "{:+,.0f}", 
                        "現状予測": "{:,.0f} 件" if is_cv else "¥ {:,.0f}", 
                        "最適化後予測": "{:,.0f} 件" if is_cv else "¥ {:,.0f}", 
                        "予測増減": "{:+,.0f}"
                    }
                    st.dataframe(
                        df_display.style.format(format_dict).map(lambda x: f"color: {COLOR_SECONDARY}; font-weight: 700;" if x > 0 else f"color: {COLOR_ROSE}; font-weight: 700;" if x < 0 else f"color: {COLOR_GRAY};", subset=['予算増減', '予測増減']),
                        use_container_width=True, hide_index=True
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button("このシナリオをCSVでダウンロード", data=convert_df_to_csv_no_index(df_display), file_name='optimix_scenario.csv', mime='text/csv')

                    # --- 追加: 限界効率曲線 (Saturation Curves) ---
                    st.markdown("<hr style='margin: 3.5rem 0; border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>限界効率曲線 (Saturation Curves)</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.7;'>各メディアの予算を増やしていった際、どこで効果が頭打ち（飽和）になるかを示す曲線です。<span style='color:{COLOR_GRAY}; font-weight:bold;'>● 現状</span> と <span style='color:{COLOR_PRIMARY}; font-weight:bold;'>● 最適化後</span> のポイントが曲線のどの位置にあるかを確認し、さらに追加投資の余地があるメディアを探ることができます。</p>", unsafe_allow_html=True)

                    curve_cols = st.columns(2, gap="large")
                    for idx, m in enumerate(res['media_cols']):
                        b_curr = df_clean[m].mean() * n_periods
                        b_opt = opt_spends[idx]
                        
                        # X軸の描画範囲をよしなに設定（現状の3倍、または最適化の1.5倍の大きい方）
                        max_budget = max(b_curr * 3, b_opt * 1.5, 100000)
                        x_vals = np.linspace(0, max_budget, 100)
                        
                        c = get_sim_coef(res, m)
                        decay = res['decay_rates'][m]
                        K = res['hill_K_values'][m]
                        S = res['hill_S_values'][m]
                        
                        y_vals = []
                        for x in x_vals:
                            steady = (x / n_periods) / (1 - decay)
                            sat = apply_hill_saturation(steady, K, S)
                            y_vals.append(sat * c * n_periods)
                            
                        df_curve = pd.DataFrame({'予算': x_vals, '予測貢献量': y_vals})
                        
                        steady_curr = (b_curr / n_periods) / (1 - decay)
                        s_curr = apply_hill_saturation(steady_curr, K, S) * c * n_periods
                        
                        steady_opt = (b_opt / n_periods) / (1 - decay)
                        s_opt = apply_hill_saturation(steady_opt, K, S) * c * n_periods
                        
                        df_points = pd.DataFrame({
                            '予算': [b_curr, b_opt],
                            '予測貢献量': [s_curr, s_opt],
                            'シナリオ': ['1. 現状', '2. 最適化']
                        })
                        
                        with curve_cols[idx % 2]:
                            st.markdown(f"<div style='font-weight: 700; color: {COLOR_TEXT_MAIN}; padding-bottom: 0.25rem; border-bottom: 2px solid {media_color_map[m]}; margin-bottom: 1rem; display: inline-block;'>{m}</div>", unsafe_allow_html=True)
                            
                            c_title = "予測CV数" if is_cv else "予測売上貢献"
                            line_curve = alt.Chart(df_curve).mark_line(color=COLOR_TEXT_MAIN, size=2, opacity=0.5).encode(
                                x=alt.X('予算:Q', title="予算 (¥)", axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, titleColor=COLOR_GRAY, grid=False)),
                                y=alt.Y('予測貢献量:Q', title=c_title, axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_GRAY))
                            )
                            
                            # 修正: 「1. 現状」のポイント色を COLOR_BASELINE(薄いグレー) から COLOR_GRAY(濃いグレー) に変更して視認性を向上
                            points_curve = alt.Chart(df_points).mark_circle(size=120, opacity=1.0).encode(
                                x='予算:Q',
                                y='予測貢献量:Q',
                                color=alt.Color('シナリオ:N', scale=alt.Scale(domain=['1. 現状', '2. 最適化'], range=[COLOR_GRAY, COLOR_PRIMARY]), legend=alt.Legend(orient='top', title=None)),
                                tooltip=[alt.Tooltip('シナリオ:N'), alt.Tooltip('予算:Q', format=',.0f'), alt.Tooltip('予測貢献量:Q', format=',.0f')]
                            )
                            
                            st.altair_chart((line_curve + points_curve).properties(height=260).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")


                    st.markdown("<hr style='margin: 3.5rem 0; border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;'><svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z'></path><polyline points='17 21 17 13 7 13 7 21'></polyline><polyline points='7 3 7 8 15 8'></polyline></svg> 構成案（シナリオ）の保存</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #64748B; font-size: 0.9rem; margin-bottom: 1rem;'>現在の予算アロケーション結果に名前を付けて保存し、画面下部で他のシナリオと比較できます。</p>", unsafe_allow_html=True)
                    
                    col_sn1, col_sn2 = st.columns([3, 1])
                    with col_sn1:
                        st.text_input("シナリオ名", value=f"シナリオ {len(st.session_state.saved_scenarios) + 1}", key="input_scen_name", label_visibility="collapsed")
                    with col_sn2:
                        if st.button("このシナリオを保存", type="primary", use_container_width=True):
                            save_current_scenario(sim_res['total_budget'], opt_sales, sim_res['media_budgets'], is_cv)

                    st.markdown("<hr style='margin: 3.5rem 0; border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>最適化後の時期別フライトプラン (Flight Plan)</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.7;'>対象の {n_periods} 期間において、ベースライン（過去のトレンドや季節性）の波に合わせて最適化された予算をどのように配分すべきかのシミュレーションです。獲得効率が上がりやすい時期に重点的に予算を投下する傾斜配分ロジックを採用しています。</p>", unsafe_allow_html=True)

                    future_len = n_periods
                    future_time_features = get_time_features(len(df_clean) + future_len, data_granularity, res['use_trend'], res['use_seasonality']).tail(future_len)
                    
                    future_base = np.full(future_len, res['intercept'])
                    if res['use_trend'] and 'trend' in res['coefficients']:
                        future_base += (future_time_features['trend'].values - res['time_features']['trend'].mean()) * res['coefficients']['trend']
                    if res['use_seasonality'] and 'seasonality_sin' in res['coefficients']:
                        sin_vals = future_time_features['seasonality_sin'].values - res['time_features']['seasonality_sin'].mean()
                        cos_vals = future_time_features['seasonality_cos'].values - res['time_features']['seasonality_cos'].mean()
                        future_base += sin_vals * res['coefficients']['seasonality_sin'] + cos_vals * res['coefficients']['seasonality_cos']
                    
                    future_base = np.maximum(future_base, 0.1)
                    weights = future_base / np.sum(future_base)

                    flight_plan_data = []
                    flight_dates = sim_res['flight_dates']
                    
                    for i in range(future_len):
                        row = {"時期": flight_dates[i]}
                        period_sales = future_base[i]
                        
                        syn_part_1, syn_part_2 = 0, 0
                        for idx, m in enumerate(res['media_cols']):
                            spend_t = opt_spends[idx] * weights[i]
                            row[m] = spend_t
                            c = get_sim_coef(res, m)
                            steady_state = spend_t / (1 - res['decay_rates'][m])
                            sat_val = apply_hill_saturation(steady_state, res['hill_K_values'][m], res['hill_S_values'][m])
                            period_sales += sat_val * c
                            
                            if res.get('synergy_col_name'):
                                if m == res['synergy_m1']: syn_part_1 = sat_val
                                if m == res['synergy_m2']: syn_part_2 = sat_val
                                
                        if res.get('synergy_col_name') and res['synergy_col_name'] in res['coefficients']:
                            period_sales += syn_part_1 * syn_part_2 * res['coefficients'][res['synergy_col_name']]
                        
                        row["予測成果"] = period_sales
                        flight_plan_data.append(row)
                    
                    flight_df = pd.DataFrame(flight_plan_data)
                    flight_melt = flight_df.melt(id_vars=['時期', '予測成果'], value_vars=res['media_cols'], var_name='メディア', value_name='予算配分')
                    
                    base_chart = alt.Chart(flight_melt).encode(x=alt.X('時期:N', sort=flight_dates, axis=alt.Axis(labelAngle=-45, labelColor=COLOR_GRAY, titleColor=COLOR_GRAY)))
                    
                    bar_flight = base_chart.mark_bar(opacity=0.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                        y=alt.Y('予算配分:Q', title="予算配分 (¥)", axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, gridColor="rgba(0,0,0,0.05)", gridDash=[2,2], titleColor=COLOR_GRAY)),
                        color=alt.Color('メディア:N', scale=alt.Scale(domain=list(media_color_map.keys()), range=list(media_color_map.values())), legend=alt.Legend(orient='bottom', title=None)),
                        tooltip=[alt.Tooltip('時期:N'), alt.Tooltip('メディア:N'), alt.Tooltip('予算配分:Q', format=',.0f')]
                    )
                    
                    line_flight = base_chart.mark_line(color=COLOR_TEXT_MAIN, size=3, point=True).encode(
                        y=alt.Y('予測成果:Q', title="予測成果", axis=alt.Axis(format=',.0f', labelColor=COLOR_GRAY, titleColor=COLOR_TEXT_MAIN)),
                        tooltip=[alt.Tooltip('時期:N'), alt.Tooltip('予測成果:Q', format=',.0f')]
                    )
                    
                    st.altair_chart(alt.layer(bar_flight, line_flight).resolve_scale(y='independent').properties(height=400).configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button("フライトプランをCSVでダウンロード", data=convert_df_to_csv_no_index(flight_df), file_name='optimix_flight_plan.csv', mime='text/csv')

                    # 追加: レポート出力ボタンをシミュレーション結果の最後に配置
                    st.markdown("<hr style='margin: 3.5rem 0; border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: {COLOR_TEXT_MAIN}; margin-bottom: 1.5rem;'>エグゼクティブ・サマリーレポート出力</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: {COLOR_GRAY}; font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.7;'>現在表示されているモデルの精度指標、チャネル別投資効率、および最新の予算アロケーション・シミュレーション結果をまとめたHTMLレポートをダウンロードします。</p>", unsafe_allow_html=True)
                    
                    col_rep1, col_rep2, col_rep3 = st.columns([1, 2, 1])
                    with col_rep2:
                        html_report = generate_html_report(res, df_clean, is_cv, margin_r, st.session_state.last_sim_result)
                        st.download_button(
                            label="サマリーレポートを出力 (HTML)", 
                            data=html_report, 
                            file_name='mmm_executive_report.html', 
                            mime='text/html', 
                            use_container_width=True, 
                            type="primary"
                        )

            if len(st.session_state.saved_scenarios) > 0:
                st.markdown("<hr style='margin: 4rem 0; border-color: rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
                st.markdown(f"<h2 class='section-title' style='color: {COLOR_TEXT_MAIN}; display: flex; align-items: center; gap: 12px;'><svg width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='{COLOR_PRIMARY}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><line x1='18' y1='20' x2='18' y2='10'></line><line x1='12' y1='20' x2='12' y2='4'></line><line x1='6' y1='20' x2='6' y2='14'></line></svg>保存されたシナリオの比較</h2>", unsafe_allow_html=True)
                
                saved_list = st.session_state.saved_scenarios
                num_scenarios = len(saved_list)
                
                cols = st.columns(num_scenarios)
                
                for i, s in enumerate(saved_list):
                    with cols[i]:
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.7); border-radius: 16px; padding: 20px 16px; border: 1px solid rgba(0,0,0,0.08); text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.03); height: 100%; display: flex; flex-direction: column;'>
                            <h4 style='color: {COLOR_TEXT_MAIN}; margin-top: 0; margin-bottom: 4px; font-size: 1.1rem;'>{s['name']}</h4>
                            <div style='color: {COLOR_GRAY}; font-size: 0.9rem; margin-bottom: 12px; font-weight: 600;'>総予算: ¥{int(s['total_budget']):,}</div>
                        """, unsafe_allow_html=True)
                        
                        df_pie = pd.DataFrame([{"メディア": m, "予算": val} for m, val in s['media_budgets'].items() if val > 0])
                        
                        base_pie = alt.Chart(df_pie).encode(
                            theta=alt.Theta(field="予算", type="quantitative", stack=True),
                            color=alt.Color('メディア:N', scale=alt.Scale(domain=list(media_color_map.keys()), range=list(media_color_map.values())), legend=alt.Legend(orient='bottom', title=None, labelFontSize=10)),
                            tooltip=[alt.Tooltip('メディア:N'), alt.Tooltip('予算:Q', format=',.0f')]
                        )
                        pie = base_pie.mark_arc(innerRadius=65, cornerRadius=4)
                        
                        val_text = f"{int(s['predicted_result']):,}"
                        text_val = alt.Chart(pd.DataFrame({'text': [val_text]})).mark_text(
                            align='center', baseline='middle', fontSize=24, fontWeight='bold', color=COLOR_TEXT_MAIN, dy=4
                        ).encode(text='text:N')
                        
                        t_label = "予測CV数" if is_cv else "予測売上"
                        text_label = alt.Chart(pd.DataFrame({'text': [t_label]})).mark_text(
                            align='center', baseline='middle', fontSize=11, color=COLOR_GRAY, dy=-16
                        ).encode(text='text:N')
                        
                        donut_chart = (pie + text_val + text_label).properties(height=280)
                        st.altair_chart(donut_chart.configure_view(strokeWidth=0).configure(background='transparent'), use_container_width=True, theme="streamlit")
                        
                        cpa_text = f"見込CPA: ¥{int(s['total_budget'] / s['predicted_result']):,}" if is_cv and s['predicted_result'] > 0 else ""
                        roi_text = "" if is_cv else f"ROI: {(s['predicted_result'] * margin_r) / s['total_budget'] * 100:.1f}%" if s['total_budget'] > 0 else ""
                        sub_info = cpa_text if is_cv else roi_text
                        
                        if sub_info:
                            st.markdown(f"<div style='color: {COLOR_SECONDARY}; font-weight: bold; font-size: 0.95rem; text-align: center; margin-top: auto; margin-bottom: 16px; background: rgba(16, 185, 129, 0.1); padding: 4px 12px; border-radius: 8px; display: inline-block; align-self: center;'>{sub_info}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='margin-top: auto; margin-bottom: 16px;'></div>", unsafe_allow_html=True)

                        if st.button("削除", key=f"del_scen_{s['id']}", use_container_width=True):
                            st.session_state.saved_scenarios = [x for x in st.session_state.saved_scenarios if x['id'] != s['id']]
                            st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)

                if not run_opt:
                    st.info("左側のパネルで制約条件を変更して「最適化を実行」すると、別のパターンのシナリオを作成して比較に追加できます。")

        else:
            st.info("「モデルチューニング」タブで分析を実行してください。")

else:
    st.markdown(f'<div style="text-align: center; padding: 6rem 2rem; background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border-radius: 2rem; border: 1px solid rgba(255,255,255,0.8); margin-top: 3rem; box-shadow: 0 16px 40px -5px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,1); transition: all 0.3s ease;"><div style="background: rgba(37,99,235,0.1); width: 88px; height: 88px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem auto; color: {COLOR_PRIMARY}; box-shadow: inset 0 2px 4px rgba(255,255,255,0.5);"><svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg></div><h2 style="color: {COLOR_TEXT_MAIN}; margin-bottom: 0.75rem; font-weight: 800; font-size: 2rem; letter-spacing: -0.04em;">OptiMix MMM へようこそ</h2><p style="color: {COLOR_GRAY}; font-size: 1.15rem; max-width: 600px; margin: 0 auto; line-height: 1.7;">左側のサイドバーから、分析したいCSVファイルをドラッグ＆ドロップしてマーケティング分析を開始してください。</p></div>', unsafe_allow_html=True)