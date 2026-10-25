import io
import os
import re
import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient

# =========================================================
# LOAD ENVIRONMENT VARIABLES & SETUP CLIENTS
# =========================================================

load_dotenv()

load_dotenv()

# Safely check st.secrets for cloud deployment, fallback to .env for local testing
api_key = None
mongo_uri = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

try:
    if "MONGO_URI" in st.secrets:
        mongo_uri = st.secrets["MONGO_URI"]
except Exception:
    pass

if not mongo_uri:
    mongo_uri = os.getenv("MONGO_URI")

client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")

# Initialize MongoDB Client & Collections
db_client = None
users_col = None
datasets_col = None

if mongo_uri:
    try:
        db_client = MongoClient(mongo_uri)
        db = db_client["datasense_cloud_db"]
        users_col = db["users"]
        datasets_col = db["datasets"]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DataSense AI | Dataset Explorer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM THEME & CSS (MOVING PURPLE FLUID BACKGROUND)
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

@keyframes fluidMotion {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(135deg, #10061c, #240e3b, #180929, #351454, #120520, #2a0f45);
    background-size: 400% 400%;
    animation: fluidMotion 18s ease infinite;
    color: #f1f5f9;
}

.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2.5rem;
    max-width: 1440px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f8fafc !important;
}

[data-testid="stSidebar"] {
    background: #111113 !important;
    border-right: 1px solid #27272a;
}

[data-testid="stSidebar"] * {
    color: #f4f4f5 !important;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 10px;
    background: #1f1f23;
    color: #f4f4f5 !important;
    border: 1px solid #3f3f46;
    font-weight: 600;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #27272a !important;
    color: #ffffff !important;
    border: 1px solid #5f6368;
}

@keyframes agentPulse {
    0% { transform: scale(0.98); opacity: 0.7; box-shadow: 0 0 10px rgba(139, 92, 246, 0.1); }
    50% { transform: scale(1.02); opacity: 1; box-shadow: 0 0 25px rgba(236, 72, 153, 0.25); }
    100% { transform: scale(0.98); opacity: 0.7; box-shadow: 0 0 10px rgba(139, 92, 246, 0.1); }
}

.agent-side-panel {
    background: linear-gradient(145deg, rgba(36, 14, 59, 0.85), rgba(18, 5, 32, 0.9));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    animation: agentPulse 6s ease-in-out infinite;
}

.agent-badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    color: #c084fc;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
    letter-spacing: 1px;
    margin-bottom: 10px;
    text-transform: uppercase;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.agent-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
}

.agent-desc {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.5;
    margin-bottom: 12px;
}

.pulse-node-container {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 12px;
}

.p-node {
    width: 8px;
    height: 8px;
    background-color: #ec4899;
    border-radius: 50%;
    box-shadow: 0 0 8px #ec4899;
    animation: agentPulse 2s infinite;
}

.ds-hero {
    background: linear-gradient(135deg, rgba(36, 14, 59, 0.8), rgba(53, 20, 84, 0.6));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    padding: 26px 28px;
    margin-bottom: 18px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}

.ds-eyebrow {
    color: #c084fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.ds-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -1px;
    color: #ffffff;
    margin: 5px 0 8px;
}

.ds-sub {
    color: #94a3b8;
    font-size: 15px;
    max-width: 850px;
}

.ds-pill {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    color: #d8b4fe;
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 5px;
    margin-top: 5px;
}

.ds-card {
    background: rgba(36, 14, 59, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 18px;
    height: 100%;
    color: #f4f4f5;
}

.ds-card-title {
    color: #a1a1aa;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.ds-note {
    background: rgba(99, 102, 241, 0.15);
    border-left: 4px solid #818cf8;
    border-radius: 8px;
    padding: 12px 14px;
    color: #e0e7ff;
}

.ds-good {
    background: rgba(16, 185, 129, 0.15);
    border-left: 4px solid #34d399;
    border-radius: 8px;
    padding: 12px 14px;
    color: #d1fae5;
}

.ds-warn {
    background: rgba(245, 158, 11, 0.15);
    border-left: 4px solid #fbbf24;
    border-radius: 8px;
    padding: 12px 14px;
    color: #fef3c7;
}

div[data-testid="stFileUploader"] {
    background: rgba(36, 14, 59, 0.7);
    border: 1px dashed rgba(255, 255, 255, 0.2);
    border-radius: 14px;
    padding: 12px;
}
div[data-testid="stFileUploader"] section {
    background: #180929 !important;
    border-radius: 10px;
}
div[data-testid="stFileUploader"] * {
    color: #f4f4f5 !important;
}

.ds-workflow-flow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-top: 18px;
    margin-bottom: 12px;
}

.ds-wf-box {
    background: rgba(36, 14, 59, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 12px 6px;
    text-align: center;
    flex: 1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.ds-wf-box b {
    font-size: 11px;
    color: #e2e8f0;
    display: block;
    margin-top: 4px;
}

.ds-wf-arrow {
    color: #c084fc;
    font-weight: bold;
    font-size: 18px;
    text-align: center;
    flex: 0 0 auto;
    padding: 0 2px;
}

input, textarea, select {
    color: #ffffff !important;
}

@keyframes spinClockwise {
    0% { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}

.side-radar-box {
    position: relative;
    width: 100%;
    height: 310px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: rgba(24, 9, 41, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    margin-top: 15px;
}

.side-eco-ring {
    position: absolute;
    top: 50%;
    left: 50%;
    border: 1px dashed rgba(139, 92, 246, 0.25);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: spinClockwise 35s linear infinite;
}

.side-eco-ring:nth-child(1) { width: 90px; height: 90px; border-color: rgba(56, 189, 248, 0.3); animation-duration: 18s; }
.side-eco-ring:nth-child(2) { width: 155px; height: 155px; border-color: rgba(236, 72, 153, 0.25); animation-duration: 28s; animation-direction: reverse; }
.side-eco-ring:nth-child(3) { width: 220px; height: 220px; border-color: rgba(74, 222, 128, 0.2); animation-duration: 40s; }
.side-eco-ring:nth-child(4) { width: 285px; height: 285px; border-color: rgba(250, 204, 21, 0.2); animation-duration: 55s; }

.side-eco-bubble {
    position: absolute;
    width: 36px;
    height: 36px;
    background: rgba(24, 9, 41, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CLOUD DATABASE HELPER FUNCTIONS (MONGODB ATLAS)
# =========================================================

def get_user_from_db(email):
    if users_col is not None:
        try:
            return users_col.find_one({"email": email})
        except Exception:
            return None
    return None

def save_user_to_db(email, password):
    if users_col is not None:
        try:
            users_col.insert_one({"email": email, "password": password, "history": []})
            return True
        except Exception:
            return False
    return False

def get_user_history_from_db(email):
    user = get_user_from_db(email)
    if user and "history" in user:
        return user["history"] # returns list of filename strings
    return []

def add_history_to_db(email, filename):
    if users_col is not None:
        try:
            users_col.update_one(
                {"email": email},
                {"$addToSet": {"history": filename}}
            )
        except Exception as e:
            print(f"Error updating history: {e}")

def save_dataset_bytes_to_db(email, filename, file_bytes):
    if datasets_col is not None:
        try:
            datasets_col.update_one(
                {"email": email, "filename": filename},
                {"$set": {"file_bytes": file_bytes, "updated_at": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            print(f"Error saving dataset bytes: {e}")

def load_dataset_bytes_from_db(email, filename):
    if datasets_col is not None:
        try:
            record = datasets_col.find_one({"email": email, "filename": filename})
            if record and "file_bytes" in record:
                return record["file_bytes"]
        except Exception as e:
            print(f"Error loading dataset bytes: {e}")
    return None


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "ds_signed_in" not in st.session_state:
    st.session_state["ds_signed_in"] = False
if "ds_user" not in st.session_state:
    st.session_state["ds_user"] = ""
if "ds_selected_history_file" not in st.session_state:
    st.session_state["ds_selected_history_file"] = None
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "signup"


# =========================================================
# AGENT UTILITIES & GEMINI AI QUESTION HANDLER
# =========================================================

def read_dataset_from_source(file_obj_or_bytes, filename):
    name = filename.lower()
    if isinstance(file_obj_or_bytes, bytes):
        buffer = io.BytesIO(file_obj_or_bytes)
    else:
        buffer = file_obj_or_bytes

    if name.endswith(".csv"):
        try:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=None, engine='python')
        except Exception:
            try:
                buffer.seek(0)
                return pd.read_csv(buffer)
            except UnicodeDecodeError:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        buffer.seek(0)
        return pd.read_excel(buffer)
    
    raise ValueError("Please upload a valid CSV or Excel file.")

def safe_name(name):
    return str(name).replace("`", "").replace("'", "").replace('"', "")

def profile_data(df):
    missing = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(include=["object", "category", "bool"])
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_cells = int(missing.sum())
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": missing,
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2),
        "duplicates": duplicate_count,
        "numeric": list(numeric.columns),
        "categorical": list(categorical.columns),
        "numeric_df": numeric,
        "categorical_df": categorical,
    }

def make_insights(df, profile):
    insights = []
    if df.empty:
        return ["This dataset has no rows to analyze yet."]
    if profile["missing_cells"]:
        top = profile["missing"].sort_values(ascending=False)
        top = top[top > 0]
        if len(top):
            col = top.index[0]
            insights.append(f"**Data quality:** `{safe_name(col)}` has the most missing values ({int(top.iloc[0])}).")
    else:
        insights.append("**Data quality:** No missing values found.")
    if profile["duplicates"]:
        insights.append(f"**Duplicate check:** {profile['duplicates']} duplicate row(s) detected.")
    else:
        insights.append("**Duplicate check:** No fully duplicated rows detected.")
    numeric = profile["numeric_df"]
    if not numeric.empty:
        for col in list(numeric.columns)[:2]:
            series = numeric[col].dropna()
            if len(series):
                insights.append(f"**{safe_name(col)}:** average is **{series.mean():,.2f}** (min: {series.min():,.2f}, max: {series.max():,.2f}).")
    return insights[:6]

def answer_question(df, question):
    q = question.lower().strip()
    if not q:
        return "Ask a question about the uploaded dataset."
    
    if not client:
        return "⚠️ Gemini API client is not configured. Please check that your secrets contain a valid `GEMINI_API_KEY`."

    df_summary = f"""
    Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
    Columns: {list(df.columns)}
    Data Types: {df.dtypes.to_dict()}
    
    Full Dataset Preview:
    {df.to_string()}
    
    Statistical Summary:
    {df.describe().to_string()}
    """
    
    prompt = f"""
    You are DataSense AI, an intelligent data science agent. 
    Here is the exact dataset context:
    {df_summary}
    
    User's Question: "{question}"
    
    Instructions:
    - Analyze the dataset strictly based on the provided context.
    - Give clear, accurate answers (such as checking values, calculating statistics, looking up specific student records, or identifying outliers using standard IQR/statistical methods).
    - Keep the formatting clean using Markdown.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error communicating with Gemini API: {e}"


# =========================================================
# LOGO DEFINITION
# =========================================================

logo_svg = """
<svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="12" height="12" rx="3" fill="#3b82f6"/>
  <rect x="25" y="15" width="10" height="10" rx="2" fill="#60a5fa"/>
  <rect x="15" y="27" width="11" height="11" rx="2" fill="#2563eb"/>
  <rect x="30" y="29" width="12" height="12" rx="3" fill="#93c5fd"/>
  <rect x="8" y="42" width="10" height="10" rx="2" fill="#1d4ed8"/>
  <path d="M22 38H45C65 38 78 51 78 71C78 91 65 104 45 104H22V38Z" fill="url(#paint0_linear)"/>
  <path d="M38 53V89H46C56 89 62 83 62 71C62 59 56 53 46 53H38Z" fill="#ffffff"/>
  <rect x="42" y="68" width="6" height="14" rx="2" fill="#3b82f6"/>
  <rect x="51" y="60" width="6" height="22" rx="2" fill="#2563eb"/>
  <defs>
    <linearGradient id="paint0_linear" x1="22" y1="38" x2="78" y2="104" gradientUnits="userSpaceOnUse">
      <stop stop-color="#3b82f6"/>
      <stop offset="1" stop-color="#1d4ed8"/>
    </linearGradient>
  </defs>
</svg>
"""


# =========================================================
# GEMINI-STYLE DARK SIDEBAR (AFTER SIGN-IN)
# =========================================================

if st.session_state["ds_signed_in"]:
    with st.sidebar:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding:5px 0 8px 0;">
            <div style="background:#ffffff; width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; padding:3px; border:2px solid #3b82f6;">
                <svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="10" y="10" width="12" height="12" rx="3" fill="#3b82f6"/>
                  <rect x="25" y="15" width="10" height="10" rx="2" fill="#60a5fa"/>
                  <rect x="15" y="27" width="11" height="11" rx="2" fill="#2563eb"/>
                  <rect x="30" y="29" width="12" height="12" rx="3" fill="#93c5fd"/>
                  <rect x="8" y="42" width="10" height="10" rx="2" fill="#1d4ed8"/>
                  <path d="M22 38H45C65 38 78 51 78 71C78 91 65 104 45 104H22V38Z" fill="url(#paint0_linear_sb)"/>
                  <path d="M38 53V89H46C56 89 62 83 62 71C62 59 56 53 46 53H38Z" fill="#ffffff"/>
                  <rect x="42" y="68" width="6" height="14" rx="2" fill="#3b82f6"/>
                  <rect x="51" y="60" width="6" height="22" rx="2" fill="#2563eb"/>
                  <defs>
                    <linearGradient id="paint0_linear_sb" x1="22" y1="38" x2="78" y2="104" gradientUnits="userSpaceOnUse">
                      <stop stop-color="#3b82f6"/>
                      <stop offset="1" stop-color="#1d4ed8"/>
                    </linearGradient>
                  </defs>
                </svg>
            </div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; color:#e3e3e3;">DataSense AI</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if st.button("➕ New Upload", key="new_upload_btn"):
            if "main_file_uploader" in st.session_state:
                del st.session_state["main_file_uploader"]
            st.session_state["ds_selected_history_file"] = None
            st.rerun()

        st.markdown("---")
        st.markdown("### 🕒 Explored Datasets History")
        
        current_user = st.session_state["ds_user"]
        user_history = get_user_history_from_db(current_user)
        
        if user_history:
            for item in user_history[-5:]:
                if st.button(f"📄 {item}", key=f"hist_{item}"):
                    st.session_state["ds_selected_history_file"] = item
                    st.rerun()
        else:
            st.markdown("<div style='font-size:12px; color:#8e918f;'>No datasets explored yet.</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        user_initial = current_user[:1].upper() if current_user else "U"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:8px 0; margin-top:10px;">
            <div style="background:#2563eb; color:white; width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:15px;">
                {user_initial}
            </div>
            <div style="overflow:hidden;">
                <div style="font-size:13px; font-weight:700; color:#e3e3e3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{current_user}</div>
                <div style="font-size:11px; color:#a8c7fa; font-weight:600;">DataSense Cloud</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔒 Sign Out", key="sidebar_sign_out"):
            st.session_state["ds_signed_in"] = False
            st.session_state["ds_user"] = ""
            st.session_state["ds_selected_history_file"] = None
            st.session_state["auth_mode"] = "signup"
            st.rerun()


# =========================================================
# AUTH GATE WITH CLOUD SYNC
# =========================================================

if not st.session_state["ds_signed_in"]:
    left_col, center_col, right_col = st.columns([1, 1.8, 1])
    
    with left_col:
        st.markdown("""
        <div class="agent-side-panel">
            <div class="agent-badge">🔍 CLOUD DIAGNOSTICS</div>
            <div class="agent-title">Data Quality & Health</div>
            <div class="agent-desc">Inspect missing values, completeness ratios, and detect duplicate rows instantly with secure cloud persistence.</div>
            <div class="pulse-node-container">
                <div class="p-node"></div>
                <div class="p-node" style="animation-delay: 0.4s;"></div>
                <div class="p-node" style="animation-delay: 0.8s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="side-radar-box">
            <div class="side-eco-ring"><div class="side-eco-bubble" style="top: -15px; left: 25px;">📗</div></div>
            <div class="side-eco-ring"><div class="side-eco-bubble" style="bottom: -15px; right: 30px;">🐍</div></div>
            <div class="side-eco-ring"><div class="side-eco-bubble" style="top: 60px; right: -15px;">🍃</div></div>
            <div class="side-eco-ring"><div class="side-eco-bubble" style="bottom: 60px; left: -15px;">🗄️</div></div>
        </div>
        """, unsafe_allow_html=True)
        
    with right_col:
        st.markdown("""
        <div class="agent-side-panel">
            <div class="agent-badge">💬 CLOUD CHAT AGENT</div>
            <div class="agent-title">Conversational AI</div>
            <div class="agent-desc">Ask complex questions about your numerical features, trends, or outliers and get precise insights from Gemini.</div>
            <div class="pulse-node-container">
                <div class="p-node" style="animation-delay: 0.2s;"></div>
                <div class="p-node" style="animation-delay: 0.6s;"></div>
                <div class="p-node" style="animation-delay: 1.0s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="side-radar-box">
            <div class="side-eco-ring"><div class="side-eco-bubble" style="bottom: -15px; left: 30px;">📈</div></div>
            <div class="side-eco-ring"><div class="side-eco-bubble" style="top: -15px; right: 35px;">🧱</div></div>
            <div class="side-eco-ring"><div class="side-eco-bubble" style="top: 60px; left: -15px;">⚡</div></div>
            <div class="side-eco-ring"><div class="side-eco-bubble" style="bottom: 60px; right: -15px;">📊</div></div>
        </div>
        """, unsafe_allow_html=True)

    with center_col:
        st.markdown(f"""
        <div class="ds-hero" style="text-align: center; padding: 30px; margin-bottom: 20px;">
            <div style="background:#ffffff; width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin: 0 auto 12px auto; box-shadow: 0 6px 20px rgba(37,99,235,0.4); padding:3px; border:2px solid rgba(59,130,246,0.8);">{logo_svg}</div>
            <div class="ds-eyebrow">YOUR PERSONAL DATA INTELLIGENCE AGENT</div>
            <div class="ds-title" style="font-size: 28px; margin-top:6px;">Welcome to DataSense AI</div>
            <div class="ds-sub" style="margin: 0 auto; text-align: center; font-size: 14px; color:#94a3b8;">
                Drop your raw datasets and watch your cloud-backed intelligent agent transform them into deep insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        scol1, scol2 = st.columns(2)
        with scol1:
            if st.button("Create Account (Sign Up)", use_container_width=True, type="primary" if st.session_state["auth_mode"] == "signup" else "secondary"):
                st.session_state["auth_mode"] = "signup"
                st.rerun()
        with scol2:
            if st.button("Existing User (Sign In)", use_container_width=True, type="primary" if st.session_state["auth_mode"] == "signin" else "secondary"):
                st.session_state["auth_mode"] = "signin"
                st.rerun()

        is_signup = (st.session_state["auth_mode"] == "signup")

        with st.form("clean_auth_form"):
            email_val = st.text_input("Email", placeholder="name@example.com")
            pass_val = st.text_input("Password", type="password", placeholder="Enter your password")
            
            form_btn_label = "Create Account (Sign Up)" if is_signup else "Existing User (Sign In)"
            form_submitted = st.form_submit_button(form_btn_label, type="primary", use_container_width=True)
            
            if form_submitted:
                clean_e = email_val.strip().lower()
                if not clean_e or not pass_val:
                    st.error("Please enter both email and password.")
                elif is_signup:
                    existing = get_user_from_db(clean_e)
                    if existing:
                        st.warning("Account already exists! Please switch to 'Existing User (Sign In)'.")
                    else:
                        success = save_user_to_db(clean_e, pass_val)
                        if success:
                            st.session_state["ds_signed_in"] = True
                            st.session_state["ds_user"] = clean_e
                            st.success("Account created successfully in MongoDB! Logging you in...")
                            st.rerun()
                        else:
                            st.error("Database error while creating account. Please check connection.")
                else:
                    user_record = get_user_from_db(clean_e)
                    if user_record:
                        if user_record.get("password") == pass_val:
                            st.session_state["ds_signed_in"] = True
                            st.session_state["ds_user"] = clean_e
                            st.rerun()
                        else:
                            st.error("Incorrect password. Please try again.")
                    else:
                        st.error("Account not found! Please click 'Create Account (Sign Up)' above first.")

        st.markdown("""
            <div style="text-align:center; margin-top: 14px; font-size: 12px; color: #a1a1aa;">
                Protected by DataSense MongoDB Cloud Sync.
            </div>
        """, unsafe_allow_html=True)
    st.stop()


# =========================================================
# HEADER (AFTER SIGN-IN)
# =========================================================

st.markdown(f"""
<div class="ds-hero">
  <div class="ds-eyebrow">YOUR PERSONAL CLOUD DATA AGENT</div>
  <div class="ds-title" style="display:flex; align-items:center; gap:12px;">
    <div style="background:#ffffff; width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; padding:3px; border:2px solid rgba(59,130,246,0.8);">{logo_svg}</div>
    <span>DataSense AI</span>
  </div>
  <div class="ds-sub">
    Hey there! Drop your dataset and let’s uncover the story inside. Your files and history are securely synced to the cloud.
  </div>
  <div style="margin-top:12px">
    <span class="ds-pill">⌁ Smart profiling</span>
    <span class="ds-pill">▥ Auto charts</span>
    <span class="ds-pill">✧ Cloud history</span>
    <span class="ds-pill">⌕ Ask your data</span>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DATASET SELECTION (UPLOAD OR CLOUD HISTORY)
# =========================================================

active_user = st.session_state["ds_user"]
selected_history = st.session_state.get("ds_selected_history_file")

active_source_bytes = None
active_source_name = None

if selected_history:
    st.info(f"📁 Loaded dataset from cloud history: **{selected_history}**")
    if st.button("← Upload a different file instead"):
        st.session_state["ds_selected_history_file"] = None
        st.rerun()
    
    file_bytes = load_dataset_bytes_from_db(active_user, selected_history)
    if file_bytes:
        active_source_bytes = file_bytes
        active_source_name = selected_history
    else:
        st.error("Could not retrieve dataset bytes from MongoDB. Please re-upload.")
        st.session_state["ds_selected_history_file"] = None
        st.rerun()
else:
    uploaded_files = st.file_uploader("Choose CSV or Excel dataset", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="main_file_uploader")
    
    if not uploaded_files:
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("""
            <div class="ds-card">
              <div class="ds-card-title">START HERE</div>
              <h3 style="margin-top:8px">Give your data a cloud spotlight.</h3>
              <p style="color:#94a3b8">Upload any dataset. The agent automatically saves it to your cloud history so you can reload it anytime from any device.</p>
              <div class="ds-note">🧩 Bring any datasets you’re curious about — from business tables to research projects.</div>
            </div>
            """, unsafe_allow_html=True)
        with right:
            st.markdown("""
            <div class="ds-card">
              <div class="ds-card-title">CLOUD WORKFLOW</div>
              <h3 style="margin-top:8px">A guided path from data to discovery</h3>
              
              <div class="ds-workflow-flow">
                <div class="ds-wf-box">📥<b>1. Upload</b></div>
                <div class="ds-wf-arrow">→</div>
                <div class="ds-wf-box">☁️<b>2. Sync</b></div>
                <div class="ds-wf-arrow">→</div>
                <div class="ds-wf-box">⚙️<b>3. Quality</b></div>
                <div class="ds-wf-arrow">→</div>
                <div class="ds-wf-box">📊<b>4. Explore</b></div>
                <div class="ds-wf-arrow">→</div>
                <div class="ds-wf-box">💡<b>5. Insights</b></div>
              </div>

              <div class="ds-good" style="margin-top: 12px;">💡 Your data. Securely stored in the cloud.</div>
            </div>
            """, unsafe_allow_html=True)
        st.stop()

    if len(uploaded_files) > 1:
        selected_file_name = st.selectbox("Select active dataset to analyze:", [f.name for f in uploaded_files])
        active_file = next(f for f in uploaded_files if f.name == selected_file_name)
    else:
        active_file = uploaded_files[0]

    active_source_name = active_file.name
    active_source_bytes = active_file.getvalue()

    # Save automatically to MongoDB Atlas cloud history
    add_history_to_db(active_user, active_source_name)
    save_dataset_bytes_to_db(active_user, active_source_name, active_source_bytes)

try:
    with st.spinner(f"✦ Agent is scanning {active_source_name}..."):
        df = read_dataset_from_source(active_source_bytes, active_source_name)
except Exception as exc:
    st.error(f"Couldn't read this file: {exc}")
    st.stop()

if df.empty:
    st.warning("This file contains no data rows.")
    st.stop()

profile = profile_data(df)
st.success(f"Scan complete for **{active_source_name}** — Explored {profile['rows']:,} rows across {profile['cols']:,} columns.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("🧾 Total rows", f"{profile['rows']:,}")
m2.metric("▦ Total columns", f"{profile['cols']:,}")
m3.metric("🩺 Missing cells", f"{profile['missing_cells']:,}", f"{profile['missing_pct']}% of all cells")
m4.metric("⧉ Duplicate rows", f"{profile['duplicates']:,}")

# =========================================================
# TABS SETUP
# =========================================================

tab_overview, tab_validation, tab_explore, tab_agent, tab_export = st.tabs([
    "Overview", "⚙️ Data Quality & Validation", "📊 Explore", "💬 Ask the Agent", "📥 Export"
])

with tab_overview:
    st.subheader("Your dataset at a glance")
    a, b = st.columns([1.25, 1])
    with a:
        st.markdown("#### 👀 Data preview")
        st.dataframe(df.head(12), width="stretch", height=330)
    with b:
        st.markdown("#### 🧠 Agent's first take")
        for insight in make_insights(df, profile)[:5]:
            st.markdown(f"<div class='ds-note' style='margin-bottom:9px'>✦ {insight}</div>", unsafe_allow_html=True)
    st.markdown("#### 🧬 Column profile")
    col_profile = pd.DataFrame({
        "Column": [str(c) for c in df.columns],
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Non-empty values": [int(df[c].notna().sum()) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(col_profile, width="stretch", hide_index=True)

with tab_validation:
    st.subheader("⚙️ Data Quality & Validation Suite")
    st.caption("Advanced data diagnostics: structural integrity, statistical anomaly checks, and schema profiling.")
    
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("#### Missing values by column")
        missing_df = profile["missing"][profile["missing"] > 0].sort_values(ascending=False)
        if missing_df.empty:
            st.markdown("<div class='ds-good'>✓ No missing values found. Nice and tidy!</div>", unsafe_allow_html=True)
        else:
            completeness_report = pd.DataFrame({
                "Missing values": missing_df,
                "Missing %": ((missing_df / len(df)) * 100).round(2)
            })
            st.dataframe(completeness_report, width="stretch")
            
    with v2:
        st.markdown("#### Duplicate records")
        if profile["duplicates"] == 0:
            st.markdown("<div class='ds-good'>✓ No fully duplicated rows found.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ds-warn'>⚠ Warning: Found <b>{profile['duplicates']}</b> fully duplicated rows.</div>", unsafe_allow_html=True)
            if st.button("Inspect Duplicate Rows"):
                st.dataframe(df[df.duplicated(keep=False)], width="stretch")

    st.markdown("---")
    st.markdown("#### 📊 Statistical Anomaly & Outlier Detection Matrix")
    st.caption("Evaluating numerical feature distributions using Interquartile Range (IQR) and Z-Score algorithms.")
    
    numeric_cols = profile["numeric"]
    if numeric_cols:
        outlier_rows = []
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) >= 4:
                q1, q3 = s.quantile(.25), s.quantile(.75)
                iqr = q3 - q1
                iqr_count = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()) if iqr != 0 else 0
                
                mean_val, std_val = s.mean(), s.std()
                z_count = int((np.abs((s - mean_val) / std_val) > 3).sum()) if std_val > 0 else 0
                
                outlier_rows.append({
                    "Feature Column": str(col),
                    "IQR Outliers (1.5x)": iqr_count,
                    "Z-Score Anomalies (|Z|>3)": z_count,
                    "Feature Mean": round(mean_val, 2),
                    "Feature StdDev": round(std_val, 2)
                })
        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows), width="stretch", hide_index=True)
    else:
        st.info("No numerical features available for outlier matrix evaluation.")

with tab_explore:
    st.subheader("📊 Explore the patterns")
    numeric_cols = profile["numeric"]
    categorical_cols = profile["categorical"]
    if numeric_cols:
        st.markdown("#### Numerical distribution")
        chosen_num = st.selectbox("Choose a numeric column", numeric_cols, key="dist_col")
        vals = df[chosen_num].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8, 3.6))
            ax.hist(vals, bins=min(25, max(5, int(np.sqrt(len(vals))))), edgecolor="white", color="#5146e5")
            ax.set_title(f"Distribution of {chosen_num}")
            ax.set_xlabel(str(chosen_num)); ax.set_ylabel("Frequency")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        if len(numeric_cols) >= 2:
            st.markdown("#### Numeric relationship")
            xcol, ycol = st.columns(2)
            with xcol:
                x_var = st.selectbox("X-axis", numeric_cols, index=0, key="xvar")
            with ycol:
                y_var = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols)>1 else 0, key="yvar")
            if x_var != y_var:
                plot_df = df[[x_var, y_var]].dropna()
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.scatter(plot_df[x_var], plot_df[y_var], alpha=.7, color="#5146e5")
                ax.set_xlabel(str(x_var)); ax.set_ylabel(str(y_var)); ax.set_title(f"{x_var} vs {y_var}")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
    if categorical_cols:
        st.markdown("#### Category comparison")
        cat_col = st.selectbox("Choose a category column", categorical_cols, key="cat_col")
        top_n = st.slider("Number of categories to show", min_value=3, max_value=15, value=8)
        counts = df[cat_col].astype("string").fillna("(missing)").value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        counts.sort_values().plot(kind="barh", ax=ax, color="#5146e5")
        ax.set_title(f"Most frequent values in {cat_col}")
        ax.set_xlabel("Count"); ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

with tab_agent:
    st.subheader("💬 Ask your dataset")
    st.markdown("<div class='ds-note'>Hey! I'm DataSense, powered by Gemini AI. Ask me anything about your dataset — rows, patterns, student scores, or outliers!</div>", unsafe_allow_html=True)
    
    question = st.text_input("Your question", key="ds_question", placeholder="e.g. What are the main trends in this dataset?")
    
    st.markdown("""
    <div style="margin-top: 8px; margin-bottom: 12px; font-size: 13px; color: #94a3b8;">
        <b>💡 Popular example questions you can ask:</b>
        <ul style="margin: 4px 0 0 18px; color: #cbd5e1;">
            <li>What are the top 3 insights or trends in this dataset?</li>
            <li>Are there any missing values or data anomalies I should fix?</li>
            <li>Give me a statistical summary of the main numerical columns.</li>
            <li>Are there any outliers present in the dataset?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✦ Analyze with Gemini", type="primary"):
        with st.spinner("Gemini AI is analyzing your dataset..."):
            st.markdown("#### ✨ DataSense AI says")
            st.markdown(answer_question(df, question))

with tab_export:
    st.subheader("📥 Take your findings with you")
    report_lines = [
        "# DataSense AI — Dataset Analysis Report",
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        "",
        "## Dataset overview",
        f"- Rows: {profile['rows']}",
        f"- Columns: {profile['cols']}",
        f"- Missing cells: {profile['missing_cells']} ({profile['missing_pct']}% of all cells)",
        f"- Duplicate rows: {profile['duplicates']}",
        "",
        "## Column profile",
        col_profile.to_markdown(index=False),
        "",
        "## Agent insights",
    ] + [f"- {re.sub(r'\\*\\*|`', '', insight)}" for insight in make_insights(df, profile)]
    report_text = "\n".join(report_lines)
    st.markdown("Download a text report containing the dataset profile and key findings.")
    st.download_button("⬇️ Download analysis report (.md)", data=report_text, file_name="datasense_analysis_report.md", mime="text/markdown")
    st.download_button("⬇️ Download column profile (.csv)", data=col_profile.to_csv(index=False).encode("utf-8"), file_name="datasense_column_profile.csv", mime="text/csv")
    with st.expander("Preview report"):
        st.markdown(report_text)

st.divider()
st.markdown("<div style='text-align:center;color:#a1a1aa;font-size:12px'>DataSense AI · Explore smarter. Explain clearly.</div>", unsafe_allow_html=True)