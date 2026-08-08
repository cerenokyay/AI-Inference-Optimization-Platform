import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import os
import hashlib
from datetime import datetime

# ==========================================
# AYARLAR VE YARDIMCI FONKSİYONLAR
# ==========================================
API_BASE_URL = "http://127.0.0.1:8000"
LOCAL_MAP_FILE = "frontend/hash_mapper.json"

def get_prompt_hash(prompt_text):
    normalized = " ".join(prompt_text.strip().lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def save_prompt_mapping(raw_prompt, req_type):
    actual_sent = f"[test_mode] {raw_prompt}" if req_type == "test" else raw_prompt
    mapping = {}
    if os.path.exists(LOCAL_MAP_FILE):
        with open(LOCAL_MAP_FILE, "r") as f:
            mapping = json.load(f)
    
    hash_val = get_prompt_hash(actual_sent)
    mapping[hash_val] = {"text": raw_prompt, "type": req_type}
    
    with open(LOCAL_MAP_FILE, "w") as f:
        json.dump(mapping, f)

def load_mapping():
    if os.path.exists(LOCAL_MAP_FILE):
        with open(LOCAL_MAP_FILE, "r") as f:
            return json.load(f)
    return {}

# ==========================================
# SAYFA YAPILANDIRMASI VE APPLE ESTETİĞİ CSS
# ==========================================
st.set_page_config(page_title="AI Inference Gateway | Apple Style", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Apple San Francisco / Inter Tarzı Modern Tipografi */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Derin Apple Siyahı Arka Plan */
    .stApp {
        background-color: #000000;
        color: #F5F5F7;
    }
    
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    .block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 90%;}

    /* Apple Hero Başlık Efekti */
    .apple-hero {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .apple-title {
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        color: #F5F5F7;
        margin-bottom: 5px;
        background: linear-gradient(180deg, #FFFFFF 0%, #A1A1A6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .apple-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        color: #86868B;
        letter-spacing: -0.009em;
    }

    /* Apple Bento Grid / Cam Efektli Kartlar */
    div[data-testid="metric-container"], div.st-key-control_box {
        background: rgba(28, 28, 30, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s cubic-bezier(0.25, 1, 0.5, 1), border-color 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    /* Apple Tarzı Hap (Pill) Sekmeler */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(26, 26, 28, 0.8);
        border-radius: 99px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        gap: 6px;
        justify-content: center;
        display: flex;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 99px;
        padding: 8px 20px;
        border: none;
        color: #86868B;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.15);
    }
    
    /* Minimalist Input Alanları */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(44, 44, 46, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #F5F5F7 !important;
    }
    .stTextInput input:focus {
        border-color: #2997FF !important; /* Apple Mavi Vurgusu */
        box-shadow: 0 0 0 2px rgba(41, 151, 255, 0.3) !important;
    }
    
    /* Şık Gönder Butonu */
    .stButton button {
        background-color: #2997FF;
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0077ED;
        box-shadow: 0 4px 15px rgba(41, 151, 255, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Apple Tarzı Hero Alanı
st.markdown("""
    <div class="apple-hero">
        <h1 class="apple-title">AI Inference Gateway.</h1>
        <p class="apple-subtitle">Maliyetleri minimize eden, gecikmeyi yok eden yeni nesil yapay zeka geçidi.</p>
    </div>
    <br>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Genel Bakış", 
    "🧪 Test Merkezi", 
    "🧠 Sistem Mimar", 
    "📜 Açık Loglar", 
    "🔒 Güvenli Loglar"
])

def render_model_controls(tab_key):
    with st.container(border=True):
        st.markdown(f"**⚙️ {tab_key.capitalize()} Motor Yapılandırması**")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            provider = st.selectbox("Sağlayıcı", ["ollama", "openai", "anthropic"], key=f"prov_{tab_key}")
        
        default_model = "qwen2.5:3b"
        if provider == "openai": default_model = "gpt-4o"
        elif provider == "anthropic": default_model = "claude-3-5-sonnet-20240620"
        
        with c2:
            model_name = st.text_input("Model Kimliği", value=default_model, key=f"mod_{tab_key}")
            
        with c3:
            api_key = ""
            if provider in ["openai", "anthropic"]:
                api_key = st.text_input(f"{provider.capitalize()} API Key", type="password", key=f"api_{tab_key}")
            else:
                st.caption("✨ Yerel model aktif (API Key gerektirmez)")
                
    return provider, model_name, api_key

def chat_interface(tab_key, req_type, placeholder_text):
    selected_provider, selected_model, api_key = render_model_controls(tab_key)
    
    with st.expander(" Kendi Özel Modelimi Nasıl Entegre Ederim?"):
        st.markdown("""
        * **Yerel Model (Ollama):** Terminale `ollama pull <model>` komutunu yazın. İndikten sonra yukarıdaki model alanına adını yazıp doğrudan kullanmaya başlayın.
        * **Bulut Model (OpenAI / Anthropic):** Herhangi bir kurulum gerekmez. Model adını ve kendi güvenli API anahtarınızı girerek anında sisteme dahil edin.
        """)

    if f"messages_{req_type}" not in st.session_state:
        st.session_state[f"messages_{req_type}"] = []

    chat_container = st.container(height=380, border=False)
    
    with chat_container:
        for msg in st.session_state[f"messages_{req_type}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with st.form(key=f"chat_form_{req_type}", clear_on_submit=True):
        cols = st.columns([9, 1])
        with cols[0]:
            prompt = st.text_input("Mesaj", placeholder=placeholder_text, label_visibility="collapsed")
        with cols[1]:
            submit = st.form_submit_button("Gönder", use_container_width=True)

    if submit and prompt:
        st.session_state[f"messages_{req_type}"].append({"role": "user", "content": prompt})
        save_prompt_mapping(prompt, req_type)
        api_prompt = f"[test_mode] {prompt}" if req_type == "test" else prompt

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                payload = {
                    "prompt": api_prompt,
                    "provider": selected_provider,
                    "model_name": selected_model,
                    "api_key": api_key
                }
                with requests.post(f"{API_BASE_URL}/generate", json=payload, stream=True) as r:
                    response_stream = (chunk for chunk in r.iter_content(chunk_size=1024, decode_unicode=True) if chunk)
                    full_resp = st.write_stream(response_stream)
        
        st.session_state[f"messages_{req_type}"].append({"role": "assistant", "content": full_resp})
        st.rerun()

with tab1:
    try:
        metrics = requests.get(f"{API_BASE_URL}/metrics").json().get("data", {})
        total_req = metrics.get("total_requests", 0)
        cache_hits = metrics.get("cache_hits", 0) + metrics.get("semantic_cache_hits", 0)
        hit_rate = (cache_hits / total_req * 100) if total_req > 0 else 0.0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam İstek", total_req)
        c2.metric("Cache Hit Rate", f"%{hit_rate:.1f}")
        c3.metric("Ortalama Gecikme", f"{metrics.get('avg_request_latency_ms', 0.0):.2f} ms")
        c4.metric("Aktif Vektör", metrics.get("semantic_vector_count", 0))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        history_data = requests.get(f"{API_BASE_URL}/analytics/history?limit=100").json().get("data", [])
        if history_data:
            df = pd.DataFrame(history_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_lat = px.line(df, x='timestamp', y='total_latency_ms', color='cache_status', 
                                  title="Zaman İçinde Gecikme Analizi (ms)", template="plotly_dark")
                fig_lat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_lat, use_container_width=True)
            with col_chart2:
                fig_pie = px.pie(df, names='cache_status', title="Önbellek Dağılımı Oranı", template="plotly_dark",
                                 color_discrete_map={'MISS':'#FF453A', 'EXACT_HIT':'#30D158', 'SEMANTIC_HIT':'#0A84FF'})
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
    except:
        st.error("Backend sunucusuna ulaşılamadı. Lütfen API'nin çalıştığından emin olun.")

with tab2:
    st.markdown("<p style='color: #86868B;'>Çıkarım Test Merkezi: Model performansını ham ve filtresiz olarak test edin.</p>", unsafe_allow_html=True)
    chat_interface("test", "test", "Örn: Transformer mimarisi nasıl çalışır?")

with tab3:
    st.markdown("<p style='color: #86868B;'>Sistem Mimarı: Canlı telemetri verileriyle optimize edilmiş stratejik yapay zeka asistanı.</p>", unsafe_allow_html=True)
    chat_interface("mimar", "chat", "Örn: Hit rate oranını artırmak için ne yapmalıyım?")

def render_logs(req_type_filter, require_auth=False):
    history_data = requests.get(f"{API_BASE_URL}/analytics/history?limit=200").json().get("data", [])
    if not history_data:
        st.info("Kayıtlı log verisi bulunmuyor.")
        return

    mapping = load_mapping()
    display_data = []

    for row in history_data:
        h = row['prompt_hash']
        row_type = mapping.get(h, {}).get('type', 'Bilinmiyor')
        
        if row_type == req_type_filter:
            dt_str = pd.to_datetime(row['timestamp']).strftime('%d.%m.%Y %H:%M:%S')
            clear_text = mapping.get(h, {}).get('text', '*** Eşleşme Yok ***')
            
            if require_auth and not st.session_state.get("auth_ok", False):
                clear_text = "******** (Yetki Gerekli)"

            display_data.append({
                "Zaman": dt_str,
                "İstem (Prompt)": clear_text,
                "Durum": row['cache_status'],
                "Gecikme (ms)": f"{row['total_latency_ms']:.2f}",
                "Sağlayıcı": row['provider'],
                "Model": row['model_name']
            })
            
    if display_data:
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        st.info("Bu kategoriye ait işlem günlüğü henüz oluşmadı.")

with tab4:
    st.header("📜 Açık Test Günlükleri")
    render_logs("test", require_auth=False)

with tab5:
    st.header("🔒 Güvenli Mimari Günlükleri")
    col1, col2 = st.columns([1, 4])
    with col1:
        pwd = st.text_input("Yönetici Parolası", type="password", key="admin_pwd")
        if pwd == "admin123":
            st.session_state["auth_ok"] = True
            st.success("Erişim Onaylandı")
        elif pwd != "":
            st.session_state["auth_ok"] = False
            st.error("Geçersiz Parola")
            
    render_logs("chat", require_auth=True)