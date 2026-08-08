import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import os
import hashlib
from datetime import datetime

# Backend API Adresi
API_BASE_URL = "http://127.0.0.1:8000"
LOCAL_MAP_FILE = "frontend/hash_mapper.json" # Şifre-Metin eşleşmelerini tutan gizli dosya

# --- YARDIMCI FONKSİYONLAR (Güvenlik ve Hash) ---
def get_prompt_hash(prompt_text):
    """Backend'in ürettiği Hash mantığının aynısını Frontend'de kopyalarız"""
    normalized = " ".join(prompt_text.strip().lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def save_prompt_mapping(raw_prompt, req_type):
    """Atılan isteği şifreli koduyla birlikte yerel JSON dosyasına kaydeder"""
    actual_sent = f"[TEST_MODE] {raw_prompt}" if req_type == "test" else raw_prompt
    
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

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="AI Inference Gateway", page_icon="⚡", layout="wide")
st.title("⚡ AI Inference Optimization Gateway")

# 5 SEKME: Vizyonunun Koda Dökülmüş Hali
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🧪 Çıkarım Test Merkezi", 
    "🧠 Sistem Mimarı", 
    "📜 Açık Test Logları", 
    "🔒 Güvenli Mimari Loglar"
])

# ==========================================
# SEKME 1: DASHBOARD
# ==========================================
with tab1:
    try:
        metrics = requests.get(f"{API_BASE_URL}/metrics").json().get("data", {})
        total_req = metrics.get("total_requests", 0)
        cache_hits = metrics.get("cache_hits", 0) + metrics.get("semantic_cache_hits", 0)
        hit_rate = (cache_hits / total_req * 100) if total_req > 0 else 0.0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam İstek", total_req)
        c2.metric("Hit Rate", f"%{hit_rate:.1f}")
        c3.metric("Ort. Gecikme", f"{metrics.get('avg_request_latency_ms', 0.0):.2f} ms")
        c4.metric("Aktif Vektör", metrics.get("semantic_vector_count", 0))
        
        history_data = requests.get(f"{API_BASE_URL}/analytics/history?limit=100").json().get("data", [])
        if history_data:
            df = pd.DataFrame(history_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_lat = px.line(df, x='timestamp', y='total_latency_ms', color='cache_status', markers=True, title="Zaman/Gecikme (ms)")
                st.plotly_chart(fig_lat, use_container_width=True)
            with col_chart2:
                fig_pie = px.pie(df, names='cache_status', title="Önbellek Dağılımı")
                st.plotly_chart(fig_pie, use_container_width=True)
    except:
        st.error("Backend'e ulaşılamadı.")

# ==========================================
# ORTAK SOHBET FONKSİYONU
# ==========================================
def chat_interface(tab_name, req_type, placeholder_text):
    if f"messages_{req_type}" not in st.session_state:
        st.session_state[f"messages_{req_type}"] = []

    # Mesaj ekranı için sabit yükseklik (Input kutusu alta sabitlenir)
    chat_container = st.container(height=450, border=False)
    
    with chat_container:
        for msg in st.session_state[f"messages_{req_type}"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input(placeholder_text, key=f"input_{req_type}"):
        st.session_state[f"messages_{req_type}"].append({"role": "user", "content": prompt})
        
        # Test modundaysak arkada [TEST_MODE] gizli etiketini ekle
        save_prompt_mapping(prompt, req_type)
        api_prompt = f"[TEST_MODE] {prompt}" if req_type == "test" else prompt

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with requests.post(f"{API_BASE_URL}/generate", json={"prompt": api_prompt}, stream=True) as r:
                    response_stream = (chunk for chunk in r.iter_content(chunk_size=1024, decode_unicode=True) if chunk)
                    full_resp = st.write_stream(response_stream)
        
        st.session_state[f"messages_{req_type}"].append({"role": "assistant", "content": full_resp})

# ==========================================
# SEKME 2 & 3: SOHBET EKRANLARI
# ==========================================
with tab2:
    st.header("🧪 Çıkarım Test Merkezi")
    st.caption("Buraya yazılan sorular ham halleriyle, persona olmadan doğrudan modele ve önbelleğe gider. Performans testi içindir.")
    chat_interface("tab2", "test", "Örn: Hiperparametre optimizasyonu nedir?")

with tab3:
    st.header("🧠 Sistem Mimarı")
    st.caption("Canlı sistem metriklerine hakim yapay zeka asistanından optimizasyon ve mimari önerileri alın.")
    chat_interface("tab3", "chat", "Örn: Şu anki Hit Rate değerimi nasıl artırabilirim?")

# ==========================================
# SEKME 4 & 5: LOG KAYITLARI (GÜVENLİ VE AÇIK)
# ==========================================
def render_logs(req_type_filter, require_auth=False):
    history_data = requests.get(f"{API_BASE_URL}/analytics/history?limit=200").json().get("data", [])
    if not history_data:
        st.info("Henüz log bulunmuyor.")
        return

    mapping = load_mapping()
    display_data = []

    # Backend'den gelen her veriyi frontend'deki haritayla eşleştir
    for row in history_data:
        h = row['prompt_hash']
        row_type = mapping.get(h, {}).get('type', 'Bilinmiyor')
        
        if row_type == req_type_filter:
            dt_str = pd.to_datetime(row['timestamp']).strftime('%d.%m.%Y %H:%M:%S')
            clear_text = mapping.get(h, {}).get('text', '*** İstemci Eşleşmesi Yok ***')
            
            # Eğer şifreli sekmedeysek ve yetki yoksa metni gizle
            if require_auth and not st.session_state.get("auth_ok", False):
                clear_text = "******** (Görüntülemek için şifre girin)"

            display_data.append({
                "Tarih/Saat": dt_str,
                "İstek (Metin)": clear_text,
                "İşlem Sonucu": row['cache_status'],
                "Gecikme (ms)": f"{row['total_latency_ms']:.2f}",
                "Model": row['model_name'],
                "KVKK Hash": h[:15] + "..."
            })
            
    if display_data:
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        st.info(f"Bu kategoriye ait log henüz oluşmamış.")

with tab4:
    st.header("📜 Açık Test Logları")
    st.caption("Sadece test merkezinden atılan ve KVKK/Gizlilik içermeyen sistem sorgularının kayıtları.")
    render_logs("test", require_auth=False)

with tab5:
    st.header("🔒 Güvenli Mimari Sohbet Logları")
    st.caption("Sistem mimarıyla yapılan stratejik tartışmalar şifrelidir. Sadece yetkili personel görüntüleyebilir.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        pwd = st.text_input("Admin Şifresi", type="password", key="admin_pwd")
        if pwd == "admin123":
            st.session_state["auth_ok"] = True
            st.success("Yetki Onaylandı!")
        elif pwd != "":
            st.session_state["auth_ok"] = False
            st.error("Hatalı Şifre!")
            
    render_logs("chat", require_auth=True)