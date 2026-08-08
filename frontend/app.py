import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Backend API Adresi
API_BASE_URL = "http://127.0.0.1:8000"

# Sayfa Yapılandırması
st.set_page_config(
    page_title="AI Inference Optimization Platform",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI Inference Optimization Gateway")
st.markdown("API maliyetlerini ve gecikmeyi minimize eden, Clean Architecture tabanlı asenkron yapay zeka geçidi.")

# Vizyondaki İki Sayfalı Yapı: Sekmeler
tab1, tab2 = st.tabs(["📊 Observability Dashboard", "🧠 AI Analyst Chat"])

# ==========================================
# 1. SEKME: ÖLÇÜMLER (DASHBOARD)
# ==========================================
with tab1:
    st.header("Sistem Performans Metrikleri")
    
    # Anlık Metrikleri Çek
    try:
        metrics_resp = requests.get(f"{API_BASE_URL}/metrics").json()
        metrics = metrics_resp.get("data", {})
        
        total_req = metrics.get("total_requests", 0)
        cache_hits = metrics.get("cache_hits", 0)
        semantic_hits = metrics.get("semantic_cache_hits", 0)
        
        hit_rate = 0.0
        if total_req > 0:
            hit_rate = ((cache_hits + semantic_hits) / total_req) * 100
            
        avg_latency = metrics.get("avg_request_latency_ms", 0.0)

        # Tepe Kartları (KPIs)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Toplam İstek", total_req)
        col2.metric("Cache Hit Rate", f"%{hit_rate:.1f}")
        col3.metric("Ortalama Gecikme", f"{avg_latency:.2f} ms")
        col4.metric("Aktif Vektör Sayısı", metrics.get("semantic_vector_count", 0))
        
        st.divider()
        
        # Geçmiş Verileri (SQLite) Çek ve Grafiğe Dök
        history_resp = requests.get(f"{API_BASE_URL}/analytics/history?limit=100").json()
        history_data = history_resp.get("data", [])
        
        if history_data:
            df = pd.DataFrame(history_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("İstek Süreleri (Latency)")
                fig_latency = px.line(df, x='timestamp', y='total_latency_ms', color='cache_status', 
                                      markers=True, title="Zaman İçinde Gecikme (ms)")
                st.plotly_chart(fig_latency, use_container_width=True)
                
            with col_chart2:
                st.subheader("Cache Durumu Dağılımı")
                fig_pie = px.pie(df, names='cache_status', title="Önbellek İsabet Oranları", 
                                 color='cache_status', 
                                 color_discrete_map={'MISS':'#EF553B', 'EXACT_HIT':'#00CC96', 'SEMANTIC_HIT':'#636EFA'})
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Henüz geçmiş veri bulunmuyor. Sistem test edildikçe grafikler burada oluşacaktır.")
            
    except Exception as e:
        st.error(f"Backend API'ye ulaşılamadı. Sunucunun açık olduğundan emin olun. Hata: {e}")

# ==========================================
# 2. SEKME: AI ANALİST (CHAT)
# ==========================================
with tab2:
    st.header("Sistem Mimarı ile Görüş")
    st.markdown("Arka plandaki canlı telemetri verilerine hakim yapay zeka asistanından optimizasyon önerileri alın.")

    # Sohbet Geçmişini Başlat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları ekranda göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Backend'den streaming yanıt alan jeneratör fonksiyon
    def generate_response(prompt_text):
        with requests.post(f"{API_BASE_URL}/generate", json={"prompt": prompt_text}, stream=True) as r:
            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    yield chunk

    # Kullanıcı yeni mesaj girdiğinde
    if prompt := st.chat_input("Örn: Şu anki performansımı nasıl artırabilirim?"):
        # Kullanıcı mesajını kaydet ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI yanıtını stream olarak al ve göster
        with st.chat_message("assistant"):
            response_stream = generate_response(prompt)
            full_response = st.write_stream(response_stream)
        
        # Tamamlanan AI yanıtını geçmişe kaydet
        st.session_state.messages.append({"role": "assistant", "content": full_response})