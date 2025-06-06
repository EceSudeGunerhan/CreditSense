import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="CreditSense", layout="wide")
st.title("CreditSense - Kredi Onay Tahmini ve Yardımcı Asistan")

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_API_URL = st.secrets["OPENROUTER_API_URL"]

st.sidebar.header("Kredi Başvuru Formu")

with st.sidebar.form("credit_form"):
    LOAN = st.number_input("Talep Edilen Kredi Tutarı", min_value=0.0)
    MORTDUE = st.number_input("Mevcut Konut Kredisi Borcu", min_value=0.0)
    VALUE = st.number_input("Ev/Teminat Değeri", min_value=0.0)
    REASON = st.selectbox("Kredi Kullanım Amacı", ["Borç Birleştirme (DebtCon)", "Ev Tadilatı (HomeImp)"])
    JOB = st.selectbox("Çalışma Pozisyonu", ["Ofis Çalışanı", "Diğer", "Uzman/Yönetici", "Satış Personeli", "Yönetici", "Serbest Meslek"])
    YOJ = st.number_input("Mevcut İşte Çalışma Süresi (Yıl)", min_value=0.0)
    DEROG = st.number_input("Olumsuz Kredi Kayıt Sayısı", min_value=0.0)
    DELINQ = st.number_input("Gecikmiş Ödeme Sayısı", min_value=0.0)
    CLAGE = st.number_input("Kredi Hesabı Yaşı (Ay)", min_value=0.0)
    NINQ = st.number_input("Son Dönemde Açılan Kredi Sayısı", min_value=0.0)
    CLNO = st.number_input("Toplam Kredi Hesap Sayısı", min_value=0.0)
    DEBTINC = st.number_input("Borç / Gelir Oranı (%)", min_value=0.0)

    submitted = st.form_submit_button("Tahmin Et")

if submitted:
    st.session_state["input_data"] = {
        "LOAN": LOAN,
        "MORTDUE": MORTDUE,
        "VALUE": VALUE,
        "REASON": REASON,
        "JOB": JOB,
        "YOJ": YOJ,
        "DEROG": DEROG,
        "DELINQ": DELINQ,
        "CLAGE": CLAGE,
        "NINQ": NINQ,
        "CLNO": CLNO,
        "DEBTINC": DEBTINC
    }

tab1, tab2, tab3 = st.tabs(["Tahmin Sonucu", "Karar Açıklaması", "Kredi Asistanı"])

with tab1:
    st.subheader("Model Tahmini")
    if "input_data" in st.session_state:
        try:
            response = requests.post("https://creditsense.onrender.com/predict", json=st.session_state["input_data"])
            prediction = response.json()["prediction_label"]
            st.success(f"Sonuç: {prediction}")
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")
    else:
        st.info("Formu doldurup tahmin etmelisiniz.")

with tab2:
    st.subheader("Karar Açıklaması")
    if "input_data" in st.session_state:
        try:
            shap_res = requests.post("https://creditsense.onrender.com/explain", json=st.session_state["input_data"])
            shap_json = shap_res.json()
            st.markdown("### Model Açıklaması")
            st.code(shap_json["explanation"], language="markdown")
            chart_df = pd.DataFrame({
                "Özellik": shap_json["shap_chart"]["features"],
                "Etki": shap_json["shap_chart"]["values"]
            })
            st.bar_chart(chart_df.set_index("Özellik"))
        except Exception as e:
            st.error(f"SHAP açıklaması alınamadı: {e}")
    else:
        st.info("Tahmin yaptıktan sonra açıklama görüntülenebilir.")

with tab3:
    st.subheader(" Kredi Asistanı")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_query = st.text_input("Soru sor (örn: Neden kredim onaylanmadı?)")

    if user_query:
        if "input_data" not in st.session_state:
            st.warning("Önce formu doldurup tahmin yapmalısınız.")
        else:
            try:
                payload = {
                    "question": user_query,
                    "features": st.session_state["input_data"]
                }
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://creditsense.local",
                    "X-Title": "CreditSense Assistant"
                }
                llm_payload = {
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "Sen bir kredi başvuru asistanısın. Kullanıcının verilerine göre açıklama yap."},
                        {"role": "user", "content": f"Kullanıcı verileri: {payload['features']}\nSoru: {payload['question']}"}
                    ],
                    "temperature": 0.7
                }
                res = requests.post(OPENROUTER_API_URL, headers=headers, json=llm_payload)
                res.raise_for_status()
                reply = res.json()["choices"][0]["message"]["content"]
                st.session_state["chat_history"].append(("Siz", user_query))
                st.session_state["chat_history"].append(("Asistan", reply))
            except Exception as e:
                st.error(f"API hatası: {e}")

    for role, msg in st.session_state["chat_history"]:
        st.markdown(f"**{role}:** {msg}")

# streamlit run streamlit_app.py
