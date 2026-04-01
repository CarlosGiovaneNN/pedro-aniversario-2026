import time

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras_facenet import FaceNet
import joblib
import requests  
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Feliz Aniversário Pedro", layout="centered")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_celebration = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_u4yrau.json")

@st.cache_resource
def load_models():
    embedder = FaceNet()
    
    model = joblib.load('face_classifier_svm.pkl')
    encoder = joblib.load('label_encoder.pkl')
    
    return embedder, model, encoder

try:
    embedder, model, encoder = load_models()
except FileNotFoundError:
    st.error("Arquivos de modelo (.pkl) não encontrados. Certifique-se de subir o modelo e o encoder.")
    st.stop()

st.title("Quem é a pessoa?", text_alignment="center")
st.write("Faça upload de uma foto para identificar se é o Pedro ou outra pessoa." )

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    img_array = np.array(image)
    
    st.image(image, caption='Imagem carregada', use_column_width=True)

    c = st.container(horizontal_alignment="center")
    
    if c.button('Identificar Pessoa'):
        with c.spinner('Analisando rosto...'):
            detections = embedder.extract(img_array, threshold=0.95)

            if len(detections) > 0:
                embedding = detections[0]["embedding"]
                embedding = np.expand_dims(embedding, axis=0)
                
                prediction = model.predict(embedding)
                prob = model.predict_proba(embedding)
                class_name = encoder.inverse_transform(prediction)[0]
                confianca = np.max(prob)

                if class_name == "Pedro":
                    st.success(f"Identificado como **{class_name}**")
                    st.success(f"🎊 IDENTIDADE CONFIRMADA! FELIZ ANIVERSÁRIO, PEDRO! 🎊")
                    st.balloons() 
                else:
                    st.info(f"Rosto identificado como: **Outro desgraçado**", )
                c.metric("Confiança", f"{np.max(prob):.2%}")

                if class_name == "Pedro":
                    time.sleep(2)
                    st.balloons() 
            else:
                st.warning("Nenhum rosto detectado na imagem. Tente outra foto.")