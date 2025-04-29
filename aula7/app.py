import streamlit as st
from transformers import pipeline

# Título do aplicativo
st.title("Aplicativo de Análise de Sentimentos com Hugging Face Transformers")

# Área de texto para entrada do usuário
text = st.text_area("Por favor, escreva sua sentença.")

# Carregando o modelo de análise de sentimentos
model = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None)

# Botão para executar a análise
if st.button("Análise de Sentimentos"):
    if len(text) > 0:
        # Realizando a análise de sentimentos
        result = model(text)
        
        # Exibindo o resultado
        st.write(f"A sentença é {round(result['score']*100, 2)}% {result['label']}.")
        
        # Exibindo um gráfico de barras para visualizar o resultado
        st.bar_chart({result['label']: result['score']})
    else:
        st.warning("Por favor, insira um texto para análise.")

