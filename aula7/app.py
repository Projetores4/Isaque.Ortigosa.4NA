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

        # Certifica-se de que 'result' é uma lista e pega o primeiro elemento
        if isinstance(result, list) and len(result) > 0:
            top_result = result[0]
            st.write(f"A sentença é {round(top_result['score']*100, 2)}% {top_result['label']}.")

            # Exibindo um gráfico de barras para visualizar o resultado
            st.bar_chart({top_result['label']: top_result['score']})
        else:
            st.warning("Não foi possível analisar o sentimento.")
    else:
        st.warning("Por favor, insira um texto para análise.")

