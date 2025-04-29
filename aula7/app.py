import streamlit as st
from transformers import pipeline


# Título do aplicativo
st.title("Aplicativo de Análise de Sentimentos com Hugging Face Transformers")

# Área de texto para entrada do usuário
text = st.text_area("Por favor, escreva sua sentença:")

@st.cache_resource
def load_model():
    return pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                  top_k=None)

model = load_model()


if st.button("Análise de Sentimentos"):
    if text.strip():
        try:
            # Realizando a análise de sentimentos
            results = model(text)
            
            # O resultado é uma lista de dicionários (para cada sentimento)
            st.subheader("Resultados Detalhados:")
            
            # Criando um dicionário com os resultados para exibição
            sentiment_data = {}
            for item in results[0]:  # Acessando a primeira posição da lista
                label = item['label']
                score = item['score']
                sentiment_data[label] = score
                st.write(f"{label}: {score*100:.2f}%")
            
            # Determinando o sentimento predominante
            predominant = max(results[0], key=lambda x: x['score'])
            
            # Exibindo conclusão
            st.success(
                f"Sentimento predominante: {predominant['label']} "
                f"({predominant['score']*100:.2f}% de confiança)"
            )
            
            # Gráfico de barras
            st.bar_chart(sentiment_data)
            
        except Exception as e:
            st.error(f"Ocorreu um erro: {str(e)}")
    else:
        st.warning("Por favor, insira um texto para análise.")
