import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Configuração da página
st.set_page_config(page_title="Análise de Sentimentos Avançada", layout="wide")

# Título e descrição
st.title("Análise de Sentimentos Avançada com BERT")
st.markdown("""
Esta aplicação utiliza um modelo BERT pré-treinado para analisar o sentimento de textos.
O modelo classifica o texto como positivo, negativo ou neutro.
""")

# Carregando o modelo e o tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("assemblyai/distilbert-base-uncased-sst2")
    model = AutoModelForSequenceClassification.from_pretrained("assemblyai/distilbert-base-uncased-sst2")
    return tokenizer, model

tokenizer, model = load_model()

# Função para tokenizar o texto
def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens

# Função para prever o sentimento
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probabilities = probabilities.numpy()[0]

    # Mapear os índices para os rótulos
    labels = ["Negativo", "Positivo"]
    results = {label: float(prob) for label, prob in zip(labels, probabilities)}

    return results

# Interface do usuário
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Digite o texto para análise:", height=200)
    analyze_button = st.button("Analisar Sentimento")
    tokenize_button = st.button("Tokenizar Texto") # Novo botão para tokenizar

# Exemplos de textos
with col2:
    st.subheader("Exemplos de Textos")
    example_texts = [
        "Eu adorei este produto! É incrível e superou todas as minhas expectativas.",
        "Estou muito decepcionado com o serviço. Foi horrível e não recomendo.",
        "O filme foi ok, nem bom nem ruim."
    ]

    for i, example in enumerate(example_texts):
        if st.button(f"Exemplo {i+1}", key=f"example_{i}"):
            text_input = example
            st.session_state.text_input = example
            analyze_button = True
            tokenize_button = False # Desabilitar o botão de tokenizar ao carregar um exemplo

# Análise e exibição dos resultados
if analyze_button and text_input:
    st.subheader("Resultado da Análise")

    with st.spinner("Analisando o sentimento..."):
        sentiment_results = predict_sentiment(text_input)

    # Exibindo os resultados
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.subheader("Probabilidades")
        for label, prob in sentiment_results.items():
            st.metric(label=label, value=f"{prob*100:.2f}%")

    with col_res2:
        st.subheader("Visualização")
        chart_data = {label: [prob] for label, prob in sentiment_results.items()}
        st.bar_chart(chart_data)

    # Determinando o sentimento predominante
    predominant_sentiment = max(sentiment_results.items(), key=lambda x: x[1])[0]
    confidence = max(sentiment_results.values())

    st.subheader("Conclusão")
    st.markdown(f"O texto tem um sentimento predominantemente **{predominant_sentiment}** com {confidence*100:.2f}% de confiança.")

# Exibição dos tokens
if tokenize_button and text_input:
    st.subheader("Tokens do Texto")
    tokens = tokenize_text(text_input)
    st.write(tokens)

# Removendo a segunda parte do código, pois ela carrega outro modelo e não está relacionada à tokenização