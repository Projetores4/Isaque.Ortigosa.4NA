import streamlit as st
import random
import openai

# Configure sua chave da API da OpenAI
openai.api_key = "SUA_CHAVE_DA_API_OPENAI"

def gerar_saudacao():
    saudacoes = ["Olá!", "Oi!", "E aí!", "Saudações!", "Como vai?"]
    return random.choice(saudacoes)

def obter_resposta_ia(mensagem):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # Escolha o modelo desejado
            prompt=mensagem,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Ocorreu um erro ao conectar com a IA: {e}"

# Inicialize mensagem_usuario com uma string vazia
if 'mensagem_usuario' not in st.session_state:
    st.session_state['mensagem_usuario'] = ""

# Inicialize o estado da conversa (se ainda não tiver)
if 'historico_chat' not in st.session_state:
    st.session_state['historico_chat'] = []

# Área para o usuário digitar a mensagem
mensagem_usuario = st.text_input("Você:", key="input_usuario", value=st.session_state['mensagem_usuario'])

# Lógica quando o usuário envia uma mensagem
if mensagem_usuario:
    # Atualiza o valor em session_state
    st.session_state['mensagem_usuario'] = mensagem_usuario

    # Adiciona a mensagem do usuário ao histórico
    st.session_state['historico_chat'].append({"usuario": mensagem_usuario})

    if mensagem_usuario.lower() == "olá":
        resposta_chatbot = gerar_saudacao()
        st.session_state['historico_chat'].append({"chatbot": resposta_chatbot})
    else:
        # Conecta com a IA para obter uma resposta
        resposta_ia = obter_resposta_ia(mensagem_usuario)
        st.session_state['historico_chat'].append({"chatbot": resposta_ia})

    # Limpa a caixa de entrada após o envio
    st.session_state["input_usuario"] = ""

# Exibe o histórico do chat
st.subheader("Histórico do Chat")
for mensagem in st.session_state['historico_chat']:
    if "usuario" in mensagem:
        st.write(f"**Você:** {mensagem['usuario']}")
    elif "chatbot" in mensagem:
        st.write(f"**Chatbot:** {mensagem['chatbot']}")