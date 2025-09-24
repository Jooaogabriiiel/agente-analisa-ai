# app.py (Versão Final Corrigida para Gráficos)

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import io
import contextlib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. CONFIGURAÇÃO INICIAL ---

st.set_page_config(layout="wide", page_title="Agente de Análise de Dados")
st.title("Analisa AI")
st.write("Envie seu arquivo CSV e faça perguntas. A IA irá gerar a análise e o código para você.")

# Tenta pegar a chave da API das variáveis de ambiente ou dos segredos do Streamlit
try:
    api_key = os.environ['GEMINI_API_KEY']
except KeyError:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("Chave da API do Gemini não encontrada. Verifique as configurações de segredos.")
        api_key = None

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

# --- 3. UPLOAD DO ARQUIVO E LÓGICA PRINCIPAL ---

st.sidebar.header("1. Envie seu arquivo CSV")
arquivo_enviado = st.sidebar.file_uploader("Escolha um arquivo CSV", type=["csv"])

if 'df' not in st.session_state:
    st.session_state.df = None

if arquivo_enviado is not None:
    try:
        st.session_state.df = pd.read_csv(arquivo_enviado)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {e}")
else:
    pass

# --- 4. O CÉREBRO DO AGENTE (FUNÇÕES) ---

@st.cache_data
def gerar_contexto_dados(df_para_contexto):
    if df_para_contexto is None:
        return "Nenhum dado carregado ainda."
    buffer = io.StringIO()
    df_para_contexto.info(buf=buffer)
    info_dados = buffer.getvalue()
    contexto = f"""
    Informações sobre o DataFrame (df):
    {info_dados}
    Primeiras 5 linhas do DataFrame (df):
    {df_para_contexto.head().to_string()}
    """
    return contexto

def perguntar_ao_agente(pergunta_usuario, df_atual):
    contexto_dados = gerar_contexto_dados(df_atual)
    prompt = f"""
    Você é um assistente de análise de dados expert em Python.
    Sua tarefa é gerar um código Python para responder a uma pergunta do usuário sobre um DataFrame do pandas chamado 'df'.

    **Regras Estritas:**
    1.  **Gere APENAS o código Python.**
    2.  Não inclua nenhuma explicação, texto, introdução ou conclusão.
    3.  Não use a formatação de bloco de código do markdown (ex: ```python ... ```).
    4.  O código deve usar o DataFrame que já está em memória, chamado 'df'.
    5.  Para respostas em texto, o código deve imprimir o resultado final na tela (usando a função print()).
    6.  Use as bibliotecas 'matplotlib.pyplot as plt' e 'seaborn as sns' para gerar gráficos.
    7.  Se a pergunta pedir uma análise visual, gere o código para um gráfico apropriado.
    8.  **SEMPRE inclua 'import matplotlib.pyplot as plt' e 'import seaborn as sns' no início do código do gráfico.**
    9.  ### A REGRA MAIS IMPORTANTE ### Para qualquer gráfico, SEMPRE use o objeto de eixos 'ax' que já está disponível no ambiente de execução. Adicione o parâmetro 'ax=ax' a todas as chamadas de plotagem. Exemplos: `sns.histplot(data=df, x='column', ax=ax)`, `df.plot(kind='scatter', x='col1', y='col2', ax=ax)`.
    10. **NUNCA use `plt.show()` no seu código.** A exibição será feita pelo Streamlit.
    11. Adicione títulos claros e nomes para os eixos nos gráficos usando os métodos de `ax`. Exemplo: `ax.set_title('Título')`, `ax.set_xlabel('Eixo X')`.

    Contexto dos dados:
    {contexto_dados}
    ---
    **Pergunta do Usuário:** "{pergunta_usuario}"
    ---
    **Código Python gerado por você:**
    """
    response = model.generate_content(prompt)
    codigo_gerado = response.text
    codigo_gerado = codigo_gerado.replace("```python", "").replace("```", "").strip()
    return codigo_gerado

# --- 5. A INTERFACE DO USUÁRIO (A "CARA" DO APP) ---

if st.session_state.df is not None:
    st.sidebar.header("2. Faça sua pergunta")
    df = st.session_state.df

    st.subheader("Visualização dos Dados Carregados")
    st.dataframe(df.head())

    pergunta = st.sidebar.text_input("Sua pergunta:", placeholder="Ex: Crie um histograma da coluna 'Amount'")

    if st.sidebar.button("Analisar 👨‍💻"):
        if not api_key:
            st.warning("A chave da API do Gemini não foi configurada.")
        elif pergunta:
            with st.spinner("O agente está pensando e gerando o código..."):
                codigo_gerado = perguntar_ao_agente(pergunta, df)

            st.subheader("Código Gerado pelo Agente")
            st.code(codigo_gerado, language='python')

            st.subheader("Resultado da Análise")
            with st.spinner("Executando a análise..."):
                # ### MUDANÇA: Lógica de execução simplificada ###
                try:
                    # Prepara a "tela de pintura"
                    fig, ax = plt.subplots()

                    # Prepara o ambiente para a execução
                    local_vars = {'df': df, 'plt': plt, 'sns': sns, 'fig': fig, 'ax': ax}

                    # Executa o código gerado
                    exec(codigo_gerado, {}, local_vars)

                    # Verifica se algo foi desenhado no gráfico
                    if ax.has_data():
                        st.pyplot(fig)
                    else:
                        # Se não for gráfico, tenta capturar a saída de texto
                        output_buffer = io.StringIO()
                        with contextlib.redirect_stdout(output_buffer):
                             exec(codigo_gerado)
                        resultado_texto = output_buffer.getvalue()
                        if resultado_texto:
                            st.text(resultado_texto)
                        else:
                            st.write("O código foi executado, mas não produziu uma saída de texto ou gráfico visível.")

                except Exception as e:
                    st.error(f"Ocorreu um erro ao executar o código: {e}")
        else:
            st.warning("Por favor, digite uma pergunta.")
else:
    st.info("Aguardando o envio de um arquivo CSV para começar a análise.")
