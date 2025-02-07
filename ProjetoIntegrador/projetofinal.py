import os
import sqlite3
import openai
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
 
# Carregar variáveis de ambiente
load_dotenv()
 
# Inicializa o cliente OpenAI
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
# Banco de dados SQLite
 
def init_db():
    connection = sqlite3.connect("chatbot.db")
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        bot_response TEXT
    )
    """)
    connection.commit()
    return connection
 
def save_interaction(connection, user_input, bot_response):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO interactions (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
    connection.commit()
 
def get_interactions(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT user_input, bot_response FROM interactions ORDER BY id ASC")
    return cursor.fetchall()
 
# Funções de áudio
def gravar_audio(nome_arquivo="audio.wav", duracao=5, samplerate=44100):
    print("🎤 Gravando... Fale agora!")
    audio = sd.rec(int(duracao * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(nome_arquivo, samplerate, audio)
    print("✅ Gravação finalizada!")
 
def transcrever_audio(nome_arquivo="audio.wav"):
    with open(nome_arquivo, "rb") as audio_file:
        resposta = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return resposta.text
 
def texto_para_audio(texto, nome_arquivo="resposta.mp3"):
    resposta = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=texto
    )
    with open(nome_arquivo, "wb") as f:
        f.write(resposta.content)
    os.system(f"start {nome_arquivo}" if os.name == "nt" else f"mpg123 {nome_arquivo}")
 
# Processamento de PDF
def read_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, "rb") as file:
            reader = PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Erro ao ler o PDF: {e}")
        return ""
 
def split_text(text, max_tokens=1000):
    paragraphs = text.split("\n")
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
 
def retrieve_info(query, db, k=5):
    try:
        return db.similarity_search(query, k=k)
    except Exception as e:
        print(f"Erro ao buscar no banco de vetores: {e}")
        return []
 
def generate_response(user_input, best_practice, chain, previous_context):
    try:
        return chain.run({"message": user_input, "best_practice": best_practice, "context": previous_context})
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return "Desculpe, não consegui processar sua solicitação."
 
def chatbot():
    pdf_file_path = "c:\\Users\\lucas.eosilva\\Desktop\\ProjetoIntegrador\\Guarda.pdf"
    connection = init_db()
    pdf_text = read_pdf(pdf_file_path)
    if not pdf_text:
        print("Erro ao extrair conteúdo do PDF.")
        return
 
    text_chunks = split_text(pdf_text.replace("\n", " ").strip())
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(documents, embeddings)
   
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    prompt_template = """CONTEXTO:
    VOCÊ É UM PROFESSOR ESPECIALISTA NO ASSUNTO TRATADO NO ARQUIVO ANEXADO, INDEPENDENTE DE QUAL FOR O ASSUNTO. VOCÊ DEVE ENSINAR O USUÁRIO ATÉ QUE O MESMO SE SINTA UM ESPECIALISTA NO ASSUNTO.
 
    ESTILO E TOM:
    VOCÊ DEVE TER UM TOM AMIGÁVEL E DEVE ENSINAR DA FORMA MAIS CLARA POSSÍVEL PARA QUE SEJA CLARO AO USUÁRIO AQUILO O QUE ELE QUER APRENDER.
 
    REGRAS:
    1° NUNCA FAZER A MESMA PERGUNTA DURANTE OS TESTES;
    2° VOCÊ DEVE INICIAR FAZENDO UM TESTE COM 3 PERGUNTAS(UMA DE CADA VEZ);
        - 1° PERGUNTA: O QUE VOCÊ JÁ SABE SOBRE O ASSUNTO?
        - 2° PERGUNTA: QUAL É A SUA MAIOR DIFICULDADE SOBRE O ASSUNTO?
        - 3° VOCÊ PREFERE APRENDER DE QUE FORMA?
    3° APÓS O USUÁRIO RESPONDER ESSAS 3 PERGUNTAS E VOCÊ ARMAZENA-LAS, PERGUNTE: COMO VOCÊ QUER COMEÇAR?;
    4° VOCÊ FARÁ UMA PERGUNTA POR VEZ. OU SEJA, PARA FAZER A PERGUNTA SEGUINTE O USUÁRIO TERÁ QUE RESPONDER A PERGUNTA ANTERIOR;
    5° VOCÊ DEVE ORIENTAR O USUÁRIO PROGRESSIVAMENTE E NÃO ENTREGAR O CONTEÚDO DE UMA VEZ;
    6° SUA MISSÃO É FAZER COM QUE O USUÁRIO SE TORNE UM ESPECIALISTA NO ASSUNTO.
    
   
    FORMATO:
    O FORMATO DE ENSINO DEVE SER CLARO PARA O USUÁRIO.
 
    CONVERSA ATUAL:
    {context}
 
    MENSAGEM DO USUÁRIO:
    {message}
 
    MELHORES PRÁTICAS:
    {best_practice}
    """
    prompt = PromptTemplate(input_variables=["message", "best_practice", "context"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
 
    print("Assistente: Digite 'sair' para encerrar. Ou diga algo para iniciar pelo áudio!")
    while True:
        escolha = input("Digite '1' para texto ou '2' para áudio: ")
        if escolha == "1":
            user_input = input("Você: ")
        elif escolha == "2":
            gravar_audio()
            user_input = transcrever_audio()
            print("👤 Usuário:", user_input)
        else:
            print("Opção inválida.")
            continue
       
        if user_input.lower() == "sair":
            print("Assistente: Até logo!")
            break
 
        interactions = get_interactions(connection)
        previous_context = "\n".join([f"Usuário: {u}\nBot: {b}" for u, b in interactions])
        best_practice = "\n\n".join([doc.page_content for doc in retrieve_info(user_input, db)])
        response = generate_response(user_input, best_practice, chain, previous_context)
        save_interaction(connection, user_input, response)
        print(f"Assistente: {response}")
        texto_para_audio(response)
 
if __name__ == "__main__":
    chatbot()