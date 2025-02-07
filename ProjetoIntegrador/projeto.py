import sqlite3  # Importa o SQLite
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import openai
 
# Carrega as variáveis de ambiente
load_dotenv()
 
# Lista global para armazenar perguntas já feitas
asked_questions = []
 
# Função para inicializar o banco de dados SQLite
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
 
# Função para salvar interações no banco de dados
def save_interaction(connection, user_input, bot_response):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO interactions (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
    connection.commit()
 
# Função para recuperar interações anteriores
def get_interactions(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT user_input, bot_response FROM interactions ORDER BY id ASC")
    rows = cursor.fetchall()
    return rows
 
# Função para ler o conteúdo do PDF
def read_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Erro ao ler o PDF: {e}")
        return ""
 
# Função para dividir o conteúdo do PDF em partes menores
def split_text(text, max_tokens=1000):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
   
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph
    if current_chunk:
        chunks.append(current_chunk)
   
    return chunks
 
# Função para buscar informações relevantes no banco de vetores
def retrieve_info(query, db, k=5):
    try:
        similar_response = db.similarity_search(query, k=k)
        return similar_response
    except Exception as e:
        print(f"Erro ao realizar a busca: {e}")
        return []
 
# Função para gerar a resposta do chatbot
def generate_response(user_input, best_practice, chain, previous_context):
    global asked_questions  # Usa a lista global para rastrear perguntas já feitas
 
    try:
        # Inclui o contexto anterior no prompt
        response = chain.run({
            "message": user_input,
            "best_practice": best_practice,
            "context": previous_context
        })
 
       
 
        # Se a resposta for uma pergunta, armazena para evitar repetição
        if "?" in response:
            asked_questions.append(response)
 
        return response
 
    except Exception as e:
        print(f"Erro ao gerar a resposta: {e}")
        return "Desculpe, não consegui processar sua solicitação."
 
# Função principal para executar o chatbot
def main():
    pdf_file_path = "c:\\Users\\lucas.eosilva\\Desktop\\ProjetoIntegrador\\Guarda.pdf"
    print(f"Verificando o caminho do arquivo: {pdf_file_path}")
 
    # Inicializa o banco de dados SQLite
    connection = init_db()
 
    # Lê o conteúdo do PDF
    pdf_text = read_pdf(pdf_file_path)
    if not pdf_text:
        print("Erro ao extrair o conteúdo do PDF.")
        return
 
    pdf_text_cleaned = pdf_text.replace("\n", " ").strip()
    text_chunks = split_text(pdf_text_cleaned)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Erro: a chave da API do OpenAI não foi encontrada.")
        return
 
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
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
 
    print("Assistente PDF: Digite 'sair' para encerrar a conversa.")
   
    while True:
        user_input = input("Você: ")
        if user_input.lower() == "sair":
            print("Assistente: Até logo!")
            break
       
        # Recupera as interações anteriores
        interactions = get_interactions(connection)
        previous_context = "\n".join([f"Usuário: {u}\nBot: {b}" for u, b in interactions])
 
        # Recupera as melhores práticas ou informações
        best_practice = "\n\n".join([doc.page_content for doc in retrieve_info(user_input, db)])
 
        # Gera a resposta
        response = generate_response(user_input, best_practice, chain, previous_context)
       
        # Salva a interação no banco de dados
        save_interaction(connection, user_input, response)
       
        print(f"Assistente: {response}")
 
if __name__ == "__main__":
    main()


