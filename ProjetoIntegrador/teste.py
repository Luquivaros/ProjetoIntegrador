import openai
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

# CONFIGURANDO O CLIENTE OPENAI
client = openai.OpenAI(api_key="API-KEY")

# 🎤 Função para gravar áudio do usuário
def gravar_audio(nome_arquivo="audio.wav", duracao=5, samplerate=44100):
    print("🎤 Gravando... Fale agora!")
    audio = sd.rec(int(duracao * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(nome_arquivo, samplerate, audio)
    print("✅ Gravação finalizada!")

# 📝 Função para transcrever áudio com Whisper (NOVA API)
def transcrever_audio(nome_arquivo="audio.wav"):
    with open(nome_arquivo, "rb") as audio_file:
        resposta = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return resposta.text  # Acessando o texto da resposta corretamente

# 🔊 Função para converter texto em áudio (Text-to-Speech)
def texto_para_audio(texto, nome_arquivo="resposta.mp3"):
    resposta = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=texto
    )

    with open(nome_arquivo, "wb") as f:
        f.write(resposta.content)  

    print("🔊 Áudio gerado:", nome_arquivo)
    
    # 🎧 Reproduzir o áudio automaticamente
    if os.name == "nt":
        os.system(f"start {nome_arquivo}")  # Windows
    else:
        os.system(f"mpg123 {nome_arquivo}")  # Linux/Mac

# 🤖 Função principal do Chatbot
def chatbot_com_voz():
    gravar_audio()
    texto_usuario = transcrever_audio()
    print("👤 Usuário:", texto_usuario)

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": texto_usuario}]
    ).choices[0].message.content

    print("🤖 Chatbot:", resposta)

    texto_para_audio(resposta)
    print("🎧 Ouvindo resposta...")

# 🚀 Executar chatbot com voz
chatbot_com_voz()
