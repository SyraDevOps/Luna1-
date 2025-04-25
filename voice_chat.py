import os
import time
import speech_recognition as sr
import pyttsx3
from Neutron.NeuronChat import Cerebro, listar_modelos
import sys
import tempfile
import traceback
import whisper
import subprocess

# Configuração de caminho para o FFmpeg
script_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(script_dir, "ffmpeg-tools", "bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

# Aviso sobre dependências
print("Configurando Luna Voice Chat...")
print(f"Usando FFmpeg de: {ffmpeg_path}")

class LunaVoiceChat:
    def __init__(self):
        # Inicializar reconhecimento de voz
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Inicializar sintetizador de voz
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Velocidade da voz
        
        # Configurar voz feminina se disponível
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower() or any(name in voice.name.lower() for name in ["maria", "joana", "zira"]):
                self.engine.setProperty('voice', voice.id)
                break
        
        # Inicialização do modelo Luna
        self.modelo = None
        self.modelos_disponiveis = list(listar_modelos())
        
        # Controle de modo de entrada
        self.voice_mode = False  # Começar com texto é mais seguro
        
        # Verificar FFmpeg antes de carregar o Whisper
        self.ffmpeg_available = self.check_ffmpeg()
        if not self.ffmpeg_available:
            print("\n⚠️ AVISO: FFmpeg não encontrado no sistema!")
            print("O Whisper precisa do FFmpeg para funcionar.")
            print("Verifique se o FFmpeg está na pasta ffmpeg-tools/bin")
        else:
            # Carregar Whisper apenas se FFmpeg estiver disponível
            self.whisper_model = self.load_whisper_model()

    def check_ffmpeg(self):
        """Verifica se o FFmpeg está instalado e disponível no PATH"""
        try:
            # Tenta executar ffmpeg com o caminho explícito primeiro
            ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
            if os.path.exists(ffmpeg_exe):
                subprocess.run([ffmpeg_exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                print(f"FFmpeg encontrado em: {ffmpeg_exe}")
                return True
            
            # Tenta via PATH como fallback
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            print("FFmpeg encontrado no PATH do sistema")
            return True
        except FileNotFoundError:
            print("FFmpeg não encontrado")
            return False
        except Exception as e:
            print(f"Erro ao verificar FFmpeg: {e}")
            return False

    def load_whisper_model(self):
        """Carrega o modelo Whisper para reconhecimento de voz"""
        try:
            print("Carregando modelo Whisper (isso pode demorar na primeira vez)...")
            model = whisper.load_model("base")
            print("Modelo Whisper carregado com sucesso!")
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo Whisper: {e}")
            return None

    def recognize_with_google(self, audio):
        """Reconhecimento com Google como fallback"""
        try:
            texto = self.recognizer.recognize_google(audio, language='pt-BR')
            print(f"Reconhecimento Google: {texto}")
            return texto.lower()
        except Exception as e:
            print(f"Erro no reconhecimento Google: {e}")
            return None

    def speak(self, text):
        """Converte texto para fala"""
        print(f"Luna: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Captura a entrada do usuário via microfone ou teclado"""
        if not self.voice_mode:
            # Modo de texto ativado
            user_input = input("\nDigite sua mensagem: ")
            print(f"Você: {user_input}")
            return user_input.lower()
        
        # Verificar se o reconhecimento de voz é possível
        if not self.ffmpeg_available or not hasattr(self, 'whisper_model') or self.whisper_model is None:
            print("Reconhecimento de voz não disponível. Usando entrada de texto.")
            self.voice_mode = False
            return self.listen()
        
        # Modo de voz ativado
        print("\nOuvindo... - Pressione Enter para digitar")
        
        # Verifica se o usuário quer trocar para modo de texto
        if os.name == 'nt':  # Windows
            import msvcrt
            if msvcrt.kbhit():
                if msvcrt.getch() == b'\r':
                    return self.fallback_to_text()
        else:  # Unix/Linux/Mac
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                return self.fallback_to_text()
                
        # Captura áudio
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processando sua fala...")
                
                # Tenta com Google primeiro (mais rápido e mais confiável para comandos simples)
                texto = self.recognize_with_google(audio)
                if texto:
                    print(f"Você: {texto}")
                    return texto.lower()
                
                # Se o Google falhar, não tenta mais o Whisper devido ao erro com FFmpeg
                print("Não foi possível entender o áudio.")
                return self.fallback_to_text()
                    
            except sr.WaitTimeoutError:
                print("Tempo esgotado. Nenhuma fala detectada.")
                return self.fallback_to_text()
            except Exception as e:
                print(f"Erro no reconhecimento: {e}")
                return self.fallback_to_text()
    
    def fallback_to_text(self):
        """Método de fallback para entrada por texto"""
        self.speak("Mudando para modo de texto.")
        user_input = input("Digite sua mensagem: ")
        print(f"Você digitou: {user_input}")
        
        # Pergunta se quer manter modo de texto
        should_stay = input("Manter modo de texto? (s/n): ").lower()
        if should_stay.startswith('s'):
            self.voice_mode = False
        else:
            # Só volta para voz se o FFmpeg estiver disponível
            self.voice_mode = self.ffmpeg_available and hasattr(self, 'whisper_model') and self.whisper_model is not None
            if not self.voice_mode:
                print("Não foi possível voltar para modo de voz - FFmpeg não encontrado.")
            
        return user_input.lower()

    def toggle_input_mode(self):
        """Alterna entre modo de voz e texto"""
        if not self.voice_mode and (not self.ffmpeg_available or not hasattr(self, 'whisper_model') or self.whisper_model is None):
            self.speak("Modo de voz não disponível sem FFmpeg instalado.")
            return
        
        self.voice_mode = not self.voice_mode
        mode = "texto" if not self.voice_mode else "voz"
        self.speak(f"Modo de entrada alterado para {mode}.")

    def select_model(self):
        """Permite a seleção de um modelo disponível"""
        if not self.modelos_disponiveis:
            self.speak("Nenhum modelo encontrado. Por favor, treine um modelo primeiro.")
            return False
        
        print("\n=== Modelos Disponíveis ===")
        for i, modelo in enumerate(self.modelos_disponiveis):
            print(f"{i+1}. {modelo}")
        
        self.speak("Por favor, escolha o número do modelo que deseja usar.")
        
        # Opção simples para seleção numérica direta
        user_input = input(f"Digite o número do modelo (1-{len(self.modelos_disponiveis)}): ")
        try:
            select_num = int(user_input.strip())
            if 1 <= select_num <= len(self.modelos_disponiveis):
                model_name = self.modelos_disponiveis[select_num-1]
                self.modelo = Cerebro(model_name)
                self.speak(f"Modelo {model_name} selecionado.")
                return True
            else:
                self.speak("Número inválido.")
                return self.select_model()
        except Exception as e:
            self.speak(f"Erro: {str(e)}. Por favor, tente novamente.")
            return self.select_model()
    
    def set_context(self):
        """Define um novo contexto para a conversa"""
        self.speak("Qual contexto você deseja definir? Diga 'nenhum' para contexto global.")
        contexto = self.listen()
        
        if not contexto or contexto == "nenhum" or contexto == "global" or contexto == "none":
            self.modelo.contexto_atual = None
            self.speak("Contexto definido como global.")
        else:
            self.modelo.contexto_atual = contexto
            self.speak(f"Contexto definido como {contexto}.")
    
    def run(self):
        """Executa o loop principal de chat por voz"""
        print("\n=== Luna Voice Chat ===")
        print("Dicas rápidas:")
        print("- Pressione Enter durante reconhecimento para alternar para modo texto")
        print("- Diga 'mudar modo' para alternar entre voz e texto")
        print("- Diga 'sair' para encerrar o programa")
        
        self.speak("Olá, eu sou Luna, sua assistente de voz.")
        
        # Selecionar modelo
        if not self.select_model():
            return
        
        # Loop principal de conversa
        while True:
            try:
                user_input = self.listen()
                
                if not user_input:
                    continue
                
                # Comandos especiais
                if "sair" in user_input or "encerrar" in user_input:
                    self.speak("Encerrando o chat por voz. Até logo!")
                    break
                
                elif "trocar modelo" in user_input or "mudar modelo" in user_input:
                    self.select_model()
                    continue
                
                elif "definir contexto" in user_input or "mudar contexto" in user_input:
                    self.set_context()
                    continue
                
                elif "mudar modo" in user_input or "/texto" in user_input or "/voz" in user_input:
                    self.toggle_input_mode()
                    continue
                
                # Processar pergunta normal
                resposta = self.modelo.processar_pergunta(user_input)
                self.speak(resposta)
                
            except KeyboardInterrupt:
                self.speak("Recebido sinal de interrupção. Encerrando.")
                break
            except Exception as e:
                print(f"Erro: {e}")
                traceback.print_exc()
                self.speak(f"Desculpe, ocorreu um erro.")

if __name__ == "__main__":
    try:
        voice_chat = LunaVoiceChat()
        voice_chat.run()
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário.")
    except Exception as e:
        print(f"Erro fatal: {e}")
        traceback.print_exc()