# --- START OF FILE ChatOnly.py ---

import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

def listar_modelos_disponiveis():
    """
    Verifica a pasta atual em busca de bases de conhecimento existentes
    (arquivos de embeddings .pkl) e retorna uma lista de nomes de modelos.
    """
    modelos = set()
    for file in os.listdir('.'):
        if file.endswith("_embeddings.pkl"):
            # Extrai o nome do modelo do nome do arquivo
            # Ex: "chatbot_embeddings.pkl" -> "chatbot"
            model_name = file.replace("_embeddings.pkl", "")
            modelos.add(model_name)
    return list(modelos)

class ChatbotInferencia:
    """
    Uma versão simplificada do cérebro, focada exclusivamente em carregar um
    modelo existente e responder perguntas. Não possui funcionalidades de
    aprendizado ou modificação da base de dados.
    """
    def __init__(self):
        self.model_name = None
        self.contexto_atual = None
        self.st_model = None
        self.known_answers = []
        self.known_embeddings = None

    def carregar_modelo_linguagem(self):
        """Carrega o modelo Sentence Transformer. Acontece apenas uma vez."""
        if self.st_model is None:
            print("Carregando o modelo de linguagem (pode levar um momento na primeira vez)...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Usando dispositivo: {device}")
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print("Modelo de linguagem carregado.")

    def carregar_base_conhecimento(self, model_name):
        """Carrega uma base de conhecimento específica (perguntas e embeddings)."""
        self.model_name = model_name
        embeddings_path = f'{self.model_name}_embeddings.pkl'
        db_path = f'Neuron_{self.model_name}.db'

        if not os.path.exists(embeddings_path) or not os.path.exists(db_path):
            print(f"Erro: Arquivos de conhecimento para o modelo '{self.model_name}' não encontrados.")
            print(f"Verifique se '{embeddings_path}' e '{db_path}' existem.")
            self.known_embeddings = None
            self.known_answers = []
            return False

        try:
            print(f"Carregando base de conhecimento para o modelo '{self.model_name}'...")
            # Carrega os vetores de embeddings
            with open(embeddings_path, 'rb') as f:
                self.known_embeddings = pickle.load(f)

            # Carrega as respostas correspondentes do banco de dados
            import sqlite3
            conexao = sqlite3.connect(db_path)
            cursor = conexao.cursor()
            cursor.execute("SELECT resposta FROM neuronios")
            respostas_db = cursor.fetchall()
            conexao.close()
            
            # Extrai as respostas da lista de tuplas
            self.known_answers = [item[0] for item in respostas_db]

            if len(self.known_answers) != len(self.known_embeddings):
                print("Erro: Inconsistência entre o número de respostas e embeddings.")
                print("A base de conhecimento pode estar corrompida. Use o QuantumChat.py para reindexar.")
                return False
            
            print(f"Modelo '{self.model_name}' carregado com sucesso. {len(self.known_answers)} respostas prontas.")
            return True

        except Exception as e:
            print(f"Falha ao carregar a base de conhecimento para '{self.model_name}': {e}")
            return False

    def processar_pergunta(self, pergunta_usuario):
        """Processa uma pergunta usando busca por similaridade semântica."""
        if self.st_model is None or self.known_embeddings is None or len(self.known_embeddings) == 0:
            return "O chatbot não está pronto. Nenhum modelo ou base de conhecimento carregado."

        pergunta_com_contexto = f"{self.contexto_atual} {pergunta_usuario}" if self.contexto_atual else pergunta_usuario

        query_embedding = self.st_model.encode(pergunta_com_contexto, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.known_embeddings)[0]
        best_match_idx = torch.argmax(cos_scores).item()
        
        # Um limiar um pouco mais alto para o modo "somente leitura" é uma boa prática
        SIMILARITY_THRESHOLD = 0.70

        if cos_scores[best_match_idx] > SIMILARITY_THRESHOLD:
            return self.known_answers[best_match_idx]
        else:
            return "Não tenho confiança suficiente para dar uma resposta precisa sobre isso."

    def mudar_contexto(self, novo_contexto):
        """Altera o contexto atual."""
        self.contexto_atual = novo_contexto if novo_contexto.lower() != 'none' else None
        print(f"Contexto definido para: {self.contexto_atual}")

def main():
    """Função principal que executa a interface de chat somente leitura."""
    print("=== Chatbot (Modo de Conversa) ===")
    
    chatbot = ChatbotInferencia()
    modelos_disponiveis = listar_modelos_disponiveis()

    if not modelos_disponiveis:
        print("Nenhum modelo de chatbot (arquivo *_embeddings.pkl) foi encontrado.")
        print("Use o script QuantumChat.py para criar uma base de conhecimento primeiro.")
        return

    # Loop para seleção inicial do modelo
    while True:
        print("\nModelos disponíveis:")
        for i, nome in enumerate(modelos_disponiveis):
            print(f"  [{i+1}] {nome}")
        
        try:
            escolha = input("Selecione o número do modelo para carregar: ").strip()
            modelo_selecionado = modelos_disponiveis[int(escolha) - 1]
            
            # Carrega o modelo de linguagem (pesado) apenas uma vez
            if chatbot.st_model is None:
                chatbot.carregar_modelo_linguagem()

            # Carrega a base de conhecimento específica
            if chatbot.carregar_base_conhecimento(modelo_selecionado):
                break  # Sai do loop se o carregamento for bem-sucedido
        except (ValueError, IndexError):
            print("Seleção inválida. Por favor, digite um número da lista.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            return
    
    print("\nComandos disponíveis durante o chat:")
    print("/contexto [nome] - Define um contexto para as perguntas.")
    print("/trocar - Permite selecionar outro modelo.")
    print("/sair - Encerra o programa.")

    # Loop principal de chat
    while True:
        try:
            entrada = input("\nVocê: ").strip()
            if not entrada:
                continue

            if entrada.lower() == '/sair':
                break
            
            elif entrada.startswith('/contexto '):
                novo_contexto = entrada.split(' ', 1)[-1]
                chatbot.mudar_contexto(novo_contexto)
                continue
            
            elif entrada.lower() == '/trocar':
                # Reinicia o loop de seleção de modelo
                main()
                return # Encerra a instância atual para começar uma nova
            
            resposta = chatbot.processar_pergunta(entrada)
            print(f"Sistema: {resposta}")

        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")

    print("Chat encerrado!")

if __name__ == "__main__":
    main()