# --- START OF FILE QuantumChat.py ---

import sqlite3
import pandas as pd
import numpy as np
import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

class QuantumCerebro:
    """
    Uma abordagem completa para um chatbot, utilizando Sentence Transformers para
    entendimento semântico real. Isso resulta em uma acurácia muito maior para
    perguntas similares e elimina o complexo ciclo de treinamento/overfitting.
    Tudo contido em uma única classe para simplicidade.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.db_path = f'Neuron_{self.model_name}.db'
        self.embeddings_path = f'{self.model_name}_embeddings.pkl'
        self.contexto_atual = None

        # Carrega um modelo de ponta, otimizado para busca semântica.
        # Ele será baixado automaticamente na primeira execução.
        print("Carregando o modelo de linguagem (pode levar um momento na primeira vez)...")
        # Usamos 'cpu' para garantir compatibilidade em sistemas sem GPU da Nvidia.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {device}")
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Modelo de linguagem carregado.")

        # Armazenamento de conhecimento em memória
        self.known_questions = []
        self.known_answers = []
        self.known_embeddings = None

        self.conexao = sqlite3.connect(self.db_path)
        self.cursor = self.conexao.cursor()
        self.criar_tabelas()

        # Carrega a base de conhecimento (perguntas, respostas e embeddings)
        self.carregar_base_conhecimento()

    def criar_tabelas(self):
        """Cria a tabela no banco de dados se ela não existir."""
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS neuronios (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                pergunta TEXT UNIQUE,
                                resposta TEXT,
                                contexto TEXT
                             )''')
        self.conexao.commit()

    def carregar_base_conhecimento(self):
        """
        Carrega todas as perguntas/respostas do DB e seus embeddings do arquivo.
        Se o arquivo de embeddings não existir, ele é criado.
        """
        self.cursor.execute("SELECT pergunta, resposta, contexto FROM neuronios")
        neuronios = self.cursor.fetchall()
        
        self.known_questions = []
        self.known_answers = []
        
        for pergunta, resposta, contexto in neuronios:
            pergunta_com_contexto = f"{contexto} {pergunta}" if contexto else pergunta
            self.known_questions.append(pergunta_com_contexto)
            self.known_answers.append(resposta)

        if os.path.exists(self.embeddings_path):
            print(f"Carregando base de conhecimento de '{self.embeddings_path}'...")
            with open(self.embeddings_path, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            if len(self.known_questions) != len(self.known_embeddings):
                print("Inconsistência detectada! Reconstruindo a base de conhecimento...")
                self.reconstruir_embeddings()
        else:
            print("Nenhuma base de conhecimento encontrada. Construindo do zero...")
            self.reconstruir_embeddings()
        
        print(f"{len(self.known_questions)} neurônios carregados na memória.")

    def reconstruir_embeddings(self):
        """
        (Re)cria os embeddings para todas as perguntas conhecidas e os salva.
        Este é o processo de "treinamento" desta arquitetura.
        """
        if not self.known_questions:
            print("Nenhum dado para indexar.")
            self.known_embeddings = None
            if os.path.exists(self.embeddings_path):
                os.remove(self.embeddings_path)
            return

        print(f"Indexando {len(self.known_questions)} perguntas... (Isso pode demorar um pouco)")
        self.known_embeddings = self.st_model.encode(self.known_questions, convert_to_tensor=True, show_progress_bar=True)
        
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.known_embeddings, f)
        
        print("Indexação concluída e salva.")

    def adicionar_neuronio(self, pergunta, resposta, contexto=None):
        """Adiciona um novo neurônio ao DB."""
        try:
            self.cursor.execute("INSERT INTO neuronios (pergunta, resposta, contexto) VALUES (?, ?, ?)",
                                (pergunta, resposta, contexto))
            self.conexao.commit()
            print(f"Aprendi: '{pergunta}'")
            return True
        except sqlite3.IntegrityError:
            return False

    def carregar_csv(self, arquivo):
        """Carrega um arquivo CSV, remove duplicatas e adiciona ao DB."""
        try:
            df = pd.read_csv(arquivo)
            df.drop_duplicates(subset=['pergunta'], keep='first', inplace=True)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {arquivo}")
            return

        if 'pergunta' not in df.columns or 'resposta' not in df.columns:
            print("O arquivo CSV deve conter as colunas 'pergunta' e 'resposta'.")
            return

        if 'contexto' not in df.columns:
            df['contexto'] = ''
        df.fillna({'contexto': ''}, inplace=True)

        novos_adicionados = 0
        for _, row in df.iterrows():
            if self.adicionar_neuronio(row['pergunta'], row['resposta'], row['contexto'] or None):
                novos_adicionados += 1
        
        if novos_adicionados > 0:
            print(f"\n{novos_adicionados} novos neurônios adicionados. Reconstruindo a base de conhecimento...")
            self.carregar_base_conhecimento()
        else:
            print("Nenhum neurônio novo foi adicionado (possivelmente já existiam).")

    def processar_pergunta(self, pergunta_usuario):
        """Processa uma pergunta usando busca por similaridade semântica."""
        if self.known_embeddings is None or len(self.known_embeddings) == 0:
            return "Minha base de conhecimento está vazia. Por favor, me ensine algo usando o comando /csv."

        pergunta_com_contexto = f"{self.contexto_atual} {pergunta_usuario}" if self.contexto_atual else pergunta_usuario

        query_embedding = self.st_model.encode(pergunta_com_contexto, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.known_embeddings)[0]
        best_match_idx = torch.argmax(cos_scores).item()
        
        SIMILARITY_THRESHOLD = 0.65

        if cos_scores[best_match_idx] > SIMILARITY_THRESHOLD:
            return self.known_answers[best_match_idx]
        else:
            return "Não tenho certeza de como responder a isso. Pode reformular a pergunta?"

    def mudar_contexto(self, novo_contexto):
        """Altera o contexto atual."""
        self.contexto_atual = novo_contexto if novo_contexto.lower() != 'none' else None
        print(f"Contexto definido para: {self.contexto_atual}")

    def fechar(self):
        """Fecha a conexão com o banco de dados."""
        self.conexao.close()
        print("Conexão com o banco de dados fechada.")

def main():
    """Função principal que executa a interface de linha de comando."""
    print("=== Sistema de Chatbot com Entendimento Semântico (Arquivo Único) ===")
    print("Comandos:")
    print("/csv [arquivo]  - Carrega e indexa dados de um arquivo CSV.")
    print("/contexto [nome] - Define um contexto para as perguntas.")
    print("/reindexar - Força a reconstrução da base de conhecimento a partir do DB.")
    print("/sair - Encerra o programa.\n")
    
    modelo_nome = input("Nome do modelo para carregar/criar: ").strip() or "chatbot"
    try:
        cerebro = QuantumCerebro(modelo_nome)
    except Exception as e:
        print(f"\nErro fatal ao inicializar o cérebro: {e}")
        print("Verifique sua conexão com a internet para baixar o modelo na primeira vez.")
        return
        
    while True:
        try:
            entrada = input("\nVocê: ").strip()
            
            if not entrada:
                continue

            if entrada.lower() == '/sair':
                break
            
            elif entrada.startswith('/csv '):
                partes = entrada.split(' ', 1)
                if len(partes) > 1:
                    arquivo = partes[1]
                    cerebro.carregar_csv(arquivo)
                else:
                    print("Formato inválido! Use: /csv [caminho_do_arquivo.csv]")
                continue
                
            elif entrada.startswith('/contexto '):
                novo_contexto = entrada.split(' ', 1)[-1]
                cerebro.mudar_contexto(novo_contexto)
                continue
            
            elif entrada.lower() == '/reindexar':
                cerebro.reconstruir_embeddings()
                continue
            
            resposta = cerebro.processar_pergunta(entrada)
            print(f"Sistema: {resposta}")
            
            if "não tenho certeza" in resposta.lower():
                nova_resposta = input("Como eu deveria ter respondido? (deixe em branco para ignorar) ").strip()
                if nova_resposta:
                    novo_contexto = input("Em qual contexto essa pergunta se encaixa? (opcional) ").strip() or None
                    if cerebro.adicionar_neuronio(entrada, nova_resposta, novo_contexto):
                        print("Obrigado! Reconstruindo minha base de conhecimento com essa nova informação...")
                        cerebro.reconstruir_embeddings()
                    else:
                        print("Essa pergunta já existe. Não adicionei novamente.")

        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")

    cerebro.fechar()
    print("Sistema encerrado!")

if __name__ == "__main__":
    main()

# --- END OF FILE QuantumChat.py ---