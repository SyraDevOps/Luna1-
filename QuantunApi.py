# --- START OF FILE ApiChat.py ---

import os
import pickle
import sqlite3
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware

# --- Modelos de Dados Pydantic para a API ---

class ChatRequest(BaseModel):
    message: str
    model_name: str

class ContextRequest(BaseModel):
    context: str | None = None

class HealthCheckResponse(BaseModel):
    status: str

# --- Lógica do Cérebro do Chatbot (QuantumCerebro) ---
# Adicionamos uma função para adicionar conhecimento e reindexar

class QuantumCerebro:
    def __init__(self, model_name: str, st_model: SentenceTransformer):
        self.model_name = model_name
        self.db_path = f'Neuron_{self.model_name}.db'
        self.embeddings_path = f'{self.model_name}_embeddings.pkl'
        self.st_model = st_model
        self.known_answers = []
        self.known_embeddings = None
        self._criar_tabelas_se_nao_existir()
        self.carregar_base_conhecimento()

    def _criar_tabelas_se_nao_existir(self):
        conexao = sqlite3.connect(self.db_path)
        cursor = conexao.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS neuronios (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           pergunta TEXT UNIQUE,
                           resposta TEXT,
                           contexto TEXT
                        )''')
        conexao.commit()
        conexao.close()

    def carregar_base_conhecimento(self):
        print(f"[{self.model_name}] Carregando base de conhecimento...")
        conexao = sqlite3.connect(self.db_path)
        cursor = conexao.cursor()
        cursor.execute("SELECT pergunta, resposta, contexto FROM neuronios ORDER BY id ASC")
        neuronios = cursor.fetchall()
        conexao.close()
        
        self.known_questions_raw = [p for p, a, ctx in neuronios]
        self.known_questions_context = [f"{ctx} {p}" if ctx else p for p, a, ctx in neuronios]
        self.known_answers = [a for p, a, ctx in neuronios]

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            if len(self.known_answers) != len(self.known_embeddings):
                print(f"[{self.model_name}] Inconsistência detectada! Reindexando...")
                self.reconstruir_embeddings()
        else:
            self.reconstruir_embeddings()
        print(f"[{self.model_name}] {len(self.known_answers)} neurônios carregados.")

    def reconstruir_embeddings(self):
        if not self.known_questions_context:
            self.known_embeddings = None
            if os.path.exists(self.embeddings_path):
                os.remove(self.embeddings_path)
            return

        print(f"[{self.model_name}] Reindexando {len(self.known_questions_context)} perguntas...")
        self.known_embeddings = self.st_model.encode(self.known_questions_context, convert_to_tensor=True, show_progress_bar=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.known_embeddings, f)
        print(f"[{self.model_name}] Indexação concluída.")

    def adicionar_neuronio(self, pergunta: str, resposta: str, contexto: str | None) -> bool:
        conexao = sqlite3.connect(self.db_path)
        cursor = conexao.cursor()
        try:
            cursor.execute("INSERT INTO neuronios (pergunta, resposta, contexto) VALUES (?, ?, ?)", (pergunta, resposta, contexto))
            conexao.commit()
            print(f"[{self.model_name}] Novo conhecimento adicionado: '{pergunta}'")
            # Após adicionar, recarrega e reindexa
            self.carregar_base_conhecimento()
            return True
        except sqlite3.IntegrityError:
            return False # Pergunta já existe
        finally:
            conexao.close()

    def processar_pergunta(self, pergunta_usuario: str, contexto_chat: str | None) -> str:
        if self.known_embeddings is None or len(self.known_embeddings) == 0:
            return "Minha base de conhecimento para este modelo está vazia ou corrompida."

        pergunta_com_contexto = f"{contexto_chat} {pergunta_usuario}" if contexto_chat else pergunta_usuario
        query_embedding = self.st_model.encode(pergunta_com_contexto, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.known_embeddings)[0]
        best_match_idx = torch.argmax(cos_scores).item()
        
        SIMILARITY_THRESHOLD = 0.70
        if cos_scores[best_match_idx] > SIMILARITY_THRESHOLD:
            return self.known_answers[best_match_idx]
        else:
            # Retorna um código especial para indicar baixa confiança
            return "LOW_CONFIDENCE_RESPONSE"

# --- Gerenciador Global e Ciclo de Vida da API ---

chatbot_manager: Dict[str, QuantumCerebro] = {}
global_context: str | None = None
# MUDANÇA AQUI: Variáveis para gerenciar o estado do modo root
is_root_mode_active: bool = False
last_unanswered_question: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando a API do Chatbot Luna...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    app.state.st_model = st_model # Disponibiliza para a app
    
    model_names = {f.replace("_embeddings.pkl", "") for f in os.listdir('.') if f.endswith("_embeddings.pkl")}
    for name in sorted(list(model_names)):
        print(f"Encontrado e carregando modelo: {name}")
        chatbot_manager[name] = QuantumCerebro(model_name=name, st_model=st_model)
    yield
    print("Encerrando a API do Chatbot.")
    chatbot_manager.clear()

# --- Instância e Configuração do FastAPI ---

app = FastAPI(lifespan=lifespan, title="Luna AI API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Endpoints da API ---

@app.get("/health", response_model=HealthCheckResponse, tags=["Status"])
async def health_check():
    return {"status": "ok"}

@app.get("/models", response_model=List[str], tags=["Modelos"])
async def get_available_models():
    return list(chatbot_manager.keys())

@app.get("/context", tags=["Contexto"])
async def get_current_context():
    return {"context": global_context}

@app.post("/context", tags=["Contexto"])
async def set_session_context(request: ContextRequest):
    global global_context
    global_context = request.context
    return {"status": "success", "context": global_context}

@app.post("/chat", tags=["Chat"])
async def chat_with_luna(request: ChatRequest):
    """
    Endpoint principal para conversação, agora com lógica de modo root integrada.
    """
    global is_root_mode_active, last_unanswered_question

    # --- Lógica do Modo Root ---
    ROOT_PASSWORD = "TCPSMYA"
    
    # Ativar/Desativar modo root
    if request.message.strip() == ROOT_PASSWORD:
        is_root_mode_active = not is_root_mode_active
        last_unanswered_question = None # Limpa a pergunta pendente ao trocar de modo
        status_message = "Modo Root ativado. Agora posso aprender novas respostas." if is_root_mode_active else "Modo Root desativado."
        return {"response": status_message}

    # Se o modo root estiver ativo e houver uma pergunta pendente, esta mensagem é a resposta.
    if is_root_mode_active and last_unanswered_question:
        brain = chatbot_manager.get(request.model_name)
        if not brain:
             raise HTTPException(status_code=404, detail=f"Modelo '{request.model_name}' não encontrado.")

        new_answer = request.message
        question_to_learn = last_unanswered_question
        
        # Adiciona o novo neurônio
        brain.adicionar_neuronio(pergunta=question_to_learn, resposta=new_answer, contexto=global_context)
        
        # Limpa o estado
        last_unanswered_question = None
        
        return {"response": f"Entendido! Aprendi que para '{question_to_learn}', a resposta é '{new_answer}'. A base de dados foi atualizada."}

    # --- Lógica de Chat Normal ---
    
    if request.model_name not in chatbot_manager:
        # Se o modelo não existe e o modo root está ativo, podemos criá-lo
        if is_root_mode_active:
            print(f"Modo Root: Criando novo modelo em tempo de execução: {request.model_name}")
            chatbot_manager[request.model_name] = QuantumCerebro(model_name=request.model_name, st_model=app.state.st_model)
            return {"response": f"Modelo '{request.model_name}' criado. Faça uma pergunta para eu aprender."}
        else:
            raise HTTPException(status_code=404, detail=f"Modelo '{request.model_name}' não encontrado.")

    brain = chatbot_manager[request.model_name]
    answer = brain.processar_pergunta(request.message, global_context)

    if answer == "LOW_CONFIDENCE_RESPONSE":
        if is_root_mode_active:
            last_unanswered_question = request.message # Armazena a pergunta
            return {"response": "Não sei a resposta. Como eu deveria responder?"}
        else:
            return {"response": "Não tenho certeza de como responder a isso. Pode tentar reformular a pergunta?"}
    
    # Resposta normal, limpa qualquer pergunta pendente
    last_unanswered_question = None
    return {"response": answer}

# --- Execução do Servidor ---

if __name__ == "__main__":
    import uvicorn
    print("\nServidor da API da Luna iniciado.")
    print("Para testar os endpoints, acesse: http://127.0.0.1:8000/docs")
    print("Abra o arquivo index.html em seu navegador para usar a interface gráfica.")
    uvicorn.run(app, host="127.0.0.1", port=8000)