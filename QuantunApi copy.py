# --- START OF FILE ApiChat.py ---

import os
import pickle
import sqlite3
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware

# --- Modelos de Dados Pydantic para a API ---
# Correspondem exatamente ao que o JavaScript envia/espera.

class ChatRequest(BaseModel):
    message: str
    model_name: str

class ContextRequest(BaseModel):
    context: str | None = None

class HealthCheckResponse(BaseModel):
    status: str

# --- Lógica do Cérebro do Chatbot (QuantumCerebro) ---
# Esta classe é a mesma, pois sua lógica interna é robusta.

class QuantumCerebro:
    def __init__(self, model_name: str, st_model: SentenceTransformer):
        self.model_name = model_name
        self.db_path = f'Neuron_{self.model_name}.db'
        self.embeddings_path = f'{self.model_name}_embeddings.pkl'
        self.st_model = st_model
        self.known_answers = []
        self.known_embeddings = None
        self.carregar_base_conhecimento()

    def carregar_base_conhecimento(self):
        print(f"[{self.model_name}] Carregando base de conhecimento...")
        if not os.path.exists(self.db_path):
            print(f"[{self.model_name}] Banco de dados não encontrado. O modelo estará vazio.")
            return

        conexao = sqlite3.connect(self.db_path)
        cursor = conexao.cursor()
        # Ordenar por ID para garantir consistência com os embeddings salvos
        cursor.execute("SELECT resposta FROM neuronios ORDER BY id ASC")
        self.known_answers = [item[0] for item in cursor.fetchall()]
        conexao.close()
        
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            
            if len(self.known_answers) != len(self.known_embeddings):
                print(f"[{self.model_name}] ALERTA: Inconsistência detectada entre respostas ({len(self.known_answers)}) e embeddings ({len(self.known_embeddings)}).")
                self.known_embeddings = None # Invalida para evitar erros
        else:
            print(f"[{self.model_name}] Arquivo de embeddings não encontrado. Este modelo não poderá responder.")

        print(f"[{self.model_name}] {len(self.known_answers)} neurônios carregados.")

    def processar_pergunta(self, pergunta_usuario: str, contexto_chat: str | None) -> str:
        if self.known_embeddings is None or len(self.known_embeddings) == 0:
            return "Desculpe, minha base de conhecimento para este modelo está vazia ou corrompida. Por favor, contate o administrador."

        pergunta_com_contexto = f"{contexto_chat} {pergunta_usuario}" if contexto_chat else pergunta_usuario
        query_embedding = self.st_model.encode(pergunta_com_contexto, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.known_embeddings)[0]
        best_match_idx = torch.argmax(cos_scores).item()
        
        SIMILARITY_THRESHOLD = 0.68
        if cos_scores[best_match_idx] > SIMILARITY_THRESHOLD:
            return self.known_answers[best_match_idx]
        else:
            return "Não tenho certeza de como responder a isso. Pode tentar reformular a pergunta?"

# --- Gerenciador Global de Modelos e Ciclo de Vida da API ---

chatbot_manager: Dict[str, QuantumCerebro] = {}
# Contexto global simplificado. Em uma aplicação real, isso seria gerenciado por sessão/usuário.
global_context: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código executado na inicialização da API
    print("Iniciando a API do Chatbot Luna...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Procura por modelos baseados nos arquivos de embeddings, que são o "coração" do conhecimento.
    model_names = {f.replace("_embeddings.pkl", "") for f in os.listdir('.') if f.endswith("_embeddings.pkl")}
    
    if not model_names:
        print("Nenhum modelo (*_embeddings.pkl) encontrado. A API funcionará, mas não terá modelos para selecionar.")
    
    for name in sorted(list(model_names)): # Ordena para consistência
        print(f"Encontrado e carregando modelo: {name}")
        chatbot_manager[name] = QuantumCerebro(model_name=name, st_model=st_model)
    
    yield
    
    print("Encerrando a API do Chatbot.")
    chatbot_manager.clear()

# --- Instância e Configuração do FastAPI ---

app = FastAPI(
    lifespan=lifespan,
    title="Luna AI API",
    description="API para o chatbot Luna, com entendimento semântico.",
    version="1.0.0"
)

# Habilita o CORS para permitir que a página HTML se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem. Para produção, restrinja ao seu domínio.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints da API ---
# Os caminhos correspondem exatamente ao que o JavaScript da sua página está chamando.

@app.get("/health", response_model=HealthCheckResponse, tags=["Status"])
async def health_check():
    """Verifica se a API está online."""
    return {"status": "ok"}

@app.get("/models", response_model=List[str], tags=["Modelos"])
async def get_available_models():
    """Retorna uma lista com os nomes de todos os modelos carregados."""
    if not chatbot_manager:
        return []
    return list(chatbot_manager.keys())

@app.get("/context", tags=["Contexto"])
async def get_current_context():
    """Retorna o contexto de sessão atual."""
    return {"context": global_context}

@app.post("/context", tags=["Contexto"])
async def set_session_context(request: ContextRequest):
    """Define um novo contexto de sessão."""
    global global_context
    global_context = request.context
    return {"status": "success", "context": global_context}

@app.post("/chat", tags=["Chat"])
async def chat_with_luna(request: ChatRequest):
    """Endpoint principal para conversação."""
    if request.model_name not in chatbot_manager:
        raise HTTPException(status_code=404, detail=f"Modelo '{request.model_name}' não encontrado.")
    
    brain = chatbot_manager[request.model_name]
    answer = brain.processar_pergunta(request.message, global_context)
    
    return {"response": answer}

# --- Execução do Servidor ---

if __name__ == "__main__":
    import uvicorn
    print("\nServidor da API da Luna iniciado.")
    print("Para testar os endpoints, acesse: http://127.0.0.1:8000/docs")
    print("Abra o arquivo index.html em seu navegador para usar a interface gráfica.")
    uvicorn.run(app, host="127.0.0.1", port=8000)

# --- END OF FILE ApiChat.py ---