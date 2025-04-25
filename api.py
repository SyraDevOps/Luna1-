from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import os
import sys

# Garantir que o diretório atual está no path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Luna components
from Neutron.NeuronChat import Cerebro, listar_modelos

app = FastAPI(
    title="Luna Chat API",
    description="API para interagir com o modelo de IA Luna",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response data
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    model_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    context: Optional[str] = None
    model: str

class ContextRequest(BaseModel):
    context: str

class ContextResponse(BaseModel):
    context: Optional[str] = None
    message: str

# Store active model instances for different users
# In a production environment, this would use proper session management
active_models = {}

# Get or create a model instance for the current user
async def get_model(model_name: Optional[str] = None):
    # For simplicity, we're using a single global model
    # In production, you would use user session IDs
    user_id = "global"
    
    if user_id not in active_models or (model_name and model_name != active_models[user_id]["name"]):
        # Load models only if needed or switching models
        available_models = listar_modelos()
        
        if not model_name:
            # Default to first available model if none specified
            if available_models:
                model_name = next(iter(available_models))
            else:
                raise HTTPException(status_code=404, detail="No models available")
        
        if model_name not in available_models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        active_models[user_id] = {
            "name": model_name,
            "model": Cerebro(model_name)
        }
    
    return active_models[user_id]["model"]

# Routes
@app.get("/api/models", response_model=List[str])
async def get_models():
    """Get list of available models"""
    models = listar_modelos()
    return list(models)

@app.get("/api/context", response_model=ContextResponse)
async def get_context(model: Cerebro = Depends(get_model)):
    """Get current context"""
    return ContextResponse(
        context=model.contexto_atual,
        message="Context retrieved successfully"
    )

@app.post("/api/context", response_model=ContextResponse)
async def set_context(request: ContextRequest, model: Cerebro = Depends(get_model)):
    """Set context for conversation"""
    model.contexto_atual = request.context if request.context != "null" else None
    context_msg = f"Context set to '{model.contexto_atual}'" if model.contexto_atual else "Context cleared"
    return ContextResponse(context=model.contexto_atual, message=context_msg)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to Luna and get a response"""
    # Load specified model or use default
    model = await get_model(request.model_name)
    
    # Set context if provided
    if request.context is not None:
        model.contexto_atual = request.context if request.context != "null" else None
    
    # Process the message
    response = model.processar_pergunta(request.message)
    
    return ChatResponse(
        response=response,
        context=model.contexto_atual,
        model=active_models["global"]["name"]
    )

# Optional: Add a simple health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Luna API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)