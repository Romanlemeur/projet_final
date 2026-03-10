"""Point d'entrée de l'API FastAPI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

# Créer l'application
app = FastAPI(
    title="AI Music Transition API",
    description="API pour générer des transitions musicales intelligentes",
    version="1.0.0"
)

# Configurer CORS (pour le frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "🎵 AI Music Transition API",
        "docs": "/docs",
        "version": "1.0.0"
    }
