from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.ingestion.routes import router as ingestion_router
from app.diagnostic.routes import router as diagnostic_router
from app.rag.routes import router as rag_router


app = FastAPI(
    title="Career Copilot Modular Services",
    description="Unified API combining Ingestion and Dynamic Diagnostic Assessment",
    version="1.0.0"
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local Cloud Shell testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router)
app.include_router(diagnostic_router)
app.include_router(rag_router)


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "modules": ["ingestion", "diagnostic"]}