# Shared GCP & Firestore client setup
import os
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel
from google import genai

PROJECT_ID = os.getenv("PROJECT_ID", "career-copilot-506013")
LOCATION = os.getenv("LOCATION", "us")
VERTEX_REGION = os.getenv("REGION", "us-central1")
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6I9KWEGwmj8Tk9RfMnLvDpDJb7Ymn-RuRDV7fEk8gegYQ")

# Initialize shared Firestore Client
db = firestore.Client(project=PROJECT_ID)

# Initialize Google GenAI Client explicitly with the key
if not GEMINI_API_KEY:
    print("\n⚠️ WARNING: GEMINI_API_KEY is not set in this shell session! Export it with: export GEMINI_API_KEY='AIzaSy...'\n")
    # ai_client = None
    ai_client = genai.Client(api_key="AQ.Ab8RN6J7tD_IBIzTwK2rf-xwUhNFKgyMcXed2pS6o0yNv5usBA")
else:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Vertex AI / Gemini
try:
    vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)
    llm_model = GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Vertex AI initialization warning: {e}")
    llm_model = None


