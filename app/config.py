import os
from functools import lru_cache
from google.cloud import firestore
from google.cloud import secretmanager
import vertexai
from vertexai.generative_models import GenerativeModel
from google import genai

# ==========================================
# 1. Non-Sensitive Infrastructure Configs
# ==========================================
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("PROJECT_ID", "career-copilot-506013"))
LOCATION = os.getenv("DOCAI_LOCATION", os.getenv("LOCATION", "us"))
VERTEX_REGION = os.getenv("VERTEX_REGION", os.getenv("REGION", "us-central1"))
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID", "")

# ==========================================
# 2. Secret Manager Dynamic Retriever
# ==========================================
@lru_cache(maxsize=16)
def get_secret(secret_id: str, version_id: str = "latest") -> str:
    """
    Fetches secret from Secret Manager or environment variables.
    Cached in memory to prevent repeated network hops.
    """
    env_name = secret_id.upper().replace("-", "_")
    env_val = os.getenv(env_name)
    if env_val:
        return env_val

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception as e:
        print(f"⚠️ Warning: Could not retrieve secret '{secret_id}' from Secret Manager: {e}")
        return ""

# ==========================================
# 3. Dynamic Credential Retrieval
# ==========================================
# Fetches from Secret Manager (falls back to local env variable if running offline)
GEMINI_API_KEY = get_secret("gemini-api-key")

# ==========================================
# 4. Service Clients Initialization
# ==========================================
# Initialize shared Firestore Client using Application Default Credentials
db = firestore.Client(project=PROJECT_ID)

# Initialize Google GenAI Client with the retrieved key
if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("\n⚠️ WARNING: GEMINI_API_KEY not found in Secret Manager or Environment.\n")
    # Fallback to Application Default Credentials
    ai_client = genai.Client()


# Initialize Vertex AI SDK
try:
    vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)
    llm_model = GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print(f"Vertex AI initialization notice: {e}")
    llm_model = None