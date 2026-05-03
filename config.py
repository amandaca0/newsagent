import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

BLUEBUBBLES_SERVER_URL = os.getenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
BLUEBUBBLES_PASSWORD = os.getenv("BLUEBUBBLES_PASSWORD", "")

DB_PATH = os.getenv("NEWSAGENT_DB_PATH", "./data/newsagent.db")
CHROMA_PATH = os.getenv("NEWSAGENT_CHROMA_PATH", "./data/chroma")
PUBLIC_URL = os.getenv("NEWSAGENT_PUBLIC_URL", "http://localhost:5000")

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PROACTIVE_PUSH_HOUR = int(os.getenv("PROACTIVE_PUSH_HOUR", "8"))
MAX_ARTICLES_PER_PUSH = int(os.getenv("MAX_ARTICLES_PER_PUSH", "3"))
CONVERSATION_HISTORY_TURNS = 10
