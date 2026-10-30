import os
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client

# Load variabel dari file .env
load_dotenv()

DATA_DIR = Path('data')
DB_FILE = Path('rag.db')

# Ambil konfigurasi dari .env (berikan nilai default jika .env tidak ditemukan)
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'bge-m3')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'llama3.2')

# Inisialisasi client Ollama dengan host kustom
ollama_client = Client(host=OLLAMA_HOST)

DOC_PREFIX = ''
QUERY_PREFIX = ''

# Pengaturan indexing & chat
MAX_CHARS = 1500
BATCH_SIZE = 16
TOP_K = 5
MIN_SCORE = 0.5
SHOW_DEBUG = False