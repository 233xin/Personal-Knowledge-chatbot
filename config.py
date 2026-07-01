"""
Central configuration for the production-grade knowledge base system.
All settings, environment variables, model names, and thresholds live here.
"""

import os
from pathlib import Path

# --- Project Root ---
ROOT_DIR = Path(__file__).parent

# --- Storage Paths ---
CHROMA_DIR = ROOT_DIR / "storage" / "chroma_db"
BM25_DIR = ROOT_DIR / "storage" / "bm25"
UPLOAD_DIR = ROOT_DIR / "uploads"
CHROMA_COLLECTION_NAME = "knowledge_base"
BM25_INDEX_FILE = "bm25_corpus.pkl"

# --- API Keys ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# --- DashScope / Bailian ---
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_OCR_MODEL = os.getenv("BAILIAN_OCR_MODEL", "qwen-vl-ocr")
DASHSCOPE_MARKDOWN_MODEL = "qwen-plus"       # For markdown extraction from docx
DASHSCOPE_VISION_MAX_TOKENS = 4000
DASHSCOPE_PAGE_IMAGE_DPI = 200               # DPI for rendering PDF pages to images

# --- DeepSeek (Q&A Generation) ---
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.3
DEEPSEEK_MAX_TOKENS = 2000

# --- Embedding Model ---
# HuggingFace mirror for users behind GFW
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_MODEL_KWARGS = {"trust_remote_code": True}

# --- Parser Router Thresholds ---
PDF_TEXT_THRESHOLD = 200        # avg chars/page below which → scanned/image-based
LONG_DOC_THRESHOLD = 50         # pages above which → LlamaParse
PDF_SAMPLE_PAGES = 3            # number of pages to sample for classification
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

# --- Chunking ---
CHUNK_SIZE = 2000               # max characters per chunk (heading-aware)
MIN_CHUNK_SIZE = 200            # minimum chunk size before merging with neighbor
CHUNK_OVERLAP = 0               # No overlap — heading boundaries are exact
HEADING_PATTERN = r"^#{1,6}\s+(.+)$"

# --- Agent Retrieval ---
AGENT_MAX_ITERATIONS = 5        # prevent infinite tool-calling loops
RETRIEVAL_DEFAULT_TOP_K = 5

# --- BM25 ---
BM25_K1 = 1.5
BM25_B = 0.75

# --- Supported Extensions ---
MARKER_EXTENSIONS = {".pdf", ".docx"}
LLAMAPARSE_EXTENSIONS = {".pdf"}
DASHSCOPE_OCR_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".xml",
    ".html", ".htm", ".py", ".js", ".ts", ".yaml", ".yml", ".log",
}
DOC_EXTENSIONS = {".doc"}

# --- LlamaParse ---
LLAMAPARSE_RESULT_TYPE = "markdown"
LLAMAPARSE_MAX_PAGES = int(os.getenv("LLAMAPARSE_MAX_PAGES", "200"))
