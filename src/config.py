"""
Configuration and constants for the Rebalance Vintage scraper.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv(
    "SUPABASE_URL",
    "https://yqawmzggcgpeyaaynrjk.supabase.co",
)
SUPABASE_KEY: str = os.getenv(
    "SUPABASE_KEY",
    "",
)

# ── Store ─────────────────────────────────────────────────────────────────────
STORE_DOMAIN = "rebalancevintage.com"
STORE_BASE_URL = f"https://{STORE_DOMAIN}"
SOURCE = "scraper-rebalance"
BRAND = "Rebalance"
SECOND_HAND = False
DEFAULT_COUNTRY = "CZ"
CURRENCY = "CZK"

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORY_MAP: dict[str, str] = {
    "tees": "T-Shirts",
    "sweaters": "Sweaters",
    "long-sleeves": "Long Sleeves",
    "jackets": "Jackets",
    "pants-shorts": "Pants, Shorts",
    "shorts-1": "Shorts",
    "jerseys-1": "Jerseys",
    "fleeces": "Fleeces",
    "hats": "Hats",
    "rebalance-collective-1": "Rebalance Collective",
    "sale": "Sale",
}

# URLs to scrape – each entry is (handle, display_name)
CATEGORY_URLS: list[tuple[str, str]] = [
    ("tees", "T-Shirts"),
    ("sweaters", "Sweaters"),
    ("long-sleeves", "Long Sleeves"),
    ("jackets", "Jackets"),
    ("pants-shorts", "Pants, Shorts"),
    ("shorts-1", "Shorts"),
    ("jerseys-1", "Jerseys"),
    ("fleeces", "Fleeces"),
    ("hats", "Hats"),
    ("rebalance-collective-1", "Rebalance Collective"),
    ("sale", "Sale"),
]

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
TORCH_DEVICE: Optional[str] = os.getenv("TORCH_DEVICE")
if TORCH_DEVICE is None:
    import torch
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ── Scraping ──────────────────────────────────────────────────────────────────
SCRAPE_ALL: bool = os.getenv("SCRAPE_ALL", "true").lower() == "true"
HTTP_CONCURRENCY: int = int(os.getenv("HTTP_CONCURRENCY", "10"))
REQUEST_DELAY: float = float(os.getenv("REQUEST_DELAY", "0.5"))
SHOPIFY_PAGE_LIMIT = 250  # max per page Shopify allows

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH: Path = Path(os.getenv("CHECKPOINT_PATH", str(DATA_DIR / "checkpoint.json")))
OUTPUT_PATH: Path = Path(os.getenv("OUTPUT_PATH", str(DATA_DIR / "products.json")))

# ── Database ──────────────────────────────────────────────────────────────────
DB_BATCH_SIZE: int = int(os.getenv("DB_BATCH_SIZE", "50"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# ── Table ─────────────────────────────────────────────────────────────────────
DB_TABLE = "products"
