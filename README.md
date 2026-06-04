# 🛍️ Rebalance Vintage Scraper

Full-featured scraper for [rebalancevintage.com](https://rebalancevintage.com) — a Shopify-based vintage clothing store.

## Features

- **Scrapes all categories** with full pagination (walks pages until empty)
- **Extracts full product data** via Shopify's built-in JSON API
- **Generates 768-dim image embeddings** using `google/siglip-base-patch16-384`
- **Generates 768-dim text embeddings** from product info (title, description, price, category, etc.)
- **Uploads to Supabase** with the `products` table schema
- **Checkpointing & resumability** — pick up where you left off
- **Concurrent HTTP** for fast scraping
- **Batch processing** for efficient ML inference

## Requirements

- Python 3.10+
- PyTorch (see [pytorch.org](https://pytorch.org) for platform-specific install)
- ~2GB RAM for the SIGLIP model (CPU mode) or GPU with CUDA/MPS

## Quick Start

```bash
# 1. Clone & enter
cd scraper-rebalance

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install PyTorch first (see https://pytorch.org)
# CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.x:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env if needed (default Supabase credentials are already set)

# 6. Run full pipeline
python -m src.main
```

## Usage

```bash
# Full pipeline: scrape → embed → upload
python -m src.main

# Selective steps:
python -m src.main --scrape          # Only scrape products
python -m src.main --embed           # Only generate embeddings (from saved data)
python -m src.main --upload          # Only upload to Supabase

# Resume partial scrape (loads existing data, adds new)
python -m src.main --resume

# Force re-scrape from scratch
python -m src.main --scrape --from-scratch

# Full pipeline without embedding (scrape + upload only)
python -m src.main --skip-embed
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SUPABASE_URL` | Supabase project URL | (set in .env) |
| `SUPABASE_KEY` | Supabase service role key | (set in .env) |
| `TORCH_DEVICE` | `cpu`, `cuda`, or `mps` | auto-detected |
| `SCRAPE_ALL` | Scrape all or only missing | `true` |
| `HTTP_CONCURRENCY` | Concurrent HTTP requests | `10` |
| `REQUEST_DELAY` | Delay between page requests (s) | `0.5` |
| `DB_BATCH_SIZE` | Batch size for DB upserts | `50` |
| `EMBEDDING_BATCH_SIZE` | Batch size for model inference | `16` |

## Data Flow

```
Shopify JSON API
       │
       ▼
   scraper.py  ────►  data/products.json  ◄── checkpoint.json
       │
       ▼
  embeddings.py  ──►  (image_embedding + info_embedding filled)
       │
       ▼
  database.py  ────►  Supabase "products" table
```

## Category Mapping

| URL Handle | Database Category |
|---|---|
| `tees` | T-Shirts |
| `sweaters` | Sweaters |
| `long-sleeves` | Long Sleeves |
| `jackets` | Jackets |
| `pants-shorts` | Pants, Shorts |
| `shorts-1` | Shorts |
| `jerseys-1` | Jerseys |
| `fleeces` | Fleeces |
| `hats` | Hats |
| `rebalance-collective-1` | Rebalance Collective |
| `sale` | Sale |

Products appearing in multiple categories (e.g., a tee on sale) get merged categories.
