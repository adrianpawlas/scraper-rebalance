#!/usr/bin/env python3
"""
Rebalance Vintage Scraper — Main Entry Point

Orchestrates the full pipeline:
1. Scrape all products from Shopify (paginated categories)
2. Generate image embeddings (SIGLIP 768-dim)
3. Generate text / info embeddings (SIGLIP 768-dim)
4. Upsert everything to Supabase

Usage:
    python -m src.main              # Full pipeline: scrape → embed → upload
    python -m src.main --scrape     # Only scrape (save to disk)
    python -m src.main --embed      # Only generate embeddings (from saved data)
    python -m src.main --upload     # Only upload to Supabase (from saved data)
    python -m src.main --resume     # Resume partial scrape
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import (
    EMBEDDING_MODEL,
    OUTPUT_PATH,
    SCRAPE_ALL,
    SOURCE,
    TORCH_DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(
        description="Rebalance Vintage Scraper — Full Pipeline",
    )
    parser.add_argument(
        "--scrape", action="store_true",
        help="Only scrape products from Shopify (save to disk)",
    )
    parser.add_argument(
        "--embed", action="store_true",
        help="Only generate embeddings from saved products data",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Only upload products to Supabase",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume partial scrape (load existing data, only scrape new)",
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Force re-scrape all products even if data exists",
    )
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="Skip embedding generation in full pipeline",
    )

    args = parser.parse_args()

    # ── Determine mode ───────────────────────────────────────────────────
    mode_scrape = args.scrape
    mode_embed = args.embed
    mode_upload = args.upload
    full_pipeline = not (mode_scrape or mode_embed or mode_upload)

    print(f"""
╔══════════════════════════════════════════════╗
║     🛍️  Rebalance Vintage Scraper           ║
╠══════════════════════════════════════════════╣
║  Source:    {SOURCE:<20}   ║
║  Model:     {EMBEDDING_MODEL:<20}   ║
║  Device:    {TORCH_DEVICE:<20}   ║
║  Mode:      {"Full Pipeline" if full_pipeline else "Selective":<20}   ║
╚══════════════════════════════════════════════╝
""")

    # ── Step 1: Scrape ───────────────────────────────────────────────────
    if full_pipeline or mode_scrape:
        from src.scraper import (
            load_products_from_disk,
            save_products_to_disk,
            scrape_all_products,
        )

        should_resume = args.resume and not args.from_scratch
        if args.from_scratch:
            products = {}
            print("🧹 Starting from scratch...")
        elif should_resume:
            products = load_products_from_disk()
            print(f"🔄 Resuming with {len(products)} existing products...")
        else:
            products = {}

        import asyncio
        scraped = asyncio.run(scrape_all_products(scrape_all=not should_resume))

        # Merge with existing
        if products:
            products.update(scraped)
            save_products_to_disk(products)
        else:
            products = scraped

    # ── Step 2: Embed ────────────────────────────────────────────────────
    if (full_pipeline and not args.skip_embed) or mode_embed:
        from src.embeddings import generate_all_embeddings
        from src.scraper import load_products_from_disk, save_products_to_disk

        products = load_products_from_disk()
        if not products:
            print("❌ No products found on disk. Run --scrape first.")
            sys.exit(1)

        products = generate_all_embeddings(products)
        save_products_to_disk(products)

    # ── Step 3: Upload ───────────────────────────────────────────────────
    if (full_pipeline) or mode_upload:
        from src.database import upsert_all_products
        from src.scraper import load_products_from_disk

        products = load_products_from_disk()
        if not products:
            print("❌ No products found on disk. Nothing to upload.")
            sys.exit(1)

        upsert_all_products(products)

    # ── Filter products that have embeddings ─────────────────────────────
    if full_pipeline:
        from src.scraper import load_products_from_disk
        products = load_products_from_disk()
        if products:
            with_emb = sum(1 for p in products.values() if p["image_embedding"])
            with_info = sum(1 for p in products.values() if p["info_embedding"])
            print(f"\n📊 Final summary:")
            print(f"   Total products: {len(products)}")
            print(f"   With image embeddings: {with_emb}")
            print(f"   With info embeddings: {with_info}")
            print(f"\n💾 Data saved to: {OUTPUT_PATH}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
