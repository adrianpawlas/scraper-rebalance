#!/usr/bin/env python3
"""
Rebalance Vintage Scraper — Smart Pipeline

Orchestrates the full pipeline with intelligent product management:

1. Scrape all products from Shopify (paginated categories)
2. Compare scraped data against existing Supabase records
3. Only generate embeddings for new or image-changed products (with 0.5s stagger)
4. Batch upsert (50/batch) with 3x retry — only new/changed products
5. Handle stale products: 1st miss = warning, 2nd consecutive miss = delete
6. Print run summary

Usage:
    python -m src.main              # Full smart pipeline
    python -m src.main --scrape     # Only scrape (save to disk)
    python -m src.main --embed      # Only generate embeddings (from saved data)
    python -m src.main --upload     # Only upload to Supabase (from saved data)
    python -m src.main --force      # Force re-embed all products
    python -m src.main --from-scratch  # Force re-scrape everything
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.config import (
    EMBEDDING_MODEL,
    OUTPUT_PATH,
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
        description="Rebalance Vintage Scraper — Smart Pipeline",
    )
    parser.add_argument("--scrape", action="store_true", help="Only scrape")
    parser.add_argument("--embed", action="store_true", help="Only generate embeddings")
    parser.add_argument("--upload", action="store_true", help="Only upload to Supabase")
    parser.add_argument("--force", action="store_true", help="Force re-embed all products")
    parser.add_argument("--from-scratch", action="store_true", help="Force re-scrape everything")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embeddings")

    args = parser.parse_args()

    mode_scrape = args.scrape
    mode_embed = args.embed
    mode_upload = args.upload
    full_pipeline = not (mode_scrape or mode_embed or mode_upload)

    print(f"""
{'='*60}
   Rebalance Vintage Scraper — Smart Pipeline
   Source: {SOURCE}
   Model:  {EMBEDDING_MODEL}
   Device: {TORCH_DEVICE}
{'='*60}
""")

    run_start = time.time()

    # ── STEP 1: Scrape ────────────────────────────────────────────────────
    if full_pipeline or mode_scrape:
        from src.scraper import (
            close_client,
            load_products_from_disk,
            save_products_to_disk,
            scrape_all_products,
        )

        import asyncio

        async def _run_scrape():
            result = await scrape_all_products(scrape_all=not args.from_scratch)
            await close_client()
            return result

        scraped_products, seen_handles = asyncio.run(_run_scrape())

    # ── STEP 2 (full pipeline only): Compare, classify, embed, upsert ─────
    if full_pipeline:
        from src.database import (
            classify_stale_products,
            delete_products,
            fetch_existing_products,
            has_product_changed,
            update_stale_counts,
            upsert_products_smart,
        )

        # 2a. Fetch existing products from Supabase
        existing_products = fetch_existing_products(SOURCE)

        # 2b. Classify every scraped product
        new_handles: list[str] = []
        changed_handles: list[str] = []
        unchanged_handles: list[str] = []
        image_changed_handles: list[str] = []

        for handle, scraped in scraped_products.items():
            existing = existing_products.get(handle)
            if existing is None:
                new_handles.append(handle)
            else:
                has_changed, img_changed = has_product_changed(existing, scraped)
                if has_changed:
                    changed_handles.append(handle)
                    if img_changed:
                        image_changed_handles.append(handle)
                else:
                    unchanged_handles.append(handle)

        # 2c. Determine which products need embedding
        if args.force:
            handles_to_embed = new_handles + changed_handles + unchanged_handles
            print(f"\n--force: will re-embed all {len(handles_to_embed)} products")
        else:
            # New products + image-changed products need embedding
            handles_to_embed = list(set(new_handles + image_changed_handles))

        # 2d. Generate embeddings (only for products that need it)
        if not args.skip_embed and handles_to_embed:
            from src.embeddings import generate_embeddings_for_products
            scraped_products = generate_embeddings_for_products(
                scraped_products, handles_to_embed
            )

        # 2e. Reset stale_count for products that were previously flagged but are
        #     seen again in this scrape (so stale_count doesn't accumulate)
        from src.database import parse_other
        seen_but_had_stale: list[str] = []
        for pid in seen_handles:
            existing = existing_products.get(pid)
            if existing:
                other = parse_other(existing.get("other"))
                if other.get("stale_count", 0) > 0:
                    seen_but_had_stale.append(pid)

        if seen_but_had_stale:
            print(f"\nResetting stale count for {len(seen_but_had_stale)} previously-stale products...")
            update_stale_counts(seen_but_had_stale, stale_count=0)

        # 2f. Batch upsert only new + changed products
        records_to_upsert = [
            scraped_products[h] for h in new_handles + changed_handles
        ]
        if records_to_upsert:
            print(f"\n{'='*60}")
            print(f"Uploading {len(records_to_upsert)} products to Supabase "
                  f"({len(new_handles)} new, {len(changed_handles)} updated)")
            print(f"{'='*60}")
            success, errors, failed_records = upsert_products_smart(records_to_upsert)
            if errors:
                print(f"   {errors} products failed to upload (logged to data/failed_products.log)")
        else:
            success, errors = 0, 0
            print("\nNo new or changed products to upload")

        # 2g. Handle stale products
        stale_one_run, stale_two_runs = classify_stale_products(
            existing_products, seen_handles
        )

        if stale_one_run:
            print(f"\nMarking {len(stale_one_run)} products as stale (1st missed run)...")
            update_stale_counts(stale_one_run, stale_count=1)

        if stale_two_runs:
            print(f"\nDeleting {len(stale_two_runs)} products (2nd consecutive missed run)...")
            deleted = delete_products(stale_two_runs)

        # 2g. Print run summary
        elapsed = time.time() - run_start
        print(f"\n{'='*60}")
        print(f"RUN SUMMARY")
        print(f"{'='*60}")
        print(f"   New products:           {len(new_handles)}")
        print(f"   Products updated:       {len(changed_handles)}")
        print(f"   Products unchanged:     {len(unchanged_handles)} (skipped)")
        print(f"   Embeddings generated:   {len(handles_to_embed)}")
        print(f"   Stale (1st miss):       {len(stale_one_run)}")
        print(f"   Stale deleted:          {len(stale_two_runs)}")
        print(f"   Duration:               {elapsed:.1f}s")
        if records_to_upsert:
            print(f"   DB upsert success:      {success}")
            print(f"   DB upsert errors:       {errors}")
        print(f"{'='*60}")

    # ── Standalone embed mode ──────────────────────────────────────────────
    if mode_embed:
        from src.embeddings import generate_embeddings_for_products
        from src.scraper import load_products_from_disk, save_products_to_disk

        products = load_products_from_disk()
        if not products:
            print("No products found on disk. Run --scrape first.")
            sys.exit(1)

        handles = list(products.keys())
        products = generate_embeddings_for_products(products, handles)
        save_products_to_disk(products)

    # ── Standalone upload mode ─────────────────────────────────────────────
    if mode_upload:
        from src.database import upsert_products_smart
        from src.scraper import load_products_from_disk

        products = load_products_from_disk()
        if not products:
            print("No products found on disk. Nothing to upload.")
            sys.exit(1)

        records = list(products.values())
        print(f"Uploading {len(records)} products to Supabase...")
        success, errors, failed = upsert_products_smart(records)
        print(f"Done! {success} success, {errors} errors")

    if full_pipeline:
        print("\nDone!")


if __name__ == "__main__":
    main()
