"""
Supabase database operations for the products table.
"""

from __future__ import annotations

import logging
from typing import Any

from supabase import Client, create_client
from tqdm import tqdm

from src.config import DB_BATCH_SIZE, DB_TABLE, SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY is not set. Add it to .env or environment.")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _prepare_record_for_db(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert record to the format expected by Supabase.
    Handles None → null, lists → pg arrays, etc.
    """
    prepared = {}
    for key, value in record.items():
        if value is None:
            prepared[key] = None
        elif key == "tags" and isinstance(value, list):
            prepared[key] = value  # Supabase Python SDK handles lists as PG arrays
        elif key == "image_embedding" and isinstance(value, list):
            prepared[key] = value  # Supabase SDK serializes as vector
        elif key == "info_embedding" and isinstance(value, list):
            prepared[key] = value
        else:
            prepared[key] = value
    return prepared


def upsert_products(
    products: list[dict[str, Any]],
    batch_size: int = DB_BATCH_SIZE,
) -> tuple[int, int]:
    """
    Upsert products into Supabase. Uses the `source, product_url` unique constraint
    to handle conflicts.
    
    Returns (success_count, error_count).
    """
    if not products:
        return 0, 0

    client = get_client()
    success = 0
    errors = 0

    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        prepared = [_prepare_record_for_db(r) for r in batch]

        try:
            response = client.table(DB_TABLE).upsert(
                prepared,
                on_conflict="id",
                ignore_duplicates=False,
            ).execute()
            success += len(batch)
        except Exception as e:
            logger.error(f"❌ Batch upsert failed (offset {i}): {e}")
            # Try individually to isolate failures
            for record in prepared:
                try:
                    client.table(DB_TABLE).upsert(
                        record,
                        on_conflict="id",
                        ignore_duplicates=False,
                    ).execute()
                    success += 1
                except Exception as e2:
                    logger.error(f"❌ Failed to upsert product {record.get('id', '?')}: {e2}")
                    errors += 1

    return success, errors


def upsert_all_products(
    products: dict[str, dict[str, Any]],
    batch_size: int = DB_BATCH_SIZE,
) -> None:
    """
    Upsert all products from the dict into Supabase with a progress bar.
    """
    records = list(products.values())
    print(f"\n{'='*60}")
    print(f"📤 Uploading {len(records)} products to Supabase...")
    print(f"{'='*60}")

    success, errors = 0, 0
    with tqdm(total=len(records), desc="📡 Uploading", unit=" prod") as pbar:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            s, e = upsert_products(batch)
            success += s
            errors += e
            pbar.update(len(batch))

    print(f"\n✅ Upload complete! Success: {success}, Errors: {errors}")
    if errors > 0:
        print(f"⚠️  {errors} products failed to upload. Check logs for details.")
