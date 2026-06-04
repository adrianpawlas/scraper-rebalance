"""
Supabase database operations for the products table.

Features:
- Batch upsert with retry (3 attempts)
- Smart comparison: detect changed vs unchanged products
- Stale product tracking (2 consecutive misses = delete)
- Failure logging to file
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import Client, create_client
from tqdm import tqdm

from src.config import DB_BATCH_SIZE, DB_TABLE, SOURCE, SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

# ── HTTP Client ───────────────────────────────────────────────────────────────

_client: Client | None = None

FAILED_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "failed_products.log"


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY is not set. Add it to .env or environment.")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Helpers ────────────────────────────────────────────────────────────────────


def _prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert record to Supabase-friendly format (handles None, lists, vectors)."""
    prepared = {}
    for key, value in record.items():
        if value is None:
            prepared[key] = None
        elif key in ("tags",) and isinstance(value, list):
            prepared[key] = value  # SDK serializes as PG array
        elif key in ("image_embedding", "info_embedding") and isinstance(value, list):
            prepared[key] = value  # SDK serializes as vector
        else:
            prepared[key] = value
    return prepared


def _log_failed_products(records: list[dict[str, Any]], error: str) -> None:
    """Append failed product IDs and error to the failure log file."""
    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids = [r.get("id", "?") for r in records]
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(FAILED_LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] ERROR: {error}\n")
        f.write(f"  Failed IDs: {ids}\n\n")


# ── Fetch existing products ──────────────────────────────────────────────────


def fetch_existing_products(source: str = SOURCE) -> dict[str, dict[str, Any]]:
    """
    Fetch ALL existing products for the given source from Supabase.
    Returns a dict keyed by product ID (handle) for fast lookup.
    """
    client = get_client()
    existing: dict[str, dict[str, Any]] = {}
    page = 0
    page_size = 1000

    print(f"📖 Fetching existing products for source '{source}'...")

    while True:
        try:
            resp = (
                client.table(DB_TABLE)
                .select("*")
                .eq("source", source)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            rows = resp.data if hasattr(resp, "data") else resp
            if not rows:
                break
            for row in rows:
                pid = row.get("id")
                if pid:
                    existing[pid] = row
            page += 1
            if len(rows) < page_size:
                break
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch page {page}: {e}")
            break

    print(f"   Found {len(existing)} existing products")
    return existing


# ── Comparison logic ─────────────────────────────────────────────────────────


def _normalize(val: Any) -> str:
    """Normalize a value for comparison (handles None, list, string)."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ",".join(sorted(str(v).strip() for v in val if v))
    return str(val).strip()


def has_product_changed(
    existing: dict[str, Any],
    scraped: dict[str, Any],
) -> tuple[bool, bool]:
    """
    Compare a scraped product against the existing database record.

    Returns:
        (has_changed: bool, image_changed: bool)
    """
    # Fields to compare for content changes
    compare_fields = [
        "title",
        "price",
        "sale",
        "image_url",
        "additional_images",
        "description",
        "category",
        "size",
    ]

    has_changed = False
    image_changed = False

    for field in compare_fields:
        old_val = _normalize(existing.get(field))
        new_val = _normalize(scraped.get(field))
        if old_val != new_val:
            has_changed = True
            if field == "image_url":
                image_changed = True

    # Compare tags separately (list comparison)
    old_tags = _normalize(existing.get("tags"))
    new_tags = _normalize(scraped.get("tags"))
    if old_tags != new_tags:
        has_changed = True

    return has_changed, image_changed


# ── Batch upsert with retry ──────────────────────────────────────────────────


def batch_upsert(
    records: list[dict[str, Any]],
    batch_size: int = DB_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Upsert records into Supabase in batches of `batch_size`.
    Retries each batch up to 3 times on failure.

    Returns:
        (successful_records, failed_records)
    """
    if not records:
        return [], []

    client = get_client()
    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prepared = [_prepare_record(r) for r in batch]

        last_error = None
        for attempt in range(3):
            try:
                client.table(DB_TABLE).upsert(
                    prepared,
                    on_conflict="id",
                    ignore_duplicates=False,
                ).execute()
                successful.extend(batch)
                break
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        f"⚠️  Batch upsert failed (attempt {attempt + 1}/3, offset {i}): "
                        f"{e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"❌ Batch upsert failed after 3 attempts (offset {i}): {e}")
                    # Try individual upserts to isolate failures
                    for record in prepared:
                        try:
                            client.table(DB_TABLE).upsert(
                                record,
                                on_conflict="id",
                                ignore_duplicates=False,
                            ).execute()
                            successful.append(record)
                        except Exception as e2:
                            logger.error(f"❌ Failed to upsert product {record.get('id', '?')}: {e2}")
                            failed.append(record)

        if last_error:
            _log_failed_products(batch, last_error)

    return successful, failed


# ── Stale product tracking ────────────────────────────────────────────────────


def parse_other(other_val: Any) -> dict[str, Any]:
    """Parse the `other` JSON field safely."""
    if not other_val:
        return {}
    if isinstance(other_val, dict):
        return other_val
    try:
        return json.loads(str(other_val))
    except (json.JSONDecodeError, TypeError):
        return {}


def classify_stale_products(
    existing_products: dict[str, dict[str, Any]],
    seen_handles: set[str],
) -> tuple[list[str], list[str]]:
    """
    Classify which products are stale based on the current scrape run.

    Args:
        existing_products: Dict of existing products from DB (keyed by ID)
        seen_handles: Set of product handles seen in the current scrape

    Returns:
        (stale_one_run: list of IDs to increment stale_count, 
         stale_two_runs: list of IDs to delete)
    """
    stale_one_run: list[str] = []
    stale_two_runs: list[str] = []

    for pid, record in existing_products.items():
        if pid not in seen_handles:
            other = parse_other(record.get("other"))
            current_stale = other.get("stale_count", 0)
            new_stale = current_stale + 1

            if new_stale >= 2:
                stale_two_runs.append(pid)
            else:
                stale_one_run.append(pid)

    return stale_one_run, stale_two_runs


def update_stale_counts(ids: list[str], stale_count: int = 1) -> None:
    """
    Update the `other` field for stale products with their new stale_count.
    """
    if not ids:
        return

    client = get_client()
    other_value = json.dumps({"stale_count": stale_count})

    for pid in ids:
        try:
            client.table(DB_TABLE).update(
                {"other": other_value}
            ).eq("id", pid).execute()
        except Exception as e:
            logger.warning(f"⚠️  Failed to update stale count for {pid}: {e}")


def delete_products(ids: list[str]) -> int:
    """
    Delete products by ID. Returns number of successfully deleted.
    """
    if not ids:
        return 0

    client = get_client()
    deleted = 0

    for pid in ids:
        try:
            client.table(DB_TABLE).delete().eq("id", pid).execute()
            deleted += 1
        except Exception as e:
            logger.warning(f"⚠️  Failed to delete stale product {pid}: {e}")

    return deleted


# ── High-level orchestration ──────────────────────────────────────────────────


def upsert_products_smart(
    records: list[dict[str, Any]],
    batch_size: int = DB_BATCH_SIZE,
) -> tuple[int, int, list[dict[str, Any]]]:
    """
    Upsert products with retry logic.
    Returns (success_count, error_count, failed_records).
    """
    successful, failed = batch_upsert(records, batch_size)
    return len(successful), len(failed), failed
