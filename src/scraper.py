"""
Shopify scraper for Rebalance Vintage.

Uses the Shopify JSON API to extract all products from all category pages.
- Collection listing: /collections/{handle}/products.json?page=N&limit=250
- Product detail:   /products/{handle}.json
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm.asyncio import tqdm

from src.config import (
    CATEGORY_MAP,
    CATEGORY_URLS,
    CHECKPOINT_PATH,
    CURRENCY,
    DEFAULT_COUNTRY,
    HTTP_CONCURRENCY,
    OUTPUT_PATH,
    REQUEST_DELAY,
    SHOPIFY_PAGE_LIMIT,
    SOURCE,
    STORE_BASE_URL,
    BRAND,
    SECOND_HAND,
)

# ── HTTP Client ───────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
            },
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ── Retry helper ──────────────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
async def _fetch_json(url: str) -> dict[str, Any] | list[Any]:
    client = get_client()
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()


# ── Collection pagination ─────────────────────────────────────────────────────


async def fetch_collection_products(
    handle: str,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """
    Fetch ALL products from a collection by walking pages until an empty page.
    Returns the list of raw product dicts from the Shopify API.
    """
    all_products: list[dict[str, Any]] = []
    page = 1

    pbar_desc = f"📦 {CATEGORY_MAP.get(handle, handle)}"
    pbar = tqdm(desc=pbar_desc, unit=" prod", leave=False)

    while True:
        url = f"{STORE_BASE_URL}/collections/{handle}/products.json?page={page}&limit={SHOPIFY_PAGE_LIMIT}"
        async with semaphore:
            data = await _fetch_json(url)
            products = data.get("products", [])
            if not products:
                break
            all_products.extend(products)
            pbar.update(len(products))
            page += 1
            if page > 1:
                await asyncio.sleep(REQUEST_DELAY)

    pbar.close()
    return all_products


# ── Data extraction ───────────────────────────────────────────────────────────


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n").strip()


def _clean_handle(handle: str) -> str:
    """Create a safe, readable product ID from the Shopify handle."""
    return handle.strip().lower()


def build_product_record(
    product: dict[str, Any],
    categories: list[str],
) -> dict[str, Any]:
    """
    Transform a Shopify product dict into the database schema format.
    """
    handle: str = product.get("handle", "")
    product_id = _clean_handle(handle)

    # ── Variants (take first variant as primary) ──────────────────────────
    variants: list[dict[str, Any]] = product.get("variants", [])
    primary_variant = variants[0] if variants else {}

    # Price & sale logic
    raw_price = (primary_variant.get("price") or "").strip()
    raw_compare = (primary_variant.get("compare_at_price") or "").strip()

    price_str: str | None = None
    sale_str: str | None = None

    if raw_compare and raw_compare != "0":
        # compare_at_price is set → item is on sale
        # price = original, sale = current (sale price)
        price_str = f"{raw_compare}{CURRENCY}" if raw_compare else None
        sale_str = f"{raw_price}{CURRENCY}" if raw_price else None
    else:
        # No sale
        price_str = f"{raw_price}{CURRENCY}" if raw_price else None
        sale_str = None

    # ── Images ────────────────────────────────────────────────────────────
    images: list[dict[str, Any]] = product.get("images", [])
    image_url: str | None = None
    additional_images: list[str] = []

    for i, img in enumerate(images):
        src = img.get("src", "")
        if src:
            if i == 0:
                image_url = src
            else:
                additional_images.append(src)

    additional_images_str = " , ".join(additional_images) if additional_images else None

    # ── Description ───────────────────────────────────────────────────────
    body_html: str = product.get("body_html", "") or ""
    description = _strip_html(body_html) if body_html else None

    # ── Tags ──────────────────────────────────────────────────────────────
    # products.json returns tags as a list, product.json returns as comma-separated string
    tags_raw = product.get("tags", "") or ""
    if isinstance(tags_raw, list):
        tags_list = [t.strip() for t in tags_raw if t.strip()]
    elif isinstance(tags_raw, str):
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        tags_list = []
    
    tags_lower = (" ".join(tags_list)).lower() if tags_list else ""

    # ── Sizes & options ───────────────────────────────────────────────────
    options: list[dict[str, Any]] = product.get("options", [])
    size_values: list[str] = []
    sizes_str: str | None = None
    all_colors: list[str] = []

    for opt in options:
        opt_name = (opt.get("name") or "").lower()
        opt_values: list[str] = opt.get("values", [])
        if opt_name == "size" or opt_name == "sizes":
            size_values = opt_values
        if opt_name == "color" or opt_name == "colour" or opt_name == "colors":
            all_colors = opt_values

    if size_values:
        sizes_str = ", ".join(size_values)
    elif primary_variant:
        # Fallback: use option1 from primary variant
        opt1 = primary_variant.get("option1")
        if opt1:
            sizes_str = opt1

    # Combine size & color info for metadata
    variant_title = primary_variant.get("title", "")
    color_str = ", ".join(all_colors) if all_colors else (variant_title.split(" / ")[-1] if " / " in variant_title else None)

    # ── Metadata ─────────────────────────────────────────────────────────
    metadata = {
        "handle": handle,
        "shopify_id": product.get("id"),
        "product_type": product.get("product_type"),
        "vendor": product.get("vendor"),
        "size": sizes_str,
        "color": color_str,
        "variant_title": variant_title,
        "sku": primary_variant.get("sku"),
        "variants_count": len(variants),
        "images_count": len(images),
        "options": [
            {"name": o.get("name"), "values": o.get("values")}
            for o in options
        ],
        "all_variants": [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "price": v.get("price"),
                "compare_at_price": v.get("compare_at_price"),
                "sku": v.get("sku"),
                "option1": v.get("option1"),
                "option2": v.get("option2"),
                "option3": v.get("option3"),
            }
            for v in variants
        ],
    }

    # ── Gender inference ─────────────────────────────────────────────────
    # This is a unisex vintage store, but some items may lean one way
    # Also check variant titles and tags for gender cues
    gender: str | None = None
    title_lower = product.get("title", "").lower()
    desc_lower = body_html.lower()
    variant_titles = " ".join(v.get("title", "").lower() for v in variants)
    combined = f"{title_lower} {desc_lower} {tags_lower} {variant_titles}"
    if any(w in combined for w in ["women", "woman", "female", "ladies", "womens"]):
        gender = "women"
    elif any(w in combined for w in ["men", "man", "male", "gents", "mens"]):
        gender = "men"
    else:
        gender = "unisex"

    # ── Build record ─────────────────────────────────────────────────────
    product_url = f"{STORE_BASE_URL}/products/{handle}"
    now_iso = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "id": product_id,
        "source": SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": image_url,
        "brand": BRAND,
        "title": product.get("title"),
        "description": description,
        "category": ", ".join(categories) if categories else None,
        "gender": gender,
        "second_hand": SECOND_HAND,
        "price": price_str,
        "sale": sale_str,
        "additional_images": additional_images_str,
        "size": sizes_str,
        "country": DEFAULT_COUNTRY,
        "tags": tags_list if tags_list else None,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "created_at": now_iso,
        "other": None,
        # These will be filled later
        "image_embedding": None,
        "info_embedding": None,
        "compressed_image_url": None,
    }

    return record


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def load_checkpoint() -> dict[str, Any]:
    """Load progress checkpoint from disk."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(state: dict[str, Any]) -> None:
    """Save progress checkpoint to disk."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_products_from_disk() -> dict[str, dict[str, Any]]:
    """Load previously scraped products from disk."""
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    return {}


def save_products_to_disk(products: dict[str, dict[str, Any]]) -> None:
    """Save all scraped products to disk."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)


# ── Main scraper orchestrator ─────────────────────────────────────────────────


async def scrape_all_products(
    scrape_all: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Scrape all products from all categories with pagination.
    Returns a dict keyed by product handle (for deduplication).
    """
    products: dict[str, dict[str, Any]] = {}

    # Load existing data if resuming
    if not scrape_all:
        products = load_products_from_disk()
        print(f"🔄 Resuming: {len(products)} products already on disk")

    semaphore = asyncio.Semaphore(HTTP_CONCURRENCY)

    # Track which handles have been seen per-category to build category lists
    handle_categories: dict[str, set[str]] = {}

    for handle, display_name in CATEGORY_URLS:
        print(f"\n{'='*60}")
        print(f"📂 Scraping category: {display_name} ({handle})")
        print(f"{'='*60}")

        try:
            raw_products = await fetch_collection_products(handle, semaphore)
        except Exception as e:
            print(f"❌ Failed to scrape category '{handle}': {e}")
            continue

        if not raw_products:
            print(f"⚠️  No products found for '{display_name}'")
            continue

        # Process products
        for raw in raw_products:
            prod_handle = raw.get("handle", "").strip().lower()
            if not prod_handle:
                continue

            # Track categories for this product
            if prod_handle not in handle_categories:
                handle_categories[prod_handle] = set()
            handle_categories[prod_handle].add(display_name)

            # Build record (with categories from first occurrence)
            categories = list(handle_categories[prod_handle])
            if prod_handle not in products:
                record = build_product_record(raw, categories)
                products[prod_handle] = record
            else:
                # Update categories on existing record
                products[prod_handle]["category"] = ", ".join(
                    sorted(handle_categories[prod_handle])
                )

            # Update checkpoint periodically
            save_checkpoint({"processed": len(products), "last_category": handle})

        print(f"✅ {display_name}: {len(raw_products)} products → {len(products)} unique total")

    # Final update: ensure all products have correct merged categories
    for prod_handle, record in products.items():
        cats = handle_categories.get(prod_handle, set())
        if cats:
            record["category"] = ", ".join(sorted(cats))

    # Save to disk
    save_products_to_disk(products)
    print(f"\n{'='*60}")
    print(f"🎯 Scraping complete! {len(products)} unique products saved to {OUTPUT_PATH}")
    print(f"{'='*60}")

    return products
