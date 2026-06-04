"""
Embedding generation using Google's SIGLIP model.

Model: google/siglip-base-patch16-384 -> 768-dim embeddings
- Image embedding: SiglipVisionModel (vision tower)
- Text embedding:  SiglipTextModel  (text tower, same dim)

Features:
- Staggered 0.5s delay between individual image embeddings
- Only embeds products that need it (new or image URL changed)
- Chunked image processing for memory efficiency
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import httpx
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    SiglipTextModel,
    SiglipVisionModel,
)

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, TORCH_DEVICE

logger = logging.getLogger(__name__)

# ── Global model references (lazy-loaded) ─────────────────────────────────────

_vision_model: SiglipVisionModel | None = None
_text_model: SiglipTextModel | None = None
_processor: AutoProcessor | None = None
_device: torch.device | None = None


def _ensure_model():
    """Lazy-load the SIGLIP model and processor."""
    global _vision_model, _text_model, _processor, _device

    if _processor is not None:
        return

    _device = torch.device(TORCH_DEVICE)
    logger.info(f"Loading SIGLIP model: {EMBEDDING_MODEL} on {_device}")

    _processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL)
    _vision_model = SiglipVisionModel.from_pretrained(EMBEDDING_MODEL).to(_device)
    _text_model = SiglipTextModel.from_pretrained(EMBEDDING_MODEL).to(_device)
    _vision_model.eval()
    _text_model.eval()

    logger.info(f"SIGLIP model loaded (dim={EMBEDDING_DIM})")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def _download_image(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image."""
    resp = httpx.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ── Image Embedding (with stagger delay) ──────────────────────────────────────


def embed_images_batch_with_stagger(
    image_urls: list[str],
) -> list[list[float] | None]:
    """
    Generate image embeddings for product images with a 0.5s stagger delay
    between chunks to avoid overwhelming the system.

    Downloads and processes images in small chunks to avoid memory pressure.
    Returns a list of embeddings (or None for failures), one per URL.
    """
    _ensure_model()
    results: list[list[float] | None] = [None] * len(image_urls)

    with tqdm(total=len(image_urls), desc="Image embeddings", unit=" img") as pbar:
        chunk_size = max(1, EMBEDDING_BATCH_SIZE)
        for chunk_start in range(0, len(image_urls), chunk_size):
            chunk_urls = image_urls[chunk_start: chunk_start + chunk_size]
            chunk_indices = list(
                range(chunk_start, min(chunk_start + chunk_size, len(image_urls)))
            )

            # Stagger: 0.5s delay before each chunk (except the first)
            if chunk_start > 0:
                time.sleep(0.5)

            images: list[Image.Image] = []
            valid_indices: list[int] = []
            for i, url in zip(chunk_indices, chunk_urls):
                try:
                    img = _download_image(url)
                    images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
                    pbar.update(1)

            if not images:
                pbar.update(len(chunk_urls) - len(images))
                continue

            try:
                inputs = _processor(images=images, return_tensors="pt", padding=True).to(_device)
                with torch.no_grad():
                    outputs = _vision_model(**inputs)
                    embeddings = outputs.pooler_output  # (B, 768)
                for idx, emb in zip(valid_indices, embeddings.cpu()):
                    results[idx] = emb.tolist()
            except Exception as e:
                logger.warning(f"Chunk embedding failed at offset {chunk_start}: {e}")
                for idx, img in zip(valid_indices, images):
                    try:
                        inp = _processor(images=img, return_tensors="pt").to(_device)
                        with torch.no_grad():
                            out = _vision_model(**inp)
                            results[idx] = out.pooler_output.cpu().squeeze().tolist()
                    except Exception as e2:
                        logger.warning(f"Individual image embedding failed: {e2}")

            del images
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(len(chunk_urls))

    return results


# ── Text Embedding (Info Embedding) ───────────────────────────────────────────


def _prepare_text_for_embedding(record: dict[str, Any]) -> str:
    """
    Concatenate all meaningful product info into a single text string
    for text embedding.
    """
    parts: list[str] = []

    title = record.get("title")
    if title:
        parts.append(f"Title: {title}")

    description = record.get("description")
    if description:
        parts.append(f"Description: {description}")

    category = record.get("category")
    if category:
        parts.append(f"Category: {category}")

    price = record.get("price")
    if price:
        parts.append(f"Price: {price}")

    sale = record.get("sale")
    if sale:
        parts.append(f"Sale Price: {sale}")

    brand = record.get("brand")
    if brand:
        parts.append(f"Brand: {brand}")

    gender = record.get("gender")
    if gender:
        parts.append(f"Gender: {gender}")

    sizes = record.get("size")
    if sizes:
        parts.append(f"Sizes: {sizes}")

    tags = record.get("tags")
    if tags:
        tags_str = ", ".join(tags) if isinstance(tags, list) else tags
        parts.append(f"Tags: {tags_str}")

    metadata_raw = record.get("metadata")
    if metadata_raw:
        import json
        try:
            meta = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
            product_type = meta.get("product_type")
            if product_type:
                parts.append(f"Product Type: {product_type}")
            color = meta.get("color")
            if color:
                parts.append(f"Color: {color}")
        except (json.JSONDecodeError, TypeError):
            pass

    return "\n".join(parts)


def embed_text(text: str) -> list[float]:
    """Generate a 768-dim text embedding using SIGLIP's text encoder."""
    _ensure_model()
    inputs = _processor(
        text=text,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    ).to(_device)

    with torch.no_grad():
        outputs = _text_model(**inputs)
        embedding = outputs.pooler_output  # (1, 768)

    return embedding.cpu().squeeze().tolist()


def embed_texts_batch(texts: list[str], pbar: tqdm | None = None) -> list[list[float]]:
    """Generate text embeddings for a batch of text strings."""
    _ensure_model()
    results: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start: start + EMBEDDING_BATCH_SIZE]
        try:
            inputs = _processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            ).to(_device)
            with torch.no_grad():
                outputs = _text_model(**inputs)
                embeddings = outputs.pooler_output  # (B, 768)
            results.extend(emb.cpu().tolist() for emb in embeddings)
        except Exception as e:
            logger.warning(f"Text batch embedding failed at offset {start}: {e}")
            for t in batch:
                results.append(embed_text(t))

        if pbar:
            pbar.update(len(batch))

    return results


# ── Targeted embedding generation ─────────────────────────────────────────────


def generate_embeddings_for_products(
    products: dict[str, dict[str, Any]],
    handles_to_embed: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Generate image and text embeddings only for the specified product handles.
    Skips products not in the list.

    Returns the updated products dict.
    """
    if not handles_to_embed:
        print("   No products need embedding — skipping")
        return products

    _ensure_model()
    print(f"\n{'='*60}")
    print(f"Generating embeddings for {len(handles_to_embed)} products")
    print(f"{'='*60}")

    # ── Image embeddings (with 0.5s stagger) ──────────────────────────────
    print(f"\nProcessing image embeddings ({len(handles_to_embed)} products)...")
    image_urls: list[str] = []
    valid_handles: list[str] = []
    for h in handles_to_embed:
        url = products[h].get("image_url")
        if url:
            image_urls.append(url)
            valid_handles.append(h)
        else:
            products[h]["image_embedding"] = None

    if image_urls:
        image_embeddings = embed_images_batch_with_stagger(image_urls)
        for h, emb in zip(valid_handles, image_embeddings):
            products[h]["image_embedding"] = emb

    # ── Text embeddings ──────────────────────────────────────────────────
    print(f"\nProcessing text embeddings ({len(handles_to_embed)} products)...")
    texts: list[str] = []
    text_handles: list[str] = []
    for h in handles_to_embed:
        text = _prepare_text_for_embedding(products[h])
        if text.strip():
            texts.append(text)
            text_handles.append(h)
        else:
            products[h]["info_embedding"] = None

    if texts:
        with tqdm(total=len(texts), desc="Text embeddings", unit=" text") as pbar:
            text_embeddings = embed_texts_batch(texts, pbar=pbar)
            for h, emb in zip(text_handles, text_embeddings):
                products[h]["info_embedding"] = emb

    image_count = sum(1 for h in handles_to_embed if products[h].get("image_embedding"))
    info_count = sum(1 for h in handles_to_embed if products[h].get("info_embedding"))
    print(f"\nEmbedding summary for this batch:")
    print(f"   Image embeddings: {image_count}/{len(handles_to_embed)}")
    print(f"   Info embeddings:  {info_count}/{len(handles_to_embed)}")

    return products
