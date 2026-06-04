"""
Embedding generation using Google's SIGLIP model.

Model: google/siglip-base-patch16-384 → 768-dim embeddings
- Image embedding: SiglipVisionModel (vision tower)
- Text embedding:  SiglipTextModel  (text tower, same dim)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import httpx
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    SiglipModel,
    SiglipTextModel,
    SiglipVisionModel,
)

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, TORCH_DEVICE

logger = logging.getLogger(__name__)

# ── Global model references (lazy-loaded) ─────────────────────────────────────

_model: SiglipModel | None = None
_vision_model: SiglipVisionModel | None = None
_text_model: SiglipTextModel | None = None
_processor: AutoProcessor | None = None
_device: torch.device | None = None


def _ensure_model():
    """Lazy-load the SIGLIP model and processor."""
    global _model, _vision_model, _text_model, _processor, _device

    if _processor is not None:
        return

    _device = torch.device(TORCH_DEVICE)
    logger.info(f"📦 Loading SIGLIP model: {EMBEDDING_MODEL} on {_device}")

    _processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL)

    # Load the full model for text embeddings (shared text tower)
    # Load separate vision model for image embeddings
    _vision_model = SiglipVisionModel.from_pretrained(EMBEDDING_MODEL).to(_device)
    _text_model = SiglipTextModel.from_pretrained(EMBEDDING_MODEL).to(_device)

    _vision_model.eval()
    _text_model.eval()

    logger.info(f"✅ SIGLIP model loaded (dim={EMBEDDING_DIM})")


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


# ── Image Embedding ───────────────────────────────────────────────────────────


def embed_image(image_url: str) -> list[float] | None:
    """
    Generate a 768-dim image embedding for a single product image.
    Returns a list of floats or None on failure.
    """
    _ensure_model()
    try:
        image = _download_image(image_url)
    except Exception as e:
        logger.warning(f"⚠️  Failed to download image {image_url}: {e}")
        return None

    try:
        inputs = _processor(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _vision_model(**inputs)
            # pooler_output is the [CLS] token embedding → 768-dim
            embedding = outputs.pooler_output  # shape: (1, 768)
        return embedding.cpu().squeeze().tolist()
    except Exception as e:
        logger.warning(f"⚠️  Failed to embed image {image_url}: {e}")
        return None


def embed_images_batch(
    image_urls: list[str],
) -> list[list[float] | None]:
    """
    Generate image embeddings for a batch of image URLs.
    Downloads and processes images in small chunks to avoid memory pressure.
    Returns a list of embeddings (or None for failures).
    """
    _ensure_model()

    results: list[list[float] | None] = [None] * len(image_urls)

    with tqdm(total=len(image_urls), desc="📸 Image embeddings", unit=" img") as pbar:
        # Process in small chunks: download + infer together to free memory
        chunk_size = max(1, EMBEDDING_BATCH_SIZE)
        for chunk_start in range(0, len(image_urls), chunk_size):
            chunk_urls = image_urls[chunk_start : chunk_start + chunk_size]
            chunk_indices = list(range(chunk_start, min(chunk_start + chunk_size, len(image_urls))))

            # Download this chunk
            images: list[Image.Image] = []
            valid_indices: list[int] = []
            for i, url in zip(chunk_indices, chunk_urls):
                try:
                    img = _download_image(url)
                    images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to download {url}: {e}")
                    pbar.update(1)

            if not images:
                pbar.update(len(chunk_urls) - len(images))
                continue

            # Process this chunk's images
            try:
                inputs = _processor(images=images, return_tensors="pt", padding=True).to(_device)
                with torch.no_grad():
                    outputs = _vision_model(**inputs)
                    embeddings = outputs.pooler_output  # (B, 768)
                for idx, emb in zip(valid_indices, embeddings.cpu()):
                    results[idx] = emb.tolist()
            except Exception as e:
                logger.warning(f"⚠️  Chunk embedding failed at offset {chunk_start}: {e}")
                # Retry individually for this chunk
                for idx, img in zip(valid_indices, images):
                    try:
                        inp = _processor(images=img, return_tensors="pt").to(_device)
                        with torch.no_grad():
                            out = _vision_model(**inp)
                            results[idx] = out.pooler_output.cpu().squeeze().tolist()
                    except Exception as e2:
                        logger.warning(f"⚠️  Individual image embedding failed: {e2}")

            # Free memory
            del images, inputs, outputs, embeddings
            if torch.cuda.is_available():
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
    """
    Generate a 768-dim text embedding using SIGLIP's text encoder.
    """
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


def embed_info_text(record: dict[str, Any]) -> list[float]:
    """
    Generate a text embedding from all product info fields.
    """
    text = _prepare_text_for_embedding(record)
    return embed_text(text)


# ── Batch text embedding ──────────────────────────────────────────────────────


def embed_texts_batch(texts: list[str], pbar: tqdm | None = None) -> list[list[float]]:
    """
    Generate text embeddings for a batch of text strings.
    Optionally accepts a tqdm progress bar to update.
    """
    _ensure_model()
    results: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
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
            logger.warning(f"⚠️  Text batch embedding failed at offset {start}: {e}")
            # Fall back to individual
            for t in batch:
                results.append(embed_text(t))

        if pbar:
            pbar.update(len(batch))

    return results


# ── Process all products ──────────────────────────────────────────────────────


def generate_all_embeddings(
    products: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Generate image and text embeddings for all products.
    Returns the products dict with embeddings filled in.
    """
    _ensure_model()

    product_handles = list(products.keys())
    print(f"\n{'='*60}")
    print(f"🧠 Generating embeddings for {len(product_handles)} products")
    print(f"{'='*60}")

    # ── Image embeddings ─────────────────────────────────────────────────
    print("\n📸 Processing image embeddings...")
    image_urls: list[str | None] = []
    valid_handles_img: list[str] = []
    for h in product_handles:
        url = products[h].get("image_url")
        if url:
            image_urls.append(url)
            valid_handles_img.append(h)
        else:
            products[h]["image_embedding"] = None

    if image_urls:
        image_embeddings = embed_images_batch(image_urls)
        for h, emb in zip(valid_handles_img, image_embeddings):
            products[h]["image_embedding"] = emb

    # ── Text embeddings ──────────────────────────────────────────────────
    print("\n📝 Processing text (info) embeddings...")
    texts: list[str] = []
    valid_handles_text: list[str] = []
    for h in product_handles:
        text = _prepare_text_for_embedding(products[h])
        if text.strip():
            texts.append(text)
            valid_handles_text.append(h)
        else:
            products[h]["info_embedding"] = None

    if texts:
        with tqdm(total=len(texts), desc="🔤 Text embeddings", unit=" text") as pbar:
            text_embeddings = embed_texts_batch(texts, pbar=pbar)
            for h, emb in zip(valid_handles_text, text_embeddings):
                products[h]["info_embedding"] = emb

    print(f"\n✅ Embeddings complete! "
          f"Image: {sum(1 for p in products.values() if p['image_embedding'])}/{len(products)}, "
          f"Text: {sum(1 for p in products.values() if p['info_embedding'])}/{len(products)}")

    return products
