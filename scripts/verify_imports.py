"""Verify that all core modules can be imported successfully."""

import sys


def main():
    print(f"Python {sys.version}")

    from src.config import CATEGORY_URLS, TORCH_DEVICE
    print(f"✅ Config loaded: {len(CATEGORY_URLS)} categories, device={TORCH_DEVICE}")

    from src.scraper import build_product_record, scrape_all_products
    print("✅ Scraper module loaded")

    from src.embeddings import generate_embeddings_for_products, embed_images_batch_with_stagger, embed_text
    print("✅ Embeddings module loaded")

    from src.database import upsert_products_smart, fetch_existing_products, batch_upsert
    print("✅ Database module loaded")

    print("All imports successful!")


if __name__ == "__main__":
    main()
