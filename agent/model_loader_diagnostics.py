#!/usr/bin/env python3
"""
Model Loader Diagnostics
- Tests loading SentenceTransformer from local snapshot path
- If invalid, removes local cache and re-downloads from Hugging Face
- Falls back to direct HF load if needed
"""

import os
import shutil
from loguru import logger

from sentence_transformers import SentenceTransformer


def try_load_local(local_model_root: str) -> SentenceTransformer | None:
    try:
        # Prefer exact snapshot directory if present
        import glob
        snapshot_glob = "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        candidates = glob.glob(f"{local_model_root}/{snapshot_glob}/*")
        model_path = candidates[0] if candidates else local_model_root
        model = SentenceTransformer(model_path)
        logger.info(f" Loaded local model from: {model_path}")
        return model
    except Exception as e:
        logger.warning(f" Local model load failed: {e}")
        return None


def repair_local(local_model_root: str) -> bool:
    try:
        if os.path.isdir(local_model_root):
            logger.info(f" Removing invalid local model directory: {local_model_root}")
            shutil.rmtree(local_model_root, ignore_errors=True)
        return True
    except Exception as e:
        logger.warning(f" Failed to remove local model dir: {e}")
        return False


def try_load_hf() -> SentenceTransformer | None:
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info(" Loaded model from Hugging Face")
        return model
    except Exception as e:
        logger.error(f" Failed to load from Hugging Face: {e}")
        return None


def main() -> int:
    local_model_root = "/shared/khoja/CogComp/models/sentence_transformers"

    # 1) Try local
    model = try_load_local(local_model_root)
    if model is not None:
        return 0

    # 2) Repair local and try HF (which will also repopulate cache)
    repair_local(local_model_root)
    model = try_load_hf()
    if model is not None:
        return 0

    # 3) Final fallback: attempt local again in case HF repopulated cache implicitly
    model = try_load_local(local_model_root)
    if model is not None:
        return 0

    logger.error(" All attempts to load the sentence transformer failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())



