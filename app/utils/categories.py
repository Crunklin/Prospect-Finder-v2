from __future__ import annotations

import json
import os
from typing import List, Optional
from pydantic import BaseModel


class CategoryPack(BaseModel):
    key: str
    label: str
    includedTypes: List[str] = []
    keywords: List[str] = []
    strategy: Optional[str] = None  # optional hint: nearby | text | hybrid


def load_category_packs(path: Optional[str] = None) -> List[CategoryPack]:
    """
    Loads category packs from data/categories.json by default.
    """
    if path is None:
        # __file__ -> app/utils/categories.py
        # project_root = dirname(dirname(dirname(__file__)))
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
        default_path = os.path.join(project_root, "data", "categories.json")
        path = default_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"Category data file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    packs = [CategoryPack(**item) for item in data]
    return packs
