from __future__ import annotations

from typing import List

from app.models.schemas import PlaceLite

# Generic-only type tags commonly present; indicate weak categorization
GENERIC_TYPES = {"point_of_interest", "establishment"}


def _looks_residential_address(addr: str | None) -> bool:
    if not addr:
        return False
    a = addr.lower()
    # Very light heuristics: apartment/unit indicators or lack of commas (single-line)
    if any(token in a for token in ["apt", "apartment", "unit", "suite #", "lot "]):
        return True
    # If no comma at all, often a weak single-line address or POI label
    if "," not in a and any(ch.isdigit() for ch in a):
        return True
    return False


def apply_residential_filter(results: List[PlaceLite], exclude_service_area_only: bool = True) -> List[PlaceLite]:
    if not exclude_service_area_only:
        return results

    filtered: List[PlaceLite] = []
    for r in results:
        # Primary signal: pureServiceAreaBusiness == True
        if r.pureServiceAreaBusiness is True:
            continue

        # Safety net heuristics: only generic tags AND 0 reviews AND residential-looking address
        types_set = set(r.types or [])
        non_generic_types = [t for t in types_set if t not in GENERIC_TYPES]

        if not non_generic_types:
            # No clear trade type present
            rating_zero = (r.userRatingCount or 0) == 0
            if rating_zero and _looks_residential_address(r.formattedAddress):
                # Mark as likely home-based and drop
                continue

        filtered.append(r)

    return filtered
