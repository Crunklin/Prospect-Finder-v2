import os
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.services.places_client import PlacesClient, Center
from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    PlaceLite,
)
from app.utils.categories import load_category_packs, CategoryPack
from app.utils.filters import apply_residential_filter

load_dotenv()


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth in meters."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _parse_address_parts(addr: str) -> Dict[str, str]:
    """Heuristic US-centric parser: returns street, city, state, zip. Ignores trailing country segment."""
    out = {"street": "", "city": "", "state": "", "zip": ""}
    if not addr:
        return out
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if len(parts) >= 2:
        last = parts[-1]
        # Drop trailing country like 'USA' or 'United States'
        if last.upper() in {"USA", "US", "UNITED STATES"} or (last.isalpha() and len(last) > 2):
            parts = parts[:-1]
    if len(parts) >= 3:
        out["street"] = ", ".join(parts[:-2])
        out["city"] = parts[-2]
        last = parts[-1]
        import re
        m = re.search(r"([A-Z]{2})\s*,?\s*(\d{5})(?:-\d{4})?", last, flags=re.I)
        if m:
            out["state"] = m.group(1).upper()
            out["zip"] = m.group(2)
        else:
            segs = last.split()
            if len(segs) >= 2:
                out["state"] = segs[0].upper()
                out["zip"] = segs[1]
            else:
                out["state"] = last.upper()
    else:
        out["street"] = addr
    return out

app = FastAPI(title="Fleet Prospect Finder - MVP (Places API)")

# CORS for local dev and typical ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PLACES_API_KEY = os.getenv("PLACES_API_KEY")
if not PLACES_API_KEY:
    # Don't crash app; raise on first use instead to make DX smooth
    pass

# Load category taxonomy with a simple reload helper so JSON edits don't require server restart
CATEGORY_PACKS = load_category_packs()
CATEGORY_PACKS_BY_KEY: Dict[str, CategoryPack] = {p.key: p for p in CATEGORY_PACKS}

def reload_categories() -> None:
    global CATEGORY_PACKS, CATEGORY_PACKS_BY_KEY
    packs = load_category_packs()
    CATEGORY_PACKS = packs
    CATEGORY_PACKS_BY_KEY = {p.key: p for p in packs}

# Minimal field mask per PRD (plus pureServiceAreaBusiness when present)
FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.formattedAddress,"
    "places.location,"
    "places.types,"
    "places.primaryType,"
    "places.businessStatus,"
    "places.googleMapsUri,"
    "places.rating,"
    "places.userRatingCount,"
    "places.pureServiceAreaBusiness"
)

# In-memory cache for search responses (20 minute TTL)
_CACHE_TTL_SECONDS = 20 * 60
_SEARCH_CACHE: Dict[Tuple[float, float, int, Tuple[str, ...], bool], Tuple[float, Dict[str, Any]]] = {}


def _encode_next_token(origin: str, upstream_token: str) -> str:
    """Prefix-origin encode to avoid double-tries. origin in {'n','t'} for nearby/text."""
    if origin not in {"n", "t"}:
        raise ValueError("invalid origin")
    return f"{origin}:{upstream_token}"


def _decode_next_token(token: str) -> Tuple[str, str]:
    """Return (origin, upstream_token). If no prefix, raise ValueError."""
    if not token or ":" not in token:
        raise ValueError("Invalid nextPageToken format")
    origin, rest = token.split(":", 1)
    if origin not in {"n", "t"} or not rest:
        raise ValueError("Invalid nextPageToken format")
    return origin, rest

def _build_cache_key(center_lat: float, center_lng: float, radius_meters: int, categories: List[str], high_recall: bool) -> Tuple[float, float, int, Tuple[str, ...], bool]:
    # Round lat/lng to avoid overly granular keys; 5 decimals ~1.1 meters
    lat_r = round(center_lat, 5)
    lng_r = round(center_lng, 5)
    cats = tuple(sorted(categories or []))
    return (lat_r, lng_r, int(radius_meters), cats, bool(high_recall))

def _cache_get(key: Tuple[float, float, int, Tuple[str, ...], bool]) -> Optional[Dict[str, Any]]:
    import time
    now = time.time()
    # Opportunistic prune and fetch
    stale: List[Tuple] = []
    if _SEARCH_CACHE:
        for k, (ts, _val) in list(_SEARCH_CACHE.items()):
            if now - ts > _CACHE_TTL_SECONDS:
                stale.append(k)
    for k in stale:
        _SEARCH_CACHE.pop(k, None)
    entry = _SEARCH_CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if now - ts > _CACHE_TTL_SECONDS:
        _SEARCH_CACHE.pop(key, None)
        return None
    return val

def _cache_set(key: Tuple[float, float, int, Tuple[str, ...], bool], value: Dict[str, Any]) -> None:
    import time
    _SEARCH_CACHE[key] = (time.time(), value)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/search/places", response_model=SearchResponse)
async def search_places(req: SearchRequest) -> SearchResponse:
    # Ensure latest taxonomy (no server restart required after editing data/categories.json)
    reload_categories()
    if not PLACES_API_KEY:
        raise HTTPException(status_code=500, detail="PLACES_API_KEY not configured")

    client = PlacesClient(api_key=PLACES_API_KEY, field_mask=FIELD_MASK)

    # Resolve center
    center: Center
    if req.center.text is not None:
        center = Center(text=req.center.text)
    elif req.center.lat is not None and req.center.lng is not None:
        center = Center(lat=req.center.lat, lng=req.center.lng)
    else:
        raise HTTPException(status_code=400, detail="Invalid center: provide either text or lat/lng")

    # Resolve to numeric coordinates for strict radius filtering
    try:
        center_geo = await client.resolve_center(center)
        center_lat, center_lng = center_geo["latitude"], center_geo["longitude"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to resolve center: {e}")

    # Cache lookup (keyed by resolved center, radius, packs, highRecall)
    cache_key = _build_cache_key(center_lat, center_lng, req.radiusMeters, req.categories, req.highRecall or False)
    cached = _cache_get(cache_key)
    if cached is not None:
        # Compose response using cached payload
        cached_resp = SearchResponse(
            results=cached.get("results", []),
            nextPageToken=cached.get("nextPageToken"),
            centerLat=center_lat,
            centerLng=center_lng,
        )
        return JSONResponse(content=cached_resp.model_dump(), headers={"X-Cache": "HIT"})

    # Consolidated execution: single Nearby and single Text where applicable
    results_by_id: Dict[str, PlaceLite] = {}
    max_results = req.maxResults or 60

    # Aggregate types and keywords from selected packs
    all_included_types: List[str] = []
    all_keywords: List[str] = []
    pack_labels_by_type: Dict[str, List[str]] = {}
    pack_labels = []
    for key in req.categories:
        pack = CATEGORY_PACKS_BY_KEY.get(key)
        if not pack:
            raise HTTPException(status_code=400, detail=f"Unknown category pack: {key}")
        pack_labels.append(pack.label)
        for t in (pack.includedTypes or []):
            all_included_types.append(t)
            pack_labels_by_type.setdefault(t, []).append(pack.label)
        for kw in (pack.keywords or []):
            all_keywords.append(kw)

    # Deduplicate
    all_included_types = sorted(list({t for t in all_included_types}))
    all_keywords = sorted(list({k for k in all_keywords}))

    next_nearby_token: Optional[str] = None
    next_text_token: Optional[str] = None

    # Nearby combined
    if all_included_types:
        nearby_resp = await client.search_nearby(
            center=center,
            radius_meters=req.radiusMeters,
            included_types=all_included_types,
            max_result_count=min(15, max_results),
        )
        for r in nearby_resp.results:
            existing = results_by_id.get(r.placeId)
            # Tag packs by matching primary type to includedTypes mapping; fallback add all selected if unsure
            labels = []
            if r.primaryType and r.primaryType in pack_labels_by_type:
                labels = pack_labels_by_type.get(r.primaryType, [])
            elif r.types:
                # If any of result types are in our includedTypes mapping, collect their labels
                lbl_set = set()
                for tp in r.types:
                    for lbl in pack_labels_by_type.get(tp, []):
                        lbl_set.add(lbl)
                labels = sorted(lbl_set)
            else:
                labels = []
            if existing:
                for lbl in labels:
                    if lbl not in (existing.categories or []):
                        existing.categories.append(lbl)
            else:
                r.categories = labels
                results_by_id[r.placeId] = r
        if nearby_resp.next_page_token:
            next_nearby_token = nearby_resp.next_page_token

    # Text combined
    if all_keywords:
        seg = " OR ".join(all_keywords)
        text_resp = await client.search_text(
            text_query=seg,
            center=center,
            radius_meters=req.radiusMeters,
            max_result_count=min(15, max_results),
        )
        for r in text_resp.results:
            existing = results_by_id.get(r.placeId)
            # Heuristic tagging: if primaryType maps to any includedTypes we set labels, otherwise leave empty.
            labels = []
            if r.primaryType and r.primaryType in pack_labels_by_type:
                labels = pack_labels_by_type.get(r.primaryType, [])
            elif r.types:
                lbl_set = set()
                for tp in r.types:
                    for lbl in pack_labels_by_type.get(tp, []):
                        lbl_set.add(lbl)
                labels = sorted(lbl_set)
            if existing:
                for lbl in labels:
                    if lbl not in (existing.categories or []):
                        existing.categories.append(lbl)
            else:
                r.categories = labels
                results_by_id[r.placeId] = r
        if text_resp.next_page_token:
            next_text_token = text_resp.next_page_token

    # Recall boost remains optional; only run if explicitly requested
    try:
        AUTO_RECALL_KEYS = {
            "auto_traditional",  # general auto repair
            "quick_lube",
            "tire_shops",
            "auto_glass",
            "body_collision",
        }
        if req.highRecall and any(k in AUTO_RECALL_KEYS for k in req.categories):
            boost_terms = [
                "auto repair",
                "mechanic",
                "brake repair",
                "muffler",
                "transmission repair",
                "oil change",
                "engine repair",
                "tire service",
                "alignment",
            ]
            boost_query = " OR ".join(boost_terms)
            boost_resp = await client.search_text(
                text_query=boost_query,
                center=center,
                radius_meters=req.radiusMeters,
                max_result_count=min(10, max_results),
            )
            for r in boost_resp.results:
                existing = results_by_id.get(r.placeId)
                if existing:
                    # Tag with a generic category label if not already tagged
                    if "TRADITIONAL AUTO" not in (existing.categories or []):
                        existing.categories.append("TRADITIONAL AUTO")
                else:
                    r.categories = ["TRADITIONAL AUTO"]
                    results_by_id[r.placeId] = r
            # For simplicity, do not propagate boost pagination to reduce costs
    except Exception:
        # Boost is best-effort; do not fail the request if it errors
        pass

    # No server-side auto-pagination; leave to client 'Load More'

    merged_list = list(results_by_id.values())

    # Strict radius enforcement: drop any results outside radiusMeters from the resolved center
    radius_m = max(1, req.radiusMeters)
    in_radius: List[PlaceLite] = []
    for r in merged_list:
        if r.lat is None or r.lng is None:
            # Strict enforcement: drop if we cannot compute distance
            continue
        d = _haversine_meters(center_lat, center_lng, r.lat, r.lng)
        if d <= radius_m:
            in_radius.append(r)

    # Apply residential/home-based exclusion if requested (default True per PRD)
    filtered = apply_residential_filter(in_radius, exclude_service_area_only=req.excludeServiceAreaOnly)

    # Truncate to max_results
    filtered = filtered[:max_results]

    # Expose a composite nextPageToken encoding origin to prevent double-tries. Prefer Nearby token, else Text token.
    next_token = None
    if next_nearby_token:
        next_token = _encode_next_token("n", next_nearby_token)
    elif next_text_token:
        next_token = _encode_next_token("t", next_text_token)

    resp = SearchResponse(results=filtered, nextPageToken=next_token, centerLat=center_lat, centerLng=center_lng)

    # Store in cache
    _cache_set(cache_key, {"results": resp.results, "nextPageToken": resp.nextPageToken})

    headers = {
        "X-Cache": "MISS",
        "X-Places-Search-Nearby": str(client.metrics.get("search_nearby_calls", 0)),
        "X-Places-Search-Text": str(client.metrics.get("search_text_calls", 0)),
        "X-Places-Details": str(client.metrics.get("details_calls", 0)),
        "X-Places-Center-Cache-Hits": str(client.metrics.get("center_cache_hits", 0)),
        "X-Places-Center-Cache-Misses": str(client.metrics.get("center_cache_misses", 0)),
    }
    return JSONResponse(content=resp.model_dump(), headers=headers)

@app.get("/search/places/next", response_model=SearchResponse)
async def search_places_next(token: str = Query(..., description="Composite nextPageToken returned from /search/places")) -> SearchResponse:
    reload_categories()
    if not PLACES_API_KEY:
        raise HTTPException(status_code=500, detail="PLACES_API_KEY not configured")

    client = PlacesClient(api_key=PLACES_API_KEY, field_mask=FIELD_MASK)

    try:
        origin, upstream = _decode_next_token(token)
        origin_name = "text" if origin == "t" else "nearby"
        resp = await client.fetch_next_page(origin=origin_name, next_page_token=upstream)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Return composite token again if present
    composite = None
    if resp.next_page_token:
        composite = _encode_next_token("t" if origin_name == "text" else "n", resp.next_page_token)
    headers = {
        "X-Places-Search-Nearby": str(client.metrics.get("search_nearby_calls", 0)),
        "X-Places-Search-Text": str(client.metrics.get("search_text_calls", 0)),
    }
    return JSONResponse(content=SearchResponse(results=resp.results, nextPageToken=composite).model_dump(), headers=headers)

# New: Categories endpoint for frontend selector
@app.get("/categories")
def get_categories() -> List[Dict[str, Any]]:
    reload_categories()
    return [{"key": p.key, "label": p.label, "strategy": p.strategy, "includedTypes": p.includedTypes, "keywords": p.keywords} for p in CATEGORY_PACKS]

# New: CSV export endpoint
@app.post("/search/places/csv")
async def search_places_csv(
    req: SearchRequest = Body(...),
    filterPrimaryTypes: Optional[List[str]] = Query(None),
):
    reload_categories()
    import csv
    import io

    resp = await search_places(req)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "placeId",
        "name",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "lng",
        "primaryType",
        "categories",
        "types",
        "businessStatus",
        "googleMapsUri",
        "pureServiceAreaBusiness",
    ])
    rows = resp.results
    # Optional filter by primaryType
    if filterPrimaryTypes:
        # allow repeated query params or comma-separated list
        values: List[str] = []
        for v in filterPrimaryTypes:
            if "," in v:
                values.extend([s.strip() for s in v.split(",") if s.strip()])
            else:
                values.append(v)
        allow = set(values)
        rows = [r for r in rows if (r.primaryType in allow)]
    for r in rows:
        ap = _parse_address_parts(r.formattedAddress or "")
        writer.writerow([
            r.placeId,
            r.name,
            ap.get("street", ""),
            ap.get("city", ""),
            ap.get("state", ""),
            ap.get("zip", ""),
            r.lat if r.lat is not None else "",
            r.lng if r.lng is not None else "",
            r.primaryType or "",
            ";".join(r.categories or []),
            ";".join(r.types or []),
            r.businessStatus or "",
            r.googleMapsUri or "",
            r.pureServiceAreaBusiness if r.pureServiceAreaBusiness is not None else "",
        ])

    output.seek(0)
    headers = {"Content-Disposition": "attachment; filename=places_export.csv"}
    return StreamingResponse(output, media_type="text/csv", headers=headers)

# New: Place details enrichment (phone, website) for a list of placeIds
@app.post("/places/details")
async def places_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not PLACES_API_KEY:
        raise HTTPException(status_code=500, detail="PLACES_API_KEY not configured")
    ids = payload.get("placeIds") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="placeIds array required")
    # Cap to 50 to limit cost/time
    ids = ids[:50]
    client = PlacesClient(api_key=PLACES_API_KEY, field_mask=FIELD_MASK)

    out: Dict[str, Any] = {}
    for pid in ids:
        try:
            data = await client.get_place_details(pid)
            out[pid] = {
                "phone": data.get("nationalPhoneNumber") or data.get("internationalPhoneNumber"),
                "website": data.get("websiteUri"),
            }
        except Exception:
            out[pid] = {"phone": None, "website": None}
    headers = {
        "X-Places-Details": str(client.metrics.get("details_calls", 0))
    }
    return JSONResponse(content={"details": out}, headers=headers)

# Static frontend serving (app/web)
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not built. API is running."}
