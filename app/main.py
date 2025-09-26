import os
import json
import sqlite3
from threading import RLock
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
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

# --- Disk cache (SQLite) setup ---
_CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "cache.db")
_DB_LOCK = RLock()

def _ensure_cache_db() -> None:
    with _DB_LOCK:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    ns TEXT NOT NULL,
                    key TEXT NOT NULL,
                    ts REAL NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (ns, key)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

def _disk_cache_get(ns: str, key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    import time
    now = time.time()
    with _DB_LOCK:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        try:
            cur = conn.execute("SELECT ts, value FROM cache_entries WHERE ns=? AND key=?", (ns, key))
            row = cur.fetchone()
            if not row:
                return None
            ts, value_json = row
            if now - float(ts) > ttl_seconds:
                # prune stale
                conn.execute("DELETE FROM cache_entries WHERE ns=? AND key=?", (ns, key))
                conn.commit()
                return None
            try:
                return json.loads(value_json)
            except Exception:
                return None
        finally:
            conn.close()

def _disk_cache_set(ns: str, key: str, value: Dict[str, Any]) -> None:
    import time
    with _DB_LOCK:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        try:
            conn.execute(
                "REPLACE INTO cache_entries (ns, key, ts, value) VALUES (?, ?, ?, ?)",
                (ns, key, time.time(), json.dumps(value))
            )
            conn.commit()
        finally:
            conn.close()

_ensure_cache_db()


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
COMBINED_MODE = os.getenv("COMBINED_MODE", "").strip().lower() in {"1","true","yes","on"}
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
    "places.pureServiceAreaBusiness"
)

# In-memory cache for search responses (20 minute TTL)
_CACHE_TTL_SECONDS = 20 * 60
_SEARCH_CACHE: Dict[Tuple[float, float, int, Tuple[str, ...], bool], Tuple[float, Dict[str, Any]]] = {}

# In-memory cache for resolving center.text -> (lat,lng) (24 hour TTL)
_CENTER_TTL_SECONDS = 24 * 60 * 60
_CENTER_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}

# In-memory cache for place details (phone) to avoid repeated paid lookups (7-day TTL)
_DETAILS_TTL_SECONDS = 7 * 24 * 60 * 60
_DETAILS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

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
    if entry:
        ts, val = entry
        if now - ts <= _CACHE_TTL_SECONDS:
            return val
        _SEARCH_CACHE.pop(key, None)
    # Fallback to disk cache
    disk_key = json.dumps({
        "lat": key[0], "lng": key[1], "radius": key[2], "cats": list(key[3]), "hr": key[4]
    }, sort_keys=True)
    disk_val = _disk_cache_get("search", disk_key, _CACHE_TTL_SECONDS)
    if disk_val is not None:
        return disk_val
    return None

def _cache_set(key: Tuple[float, float, int, Tuple[str, ...], bool], value: Dict[str, Any]) -> None:
    import time
    _SEARCH_CACHE[key] = (time.time(), value)
    # Write-through to disk
    disk_key = json.dumps({
        "lat": key[0], "lng": key[1], "radius": key[2], "cats": list(key[3]), "hr": key[4]
    }, sort_keys=True)
    _disk_cache_set("search", disk_key, value)

def _center_cache_get(text: str) -> Optional[Dict[str, float]]:
    import time
    now = time.time()
    key_norm = text.strip().lower()
    entry = _CENTER_CACHE.get(key_norm)
    if not entry:
        # Try disk cache
        disk_val = _disk_cache_get("center", key_norm, _CENTER_TTL_SECONDS)
        if disk_val and "latitude" in disk_val and "longitude" in disk_val:
            return disk_val
        return None
    ts, val = entry
    if now - ts > _CENTER_TTL_SECONDS:
        _CENTER_CACHE.pop(key_norm, None)
        # Also remove from disk
        return None
    return val

def _center_cache_set(text: str, lat: float, lng: float) -> None:
    import time
    key = text.strip().lower()
    val = {"latitude": lat, "longitude": lng}
    _CENTER_CACHE[key] = (time.time(), val)
    _disk_cache_set("center", key, val)

def _details_cache_get(place_id: str) -> Optional[Dict[str, Any]]:
    import time
    now = time.time()
    entry = _DETAILS_CACHE.get(place_id)
    if not entry:
        # Try disk cache
        disk_val = _disk_cache_get("details", place_id, _DETAILS_TTL_SECONDS)
        if disk_val:
            return disk_val
        return None
    ts, val = entry
    if now - ts > _DETAILS_TTL_SECONDS:
        _DETAILS_CACHE.pop(place_id, None)
        return None
    return val

def _details_cache_set(place_id: str, data: Dict[str, Any]) -> None:
    import time
    _DETAILS_CACHE[place_id] = (time.time(), data)
    _disk_cache_set("details", place_id, data)

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

    # Resolve to numeric coordinates for strict radius filtering (with cache for center.text)
    try:
        if center.text:
            cached_center = _center_cache_get(center.text)
            if cached_center:
                center_lat, center_lng = cached_center["latitude"], cached_center["longitude"]
            else:
                center_geo = await client.resolve_center(center)
                center_lat, center_lng = center_geo["latitude"], center_geo["longitude"]
                _center_cache_set(center.text, center_lat, center_lng)
        else:
            center_geo = await client.resolve_center(center)
            center_lat, center_lng = center_geo["latitude"], center_geo["longitude"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to resolve center: {e}")

    # Cache lookup (keyed by resolved center, radius, packs, highRecall)
    cache_key = _build_cache_key(center_lat, center_lng, req.radiusMeters, req.categories, req.highRecall or False)
    cached = _cache_get(cache_key)
    if cached is not None:
        # Compose response using cached payload
        return SearchResponse(
            results=cached.get("results", []),
            nextPageToken=cached.get("nextPageToken"),
            centerLat=center_lat,
            centerLng=center_lng,
        )

    # Execute search
    results_by_id: Dict[str, PlaceLite] = {}
    paginate_queue: List[tuple[str, str]] = []  # (next_page_token, pack_label or "")

    max_results = req.maxResults or 60

    def _label_by_types(pl: PlaceLite, packs: List[CategoryPack]) -> List[str]:
        labels: List[str] = []
        ptype = (pl.primaryType or "").lower()
        types_set = set((pl.types or []))
        for pk in packs:
            inc = [t.lower() for t in (pk.includedTypes or [])]
            if not inc:
                continue
            if ptype and ptype in inc:
                labels.append(pk.label)
                continue
            if any((t.lower() in inc) for t in types_set):
                labels.append(pk.label)
        return labels

    selected_packs: List[CategoryPack] = []
    for key in req.categories:
        pack = CATEGORY_PACKS_BY_KEY.get(key)
        if not pack:
            raise HTTPException(status_code=400, detail=f"Unknown category pack: {key}")
        selected_packs.append(pack)

    if COMBINED_MODE:
        # Combined Nearby: union of includedTypes
        combined_types = sorted({t for p in selected_packs for t in (p.includedTypes or [])})
        if combined_types:
            # If too many, shard into small batches to remain specific
            batch_size = 12
            for i in range(0, len(combined_types), batch_size):
                batch = combined_types[i:i+batch_size]
                nearby_resp = await client.search_nearby(
                    center=center,
                    radius_meters=req.radiusMeters,
                    included_types=batch,
                    max_result_count=min(20, max_results),
                )
                for r in nearby_resp.results:
                    existing = results_by_id.get(r.placeId)
                    labels = _label_by_types(r, selected_packs)
                    if existing:
                        for lbl in labels:
                            if lbl not in (existing.categories or []):
                                existing.categories.append(lbl)
                    else:
                        r.categories = labels
                        results_by_id[r.placeId] = r
                if nearby_resp.next_page_token:
                    paginate_queue.append((nearby_resp.next_page_token, ""))

        # Combined Text: merged OR query, sharded by length
        combined_keywords = sorted({kw for p in selected_packs for kw in (p.keywords or [])})
        if combined_keywords:
            # Build batches under a conservative length budget
            cur_batch: List[str] = []
            cur_len = 0
            batches: List[str] = []
            for kw in combined_keywords:
                seg = (" OR " if cur_batch else "") + kw
                if cur_len + len(seg) > 512:  # conservative cap
                    if cur_batch:
                        batches.append(" OR ".join(cur_batch))
                    cur_batch = [kw]
                    cur_len = len(kw)
                else:
                    cur_batch.append(kw)
                    cur_len += len(seg)
            if cur_batch:
                batches.append(" OR ".join(cur_batch))

            for seg in batches:
                text_resp = await client.search_text(
                    text_query=seg,
                    center=center,
                    radius_meters=req.radiusMeters,
                    max_result_count=min(20, max_results),
                )
                for r in text_resp.results:
                    existing = results_by_id.get(r.placeId)
                    labels = _label_by_types(r, selected_packs)
                    if existing:
                        for lbl in labels:
                            if lbl not in (existing.categories or []):
                                existing.categories.append(lbl)
                    else:
                        r.categories = labels
                        results_by_id[r.placeId] = r
                if text_resp.next_page_token:
                    paginate_queue.append((text_resp.next_page_token, ""))
    else:
        # Original per-pack behavior
        for key in req.categories:
            pack = CATEGORY_PACKS_BY_KEY.get(key)
            pack_label = pack.label
            if pack.includedTypes:
                nearby_resp = await client.search_nearby(
                    center=center,
                    radius_meters=req.radiusMeters,
                    included_types=pack.includedTypes,
                    max_result_count=min(20, max_results),
                )
                for r in nearby_resp.results:
                    existing = results_by_id.get(r.placeId)
                    if existing:
                        if pack_label not in (existing.categories or []):
                            existing.categories.append(pack_label)
                    else:
                        r.categories = [pack_label]
                        results_by_id[r.placeId] = r
                if nearby_resp.next_page_token:
                    paginate_queue.append((nearby_resp.next_page_token, pack_label))

            if pack.keywords:
                seg = " OR ".join(pack.keywords)
                text_resp = await client.search_text(
                    text_query=seg,
                    center=center,
                    radius_meters=req.radiusMeters,
                    max_result_count=min(20, max_results),
                )
                for r in text_resp.results:
                    existing = results_by_id.get(r.placeId)
                    if existing:
                        if pack_label not in (existing.categories or []):
                            existing.categories.append(pack_label)
                    else:
                        r.categories = [pack_label]
                        results_by_id[r.placeId] = r
                if text_resp.next_page_token:
                    paginate_queue.append((text_resp.next_page_token, pack_label))

    # Recall boost: If auto-repair related packs are selected and highRecall is on, run an extra targeted text search and merge
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
                max_result_count=min(20, max_results),
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
            if boost_resp.next_page_token:
                paginate_queue.append((boost_resp.next_page_token, "TRADITIONAL AUTO"))
    except Exception:
        # Boost is best-effort; do not fail the request if it errors
        pass

    # High-recall pagination: fetch additional pages round-robin across all queued next_page_tokens
    if req.highRecall and paginate_queue:
        try:
            # Round-robin until max_results or tokens exhausted
            idx = 0
            while len(results_by_id) < max_results and paginate_queue:
                token, label = paginate_queue.pop(0)
                try:
                    page = await client.fetch_next_page(next_page_token=token)
                except Exception:
                    continue
                for r in page.results:
                    existing = results_by_id.get(r.placeId)
                    if existing:
                        if label and label not in (existing.categories or []):
                            existing.categories.append(label)
                    else:
                        r.categories = [label] if label else []
                        results_by_id[r.placeId] = r
                if page.next_page_token:
                    paginate_queue.append((page.next_page_token, label))
                idx = (idx + 1) % (len(paginate_queue) or 1)
        except Exception:
            # Don't fail the request if pagination fails
            pass

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

    # For compatibility, still expose the first available token if any
    next_token = paginate_queue[0][0] if paginate_queue else None

    resp = SearchResponse(results=filtered, nextPageToken=next_token, centerLat=center_lat, centerLng=center_lng)

    # Store in cache
    _cache_set(cache_key, {"results": resp.results, "nextPageToken": resp.nextPageToken})

    return resp

@app.get("/search/places/next", response_model=SearchResponse)
async def search_places_next(token: str = Query(..., description="Upstream Places API nextPageToken")) -> SearchResponse:
    reload_categories()
    if not PLACES_API_KEY:
        raise HTTPException(status_code=500, detail="PLACES_API_KEY not configured")

    client = PlacesClient(api_key=PLACES_API_KEY, field_mask=FIELD_MASK)

    try:
        resp = await client.fetch_next_page(next_page_token=token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # No filtering or merging on next page alone; just pass through and let client apply filters client-side if needed
    return SearchResponse(results=resp.results, nextPageToken=resp.next_page_token)

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
            cached = _details_cache_get(pid)
            if cached is not None:
                data = cached
            else:
                data = await client.get_place_details(pid)
                _details_cache_set(pid, data)
            out[pid] = {
                "phone": data.get("nationalPhoneNumber") or data.get("internationalPhoneNumber"),
            }
        except Exception:
            out[pid] = {"phone": None}
    return {"details": out}

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
