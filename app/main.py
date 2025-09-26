import os
import secrets
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, Body, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

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

# Basic Auth Configuration
USERNAME = os.getenv("AUTH_USERNAME", "admin")
PASSWORD = os.getenv("AUTH_PASSWORD", "changeme123")  # Change this in production!

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

app = FastAPI(title="Fleet Prospect Finder - MVP (Places API)")

# Security middleware
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Replace with your actual domains in production
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

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

# Minimal field mask - only what we need for basic business info
FIELD_MASK = (
    "places.id,"
    "places.displayName.text,"  # Business name
    "places.formattedAddress,"  # Full address
    "places.location.latitude,"  # Location coordinates
    "places.location.longitude,"
    "places.primaryType"  # Business category
)

# In-memory cache for search responses (20 minute TTL)
_CACHE_TTL_SECONDS = 20 * 60
_SEARCH_CACHE: Dict[Tuple[float, float, int, Tuple[str, ...], bool], Tuple[float, Dict[str, Any]]] = {}

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
        return SearchResponse(
            results=cached.get("results", []),
            nextPageToken=cached.get("nextPageToken"),
            centerLat=center_lat,
            centerLng=center_lng,
        )

    # Execute per selected category pack to tag results with pack labels
    results_by_id: Dict[str, PlaceLite] = {}
    # Store upstream pagination tokens together with their originating pack label
    paginate_queue: List[tuple[str, str]] = []  # (next_page_token, pack_label)

    max_results = req.maxResults or 60

    for key in req.categories:
        pack = CATEGORY_PACKS_BY_KEY.get(key)
        if not pack:
            raise HTTPException(status_code=400, detail=f"Unknown category pack: {key}")

        pack_label = pack.label

        # Nearby: use pack's includedTypes if any
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

        # Text Search: use pack's keywords if any
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

# Static frontend serving (app/web)
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Root endpoint with authentication
@app.get("/", response_class=HTMLResponse)
async def serve_index(username: str = Depends(verify_credentials)):
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

# Protect the existing endpoints by adding dependencies
@app.post("/search/places")
async def protected_search_places(
    req: SearchRequest, 
    username: str = Depends(verify_credentials)
):
    return await search_places(req)

@app.get("/search/next")
async def protected_search_places_next(
    token: str = Query(..., description="Upstream Places API nextPageToken"),
    username: str = Depends(verify_credentials)
):
    return await search_places_next(token)

@app.get("/api/categories")
async def protected_get_categories(username: str = Depends(verify_credentials)):
    return await get_categories()

@app.post("/export/csv")
async def protected_search_places_csv(
    req: SearchRequest = Body(...),
    filterPrimaryTypes: Optional[List[str]] = Query(None),
    username: str = Depends(verify_credentials)
):
    return await search_places_csv(req, filterPrimaryTypes)

# Health check endpoint (no auth required)
@app.get("/health")
async def health():
    return {"status": "ok"}

