from __future__ import annotations

import json
from typing import List, Optional, Dict, Any, Union

import httpx
from pydantic import BaseModel

from app.models.schemas import PlaceLite, ClientSearchResponse

PLACES_BASE = "https://places.googleapis.com/v1"


class Center(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    text: Optional[str] = None


def _map_place_to_lite(place: Dict[str, Any]) -> PlaceLite:
    loc = place.get("location", {})
    display_name = place.get("displayName", {})
    return PlaceLite(
        placeId=place.get("id"),
        name=display_name.get("text") if isinstance(display_name, dict) else display_name,
        formattedAddress=place.get("formattedAddress"),
        lat=loc.get("latitude"),
        lng=loc.get("longitude"),
        primaryType=place.get("primaryType"),
        types=place.get("types", []) or [],
        businessStatus=place.get("businessStatus"),
        rating=place.get("rating"),
        userRatingCount=place.get("userRatingCount"),
        googleMapsUri=place.get("googleMapsUri"),
        pureServiceAreaBusiness=place.get("pureServiceAreaBusiness"),
    )


class PlacesClient:
    def __init__(self, api_key: str, field_mask: str) -> None:
        self.api_key = api_key
        self.field_mask = field_mask
        self._client = httpx.AsyncClient(timeout=20.0)
        # Simple in-memory cache for center text -> lat/lng with TTL
        self._center_cache: Dict[str, Dict[str, Any]] = {}
        self._center_cache_ttl_seconds = 60 * 30  # 30 minutes
        # Lightweight metrics
        self.metrics: Dict[str, int] = {
            "search_text_calls": 0,
            "search_nearby_calls": 0,
            "details_calls": 0,
            "center_cache_hits": 0,
            "center_cache_misses": 0,
        }

    async def _post(self, path: str, json_body: Dict[str, Any], *, field_mask: Optional[str] = None) -> Dict[str, Any]:
        headers = {
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask or self.field_mask,
            "Content-Type": "application/json",
        }
        url = f"{PLACES_BASE}/{path}"
        resp = await self._client.post(url, headers=headers, json=json_body)
        # Raise detailed error if not ok
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = {"text": resp.text}
            raise httpx.HTTPStatusError(f"Places API error {resp.status_code}: {detail}", request=resp.request, response=resp)
        return resp.json()

    async def search_nearby(
        self,
        center: Center,
        radius_meters: int,
        included_types: List[str],
        max_result_count: int = 20,
    ) -> ClientSearchResponse:
        if center.text:
            geo = await self._resolve_center_text(center.text)
            lat, lng = geo["latitude"], geo["longitude"]
        else:
            if center.lat is None or center.lng is None:
                raise ValueError("center requires text or lat/lng")
            lat, lng = center.lat, center.lng

        body = {
            "includedTypes": included_types,
            "maxResultCount": max_result_count,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": radius_meters,
                }
            },
        }
        self.metrics["search_nearby_calls"] += 1
        data = await self._post("places:searchNearby", body)
        places = data.get("places", [])
        next_token = data.get("nextPageToken") or data.get("next_page_token")
        results = [_map_place_to_lite(p) for p in places]
        return ClientSearchResponse(results=results, next_page_token=next_token)

    async def search_text(
        self,
        text_query: str,
        center: Center,
        radius_meters: int,
        max_result_count: int = 20,
    ) -> ClientSearchResponse:
        location_bias: Optional[Dict[str, Any]] = None
        if center.text:
            geo = await self._resolve_center_text(center.text)
            location_bias = {
                "circle": {
                    "center": {"latitude": geo["latitude"], "longitude": geo["longitude"]},
                    "radius": radius_meters,
                }
            }
        elif center.lat is not None and center.lng is not None:
            location_bias = {
                "circle": {
                    "center": {"latitude": center.lat, "longitude": center.lng},
                    "radius": radius_meters,
                }
            }

        body = {
            "textQuery": text_query,
        }
        if location_bias:
            body["locationBias"] = {"circle": location_bias["circle"]}
        # The API caps results per page; maxResultCount can be included for Text as well
        body["maxResultCount"] = max_result_count

        self.metrics["search_text_calls"] += 1
        data = await self._post("places:searchText", body)
        places = data.get("places", [])
        next_token = data.get("nextPageToken") or data.get("next_page_token")
        results = [_map_place_to_lite(p) for p in places]
        return ClientSearchResponse(results=results, next_page_token=next_token)

    async def fetch_next_page(self, origin: str, next_page_token: str) -> ClientSearchResponse:
        """Fetch next page for a known origin ('text' or 'nearby') without double-trying."""
        if origin not in ("text", "nearby"):
            raise ValueError("origin must be 'text' or 'nearby'")
        path = "places:searchText" if origin == "text" else "places:searchNearby"
        if origin == "text":
            self.metrics["search_text_calls"] += 1
        else:
            self.metrics["search_nearby_calls"] += 1
        data = await self._post(path, {"pageToken": next_page_token})
        places = data.get("places", [])
        next_token = data.get("nextPageToken") or data.get("next_page_token")
        results = [_map_place_to_lite(p) for p in places]
        return ClientSearchResponse(results=results, next_page_token=next_token)

    async def _resolve_center_text(self, text: str) -> Dict[str, float]:
        """Resolve free-form text to lat/lng using minimal field mask and a short-lived cache."""
        key = text.strip()
        if not key:
            raise ValueError("center text is empty")
        import time
        now = time.time()
        cached = self._center_cache.get(key)
        if cached and (now - cached.get("ts", 0) < self._center_cache_ttl_seconds):
            self.metrics["center_cache_hits"] += 1
            return {"latitude": cached["lat"], "longitude": cached["lng"]}

        body = {
            "textQuery": text,
            "maxResultCount": 1,
        }
        # Request only the minimal field(s) needed
        self.metrics["center_cache_misses"] += 1
        self.metrics["search_text_calls"] += 1
        data = await self._post("places:searchText", body, field_mask="places.location")
        places = data.get("places", [])
        if not places:
            raise ValueError("Unable to resolve center text to location")
        loc = places[0].get("location", {})
        lat, lng = loc.get("latitude"), loc.get("longitude")
        if lat is None or lng is None:
            raise ValueError("Resolved place has no location")
        self._center_cache[key] = {"lat": lat, "lng": lng, "ts": now}
        return {"latitude": lat, "longitude": lng}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def resolve_center(self, center: Center) -> Dict[str, float]:
        """Public helper to resolve a Center to coordinates {latitude, longitude}."""
        if center.text:
            return await self._resolve_center_text(center.text)
        if center.lat is None or center.lng is None:
            raise ValueError("center requires text or lat/lng")
        return {"latitude": center.lat, "longitude": center.lng}

    async def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """
        Fetch limited details for a place: phone and website.
        Note: These fields may require appropriate Places API plan/quotas.
        """
        headers = {
            "X-Goog-Api-Key": self.api_key,
            # Dedicated field mask for details
            "X-Goog-FieldMask": "id,nationalPhoneNumber,internationalPhoneNumber,websiteUri",
        }
        url = f"{PLACES_BASE}/places/{place_id}"
        self.metrics["details_calls"] += 1
        resp = await self._client.get(url, headers=headers)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = {"text": resp.text}
            raise httpx.HTTPStatusError(f"Places Details error {resp.status_code}: {detail}", request=resp.request, response=resp)
        return resp.json()
