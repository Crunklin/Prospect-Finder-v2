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

    async def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": self.field_mask,
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

        data = await self._post("places:searchText", body)
        places = data.get("places", [])
        next_token = data.get("nextPageToken") or data.get("next_page_token")
        results = [_map_place_to_lite(p) for p in places]
        return ClientSearchResponse(results=results, next_page_token=next_token)

    async def fetch_next_page(self, next_page_token: str) -> ClientSearchResponse:
        # Next page for both Nearby and Text uses places:search* with pageToken
        # We can't know which one produced it. We'll try text search endpoint first; if it fails, try nearby.
        for path in ("places:searchText", "places:searchNearby"):
            try:
                data = await self._post(path, {"pageToken": next_page_token})
                places = data.get("places", [])
                next_token = data.get("nextPageToken") or data.get("next_page_token")
                results = [_map_place_to_lite(p) for p in places]
                return ClientSearchResponse(results=results, next_page_token=next_token)
            except httpx.HTTPStatusError:
                continue
        raise ValueError("Invalid or expired nextPageToken")

    async def _resolve_center_text(self, text: str) -> Dict[str, float]:
        # Use Text Search to resolve center text to a lat/lng by searching for the text and taking the first result's location.
        body = {
            "textQuery": text,
            "maxResultCount": 1,
        }
        data = await self._post("places:searchText", body)
        places = data.get("places", [])
        if not places:
            raise ValueError("Unable to resolve center text to location")
        loc = places[0].get("location", {})
        return {"latitude": loc.get("latitude"), "longitude": loc.get("longitude")}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def resolve_center(self, center: Center) -> Dict[str, float]:
        """Public helper to resolve a Center to coordinates {latitude, longitude}."""
        if center.text:
            return await self._resolve_center_text(center.text)
        if center.lat is None or center.lng is None:
            raise ValueError("center requires text or lat/lng")
        return {"latitude": center.lat, "longitude": center.lng}

