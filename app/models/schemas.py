from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class CenterModel(BaseModel):
    lat: Optional[float] = Field(default=None)
    lng: Optional[float] = Field(default=None)
    text: Optional[str] = Field(default=None, description="Free-form location string like 'Detroit, MI' or a place name")

    @field_validator("text")
    @classmethod
    def at_least_one(cls, v, values):
        # Validation handled in endpoint for clearer HTTP error; allow flexible input here
        return v


class PlaceLite(BaseModel):
    placeId: str
    name: str
    formattedAddress: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    primaryType: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    businessStatus: Optional[str] = None
    rating: Optional[float] = None
    userRatingCount: Optional[int] = None
    googleMapsUri: Optional[str] = None
    pureServiceAreaBusiness: Optional[bool] = None
    categories: List[str] = Field(default_factory=list, description="Category packs matched (labels)")


class SearchRequest(BaseModel):
    center: CenterModel
    radiusMeters: int = Field(ge=1, le=50000)
    categories: List[str] = Field(default_factory=list, description="Category pack keys")
    excludeServiceAreaOnly: bool = Field(default=True)
    maxResults: Optional[int] = Field(default=60, ge=1, le=500)
    highRecall: Optional[bool] = Field(default=True, description="Enable deeper pagination and recall boosts for broader coverage")


class SearchResponse(BaseModel):
    results: List[PlaceLite]
    nextPageToken: Optional[str] = None
    centerLat: Optional[float] = None
    centerLng: Optional[float] = None


# Internal client response model for pagination handling
class ClientSearchResponse(BaseModel):
    results: List[PlaceLite]
    next_page_token: Optional[str] = None
