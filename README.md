# Fleet Prospect Finder — MVP (Search-Only, Places API)

Backend service implementing the MVP from the PRD using Google Places API (New). Provides a FastAPI service to search by area and category packs, with a soft residential/home-based exclusion toggle, and pagination token passthrough.

## Stack
- FastAPI
- httpx (async)
- Pydantic v2
- Uvicorn

## Setup

1) Prereqs
- Python 3.11+
- A Google Places API key with access to Places API (New)

2) Clone and configure
- Create a `.env` file at the project root based on `.env.example` and set `PLACES_API_KEY`.

3) Install deps
```bash
python -m venv .venv
. .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4) Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

## Configuration

- Environment variable: `PLACES_API_KEY` (in `.env`)
- Category taxonomy: `data/categories.json`
- Field mask used (cost control):
```
places.id,places.displayName,places.formattedAddress,places.location,places.types,places.primaryType,places.businessStatus,places.googleMapsUri,places.rating,places.userRatingCount,places.pureServiceAreaBusiness
```

## API

### POST /search/places
Body:
```json
{
  "center": {"text": "Detroit, MI"},
  "radiusMeters": 10000,
  "categories": ["auto_traditional", "plumbers"],
  "excludeServiceAreaOnly": true,
  "maxResults": 50
}
```
Alternative center:
```json
{"center": {"lat": 42.3314, "lng": -83.0458}}
```
Response:
```json
{
  "results": [
    {
      "placeId": "...",
      "name": "...",
      "formattedAddress": "...",
      "lat": 42.3,
      "lng": -83.0,
      "primaryType": "car_repair",
      "types": ["car_repair", "point_of_interest", "establishment"],
      "businessStatus": "OPERATIONAL",
      "rating": 4.4,
      "userRatingCount": 78,
      "googleMapsUri": "https://...",
      "pureServiceAreaBusiness": false
    }
  ],
  "nextPageToken": "..."
}
```

### GET /search/places/next?token=...
Pass through a Google `nextPageToken` from a previous call. Returns the next page in the same shape as above.

## Category Packs
- Defined in `data/categories.json`
- Keys to use in requests (examples):
  - `auto_traditional`, `quick_lube`, `tire_shops`, `auto_glass`, `body_collision`, `car_wash`, `towing`, `dealers`
  - `plumbers`, `electricians`, `roofing`, `hvac`, `pest_control`, `locksmiths`, `landscaping`, `tree_service`, `painting`
  - `moving`, `courier`, `rental_fleets`
  - `hardware_supply`, `waste_dumpster`, `excavate_pave_concrete`

## Notes
- Nearby Search is used when `includedTypes` exist for selected packs. Text Search is used for keyword-only packs. Both can run in the same request.
- Residential/home-based exclusion:
  - Primary: exclude `pureServiceAreaBusiness == true` when present.
  - Heuristic: if only generic types, 0 reviews, and an address that looks residential, drop (toggle controlled by `excludeServiceAreaOnly`).
- We dedupe within a single request by `placeId`. Pagination across multiple upstream tokens is kept simple in MVP: we return only the first available token.

## Next Steps (Post-MVP polish)
- Optional Place Details fetch (phone/website) for selected rows.
- UI front-end and CSV export endpoint/utility.
- Phase 2: Matching & de-duplication against known accounts.
