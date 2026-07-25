"""
doctor_discovery.py

Shared Doctor Discovery service, per the outline's design: a simple
function call (NOT an agent) that maps a predicted/identified disease
to a medical specialty, then queries Google Places API for nearby
doctors/clinics matching that specialty. Shared across all three UCs --
UC1, UC2, and UC3 all pass their identified disease name to the same
find_nearby_doctors() function.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import requests

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# WHY A SIMPLE DICT, NOT AN LLM CALL:
# Disease-to-specialty mapping is a stable, well-known clinical fact --
# Migraine always maps to Neurologist, Arrhythmia always maps to
# Cardiologist. This doesn't need LLM judgment; it's a deterministic
# lookup, consistent with the project's broader "rule-based where
# possible" design philosophy (same reasoning as urgency_engine.py's
# disease-category floors in UC1).
DISEASE_TO_SPECIALTY = {
    # Respiratory
    "Bronchitis": "Pulmonologist",
    "Asthma": "Pulmonologist",
    "COPD": "Pulmonologist",
    "COVID-19": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Tuberculosis": "Pulmonologist",
    "Viral Infection": "General Physician",
    # Cardiac
    "Arrhythmia": "Cardiologist",
    "Heart Failure": "Cardiologist",
    "Hypertensive Crisis": "Cardiologist",
    # Metabolic
    "Type 2 Diabetes": "Endocrinologist",
    "Hypothyroidism": "Endocrinologist",
    "Anaemia": "Hematologist",
    # Infectious
    "Dengue Fever": "General Physician",
    "Typhoid": "General Physician",
    "Malaria": "General Physician",
    "Gastroenteritis": "Gastroenterologist",
    "Food Poisoning": "Gastroenterologist",
    "Hepatitis": "Gastroenterologist",
    "UTI": "Urologist",
    # Neurological
    "Migraine": "Neurologist",
    "Anxiety Attack": "Psychiatrist",
}

DEFAULT_SPECIALTY = "General Physician"


def get_specialty_for_disease(disease_name: str) -> str:
    """
    Deterministic lookup -- returns the mapped specialty, or a safe
    default (General Physician) if the disease isn't in the map. Never
    guesses via LLM, and never fails silently with no specialty at all --
    a General Physician referral is always a clinically reasonable
    fallback.
    """
    return DISEASE_TO_SPECIALTY.get(disease_name, DEFAULT_SPECIALTY)


def find_nearby_doctors(disease_name: str, location: str, max_results: int = 5) -> dict:
    """
    Maps disease_name to a specialty, then queries Google Places API
    (Text Search) for nearby doctors/clinics matching that specialty
    near the given location (a free-text city/area string, e.g.
    "Bandra, Mumbai").

    Returns a dict with the specialty used, the search query, and a
    list of results (name, address, rating, total_ratings), sorted by
    Google's own relevance ranking (which factors in rating and
    proximity).

    WHY TEXT SEARCH, NOT NEARBY SEARCH:
    Nearby Search requires precise lat/long coordinates, which would
    need browser geolocation (added complexity, permission prompts).
    Text Search accepts a free-text location string directly, matching
    our simpler "type your city/area" UX decision -- a legitimate,
    documented trade-off for a dissertation-scope deployment.
    """
    specialty = get_specialty_for_disease(disease_name)
    query = f"{specialty} near {location}"

    if not GOOGLE_PLACES_API_KEY:
        return {
            "specialty": specialty,
            "query": query,
            "results": [],
            "error": "Google Places API key not configured.",
        }

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": GOOGLE_PLACES_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except requests.RequestException as e:
        return {
            "specialty": specialty,
            "query": query,
            "results": [],
            "error": f"Request failed: {str(e)}",
        }

    if data.get("status") != "OK":
        return {
            "specialty": specialty,
            "query": query,
            "results": [],
            "error": data.get("status", "Unknown error"),
        }

    results = []
    for place in data.get("results", [])[:max_results]:
        results.append({
            "name": place.get("name", "Unknown"),
            "address": place.get("formatted_address", "Address not available"),
            "rating": place.get("rating", "No rating"),
            "total_ratings": place.get("user_ratings_total", 0),
        })

    return {
        "specialty": specialty,
        "query": query,
        "results": results,
        "error": None,
    }

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", GOOGLE_PLACES_API_KEY)
# WHY A SEPARATE ENV VAR NAME WITH FALLBACK: Distance Matrix and Maps
# Embed are typically enabled on the same Google Cloud project/key as
# Places, so the existing GOOGLE_PLACES_API_KEY usually works directly.
# A distinct GOOGLE_MAPS_API_KEY is supported in case the key is ever
# split or restricted differently in future.


def get_distances(origin: str, destinations: list) -> dict:
    """
    Batch lookup: given one origin (free-text location string, same as
    what the user already typed for find_nearby_doctors()) and a list
    of destination address strings, returns drive time + distance for
    each destination in a SINGLE API call (Distance Matrix supports
    multiple destinations per request, avoiding one call per doctor).

    Returns a dict keyed by destination address, each value a dict with
    "duration_text" (e.g. "12 mins") and "distance_text" (e.g. "4.1 km"),
    or None for a destination if that specific lookup failed (e.g.
    address couldn't be geocoded) -- callers should handle missing
    entries gracefully rather than assuming every destination succeeds.

    WHY TEXT ADDRESSES, NOT LAT/LNG:
    Consistent with find_nearby_doctors()'s existing design decision
    (Text Search over Nearby Search) -- avoids needing browser
    geolocation. Google's Distance Matrix API accepts free-text origins
    and destinations directly and geocodes them internally. This means
    the origin resolves to the approximate center of the named area
    (e.g. "Bandra, Mumbai"), not the user's precise GPS position -- a
    known, documented trade-off, same as the existing location text
    field already implies for Places search.
    """
    if not GOOGLE_MAPS_API_KEY or not destinations:
        return {}

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": "|".join(destinations),
        "key": GOOGLE_MAPS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except requests.RequestException:
        return {}

    if data.get("status") != "OK":
        return {}

    results = {}
    rows = data.get("rows", [])
    if not rows:
        return {}

    elements = rows[0].get("elements", [])
    for dest, element in zip(destinations, elements):
        if element.get("status") == "OK":
            results[dest] = {
                "duration_text": element["duration"]["text"],
                "distance_text": element["distance"]["text"],
            }
        else:
            results[dest] = None

    return results


def get_embed_map_url(origin: str, destination: str) -> str:
    """
    Builds a Google Maps Embed API URL showing driving directions from
    origin to destination, for use in an iframe. Maps Embed API has no
    per-load charge, unlike most other Google Maps Platform APIs.

    URL-encodes both origin and destination (via urllib.parse.quote)
    since real addresses commonly contain spaces, commas, and other
    characters that are invalid in a raw URL and were causing Google
    to reject the request as malformed.
    """
    if not GOOGLE_MAPS_API_KEY:
        return ""
    from urllib.parse import quote
    base = "https://www.google.com/maps/embed/v1/directions"
    params = f"?key={GOOGLE_MAPS_API_KEY}&origin={quote(origin)}&destination={quote(destination)}"
    return base + params


if __name__ == "__main__":
    test_cases = [
        ("Migraine", "Bandra, Mumbai"),
        ("Arrhythmia", "Pune"),
        ("Type 2 Diabetes", "Nashik"),
    ]

    for disease, location in test_cases:
        print(f"\n{'=' * 60}")
        print(f"DISEASE: {disease} | LOCATION: {location}")
        result = find_nearby_doctors(disease, location)
        print(f"SPECIALTY: {result['specialty']}")
        print(f"QUERY: {result['query']}")
        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            for doc in result["results"]:
                print(f"  - {doc['name']} | {doc['rating']}★ ({doc['total_ratings']} reviews) | {doc['address']}")