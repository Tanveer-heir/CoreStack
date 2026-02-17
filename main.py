"""
Hybrid Architecture: CodeAct + CoreStack Tool
==============================================
"""

import os
import io
import re
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
from contextlib import redirect_stdout

from dotenv import load_dotenv

# Import CodeAct from smolagents (structure from agent.py)
from smolagents import CodeAgent, tool, LiteLLMModel, DuckDuckGoSearchTool

# No need to import executor - it's specified via executor_type parameter
DOCKER_AVAILABLE = True  # Assume Docker is available like in agent.py

# Import Earth Engine
import ee

# Langfuse
from langfuse import Langfuse

load_dotenv()

# Get API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CORE_STACK_API_KEY = os.getenv("CORE_STACK_API_KEY")
CORESTACK_BASE_URL = os.getenv("CORESTACK_BASE_URL", "https://geoserver.core-stack.org/api/v1")

# Langfuse keys
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Initialize Earth Engine
GEE_PROJECT = os.getenv("GEE_PROJECT", "apt-achievment-453417-h6")
try:
	ee.Initialize(project=GEE_PROJECT)
	print(f"✅ Earth Engine initialized with project: {GEE_PROJECT}")
except Exception as e:
	print(f"⚠️  Earth Engine initialization failed: {e}")

# ======================================================
# DATA PRODUCT NAME CACHE (from layer_descriptions.csv)
# ======================================================

CORESTACK_DATA_PRODUCTS = {
	"raster_layers": [
		"land_use_land_cover_raster",
		"terrain_raster",
		"change_tree_cover_gain_raster",
		"change_tree_cover_loss_raster",
		"change_cropping_reduction_raster",
		"change_cropping_intensity_raster",
		"tree_canopy_cover_density_raster",
		"tree_canopy_height_raster",
		"stream_order_raster",
		"distance_to_upstream_drainage_line",
		"catchment_area",
		"runoff_accumulation",
		"natural_depressions",
		"clart_raster"
	],
	"vector_layers": [
		"drainage_lines_vector",
		"aquifer_vector",
		"stage_of_groundwater_extraction_vector",
		"nrega_vector",
		"admin_boundaries_vector",
		"drought_frequency_vector",
		"surface_water_bodies_vector",
		"water_balance",
		"change_in_well_depth_vector",
		"cropping_intensity_vector"
	],
	"timeseries_metrics": [
		"cropping_intensity", "precipitation", "temperature", "soil_moisture",
		"ndvi", "evapotranspiration", "groundwater_level", "rainfall", "water_balance"
	]
}

# ============================================================================
# LANGFUSE TRACING
# ============================================================================

_LANGFUSE_TRACE: ContextVar[Optional[Any]] = ContextVar("langfuse_trace", default=None)
_LANGFUSE_CLIENT: Optional[Langfuse] = None


def _get_langfuse_client() -> Optional[Langfuse]:
	global _LANGFUSE_CLIENT
	if _LANGFUSE_CLIENT is not None:
		return _LANGFUSE_CLIENT
	if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST):
		return None
	try:
		_LANGFUSE_CLIENT = Langfuse(
			public_key=LANGFUSE_PUBLIC_KEY,
			secret_key=LANGFUSE_SECRET_KEY,
			host=LANGFUSE_HOST
		)
		return _LANGFUSE_CLIENT
	except Exception:
		return None


def _start_trace(user_query: str, model_id: str) -> Optional[Any]:
	client = _get_langfuse_client()
	if client is None:
		return None
	try:
		return client.trace(
			name="run_hybrid_agent",
			input=user_query,
			metadata={
				"user_query": user_query,
				"timestamp": datetime.utcnow().isoformat(),
				"model_id": model_id
			}
		)
	except Exception:
		return None


def _start_span(trace: Optional[Any], name: str, input_data: Optional[Any] = None,
				metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
	if trace is None:
		return None
	try:
		return trace.span(name=name, input=input_data, metadata=metadata or {})
	except Exception:
		return None


def _end_span(span: Optional[Any], output_data: Optional[Any] = None,
			  metadata: Optional[Dict[str, Any]] = None) -> None:
	if span is None:
		return
	try:
		span.end(output=output_data, metadata=metadata or {})
	except Exception:
		pass


def _update_trace(trace: Optional[Any], output_data: Optional[Any] = None,
				  metadata: Optional[Dict[str, Any]] = None) -> None:
	if trace is None:
		return
	try:
		trace.update(output=output_data, metadata=metadata or {})
	except Exception:
		pass


class _TeeStdout:
	def __init__(self, *streams):
		self._streams = streams

	def write(self, data: str) -> int:
		for stream in self._streams:
			stream.write(data)
		return len(data)

	def flush(self) -> None:
		for stream in self._streams:
			stream.flush()


def _extract_code_blocks(text: str) -> str:
	if not text:
		return ""
	blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
	return "\n\n".join(blocks).strip()


# ============================================================================
# CORESTACK WORKFLOW (PURE PYTHON, NO LANGGRAPH)
# ============================================================================


def _corestack_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
	import requests

	if not CORE_STACK_API_KEY:
		raise RuntimeError("CORE_STACK_API_KEY is not set")

	url = f"{CORESTACK_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
	headers = {"X-API-Key": CORE_STACK_API_KEY}
	response = requests.get(url, headers=headers, params=params, timeout=30)
	response.raise_for_status()
	try:
		payload = response.json()
	except ValueError as exc:
		raise RuntimeError(f"CoreStack API returned non-JSON response from {endpoint}") from exc

	if isinstance(payload, dict) and payload.get("success") is False:
		message = payload.get("message") or payload.get("error") or "Unknown error"
		raise RuntimeError(f"CoreStack API error from {endpoint}: {message}")

	return payload


def _fetch_active_locations() -> Dict[str, Any]:
	payload = _corestack_request("get_active_locations/")
	print("\n[CoreStack Debug] Active locations payload type:", type(payload).__name__)
	print("[CoreStack Debug] Active locations payload (truncated):", str(payload)[:1000])
	return _normalize_active_locations(payload)


def _normalize_active_locations(payload: Any) -> Dict[str, Any]:
	if isinstance(payload, dict):
		return payload
	if isinstance(payload, list):
		normalized: Dict[str, Any] = {}
		for item in payload:
			if not isinstance(item, dict):
				continue
			state = item.get("state") or item.get("State") or item.get("label")
			districts = item.get("district") or item.get("District")
			if state and isinstance(districts, list):
				for district_item in districts:
					if not isinstance(district_item, dict):
						continue
					district = district_item.get("district") or district_item.get("District") or district_item.get("label")
					blocks = district_item.get("blocks") or district_item.get("tehsils") or district_item.get("tehsil")
					if not (state and district and isinstance(blocks, list)):
						continue
					state_entry = normalized.setdefault(state, {"districts": {}})
					district_entry = state_entry["districts"].setdefault(district, {"tehsils": []})
					for block in blocks:
						if not isinstance(block, dict):
							continue
						tehsil = block.get("tehsil") or block.get("Tehsil") or block.get("label")
						if tehsil and tehsil not in district_entry["tehsils"]:
							district_entry["tehsils"].append(tehsil)
				continue
			district = item.get("district") or item.get("District")
			tehsil = item.get("tehsil") or item.get("Tehsil")
			if not (state and district and tehsil):
				continue
			state_entry = normalized.setdefault(state, {"districts": {}})
			district_entry = state_entry["districts"].setdefault(district, {"tehsils": []})
			if tehsil not in district_entry["tehsils"]:
				district_entry["tehsils"].append(tehsil)
		if normalized:
			return normalized
		raise RuntimeError("CoreStack API returned empty active locations list")
	raise RuntimeError("CoreStack API returned unexpected active locations payload")


def _geocode_and_resolve(query: str, active_locations: Dict[str, Any]) -> Dict[str, Optional[str]]:
	"""Geocode place names in the query to lat/lon, then resolve via CoreStack admin API."""
	try:
		from geopy.geocoders import Nominatim
		geolocator = Nominatim(user_agent="corestack_agent")

		# Extract likely place names: strip common non-location words
		_strip_words = {
			"cropping", "intensity", "vector", "raster", "data", "layer", "change",
			"tree", "cover", "loss", "gain", "surface", "water", "availability",
			"over", "years", "show", "me", "how", "has", "changed", "the", "in",
			"of", "for", "from", "to", "and", "with", "average", "total", "all",
			"available", "spatial", "analysis", "could", "you", "village", "tehsil",
			"block", "district", "state", "india", "please", "what", "is", "are",
		}
		words = re.findall(r'[A-Za-z]+', query)
		place_words = [w for w in words if w.lower() not in _strip_words and len(w) > 2]
		search_query = " ".join(place_words) + ", India"
		print(f"[CoreStack Debug] Geocoding query: {search_query}")

		location = geolocator.geocode(search_query, timeout=10)
		if location is None:
			print("[CoreStack Debug] Geocoding returned no results")
			return {"state": None, "district": None, "tehsil": None}

		lat, lon = location.latitude, location.longitude
		print(f"[CoreStack Debug] Geocoded to: {lat}, {lon}")

		admin = _get_admin_details_by_latlon(lat, lon)
		print(f"[CoreStack Debug] Admin from lat/lon: {admin}")
		resolved = _normalize_location_for_active(admin, active_locations)
		if all([resolved.get("state"), resolved.get("district"), resolved.get("tehsil")]):
			print(f"[CoreStack Debug] Geocode resolved: {resolved}")
			return resolved

		return {"state": None, "district": None, "tehsil": None}
	except Exception as e:
		print(f"[CoreStack Debug] Geocoding failed: {e}")
		return {"state": None, "district": None, "tehsil": None}


def _get_admin_details_by_latlon(latitude: float, longitude: float) -> Dict[str, Optional[str]]:
	params = {"latitude": latitude, "longitude": longitude}
	data = _corestack_request("get_admin_details_by_latlon/", params=params)
	if not isinstance(data, dict):
		return {"state": None, "district": None, "tehsil": None}
	return {
		"state": data.get("State") or data.get("state"),
		"district": data.get("District") or data.get("district"),
		"tehsil": data.get("Tehsil") or data.get("tehsil")
	}


def _normalize_location_for_active(location: Dict[str, Optional[str]],
							   active_locations: Dict[str, Any]) -> Dict[str, Optional[str]]:
	state = location.get("state")
	district = location.get("district")
	tehsil = location.get("tehsil")
	if not (state and district and tehsil):
		return {"state": state, "district": district, "tehsil": tehsil}

	state_key = next((name for name in active_locations.keys() if name.lower() == state.lower()), None)
	if not state_key:
		return {"state": state, "district": district, "tehsil": tehsil}

	districts = active_locations.get(state_key, {}).get("districts", {})
	district_key = next((name for name in districts.keys() if name.lower() == district.lower()), None)
	if not district_key:
		return {"state": state_key, "district": district, "tehsil": tehsil}

	tehsils = districts.get(district_key, {}).get("tehsils", [])
	tehsil_key = next((name for name in tehsils if name.lower() == tehsil.lower()), None)
	if not tehsil_key:
		return {"state": state_key, "district": district_key, "tehsil": tehsil}

	return {"state": state_key, "district": district_key, "tehsil": tehsil_key}


def _lc_words(text: str) -> set:
	return set(re.findall(r'[a-z]+', text.lower()))


def _match_location_from_query(query: str, active_locations: Dict[str, Any]) -> Dict[str, str]:
	query_lc = query.lower()
	query_words = _lc_words(query)
	print(f"[CoreStack Debug] Location matching — query words: {query_words}")

	best_state = None
	best_district = None
	best_tehsil = None

	# Pass 1: match state, then district, then tehsil by word overlap
	for state_name, state_data in active_locations.items():
		state_words = _lc_words(state_name)
		if state_words & query_words or state_name.lower() in query_lc:
			best_state = state_name
			districts = state_data.get("districts", {})
			for district_name, district_data in districts.items():
				district_words = _lc_words(district_name)
				if district_words & query_words or district_name.lower() in query_lc:
					best_district = district_name
					tehsils = district_data.get("tehsils", [])
					for tehsil_name in tehsils:
						tehsil_words = _lc_words(tehsil_name)
						if tehsil_words & query_words or tehsil_name.lower() in query_lc:
							best_tehsil = tehsil_name
							print(f"[CoreStack Debug] Matched: {best_state} > {best_district} > {best_tehsil}")
							return {
								"state": best_state,
								"district": best_district,
								"tehsil": best_tehsil
							}

	# Pass 2: state + district matched, pick first tehsil
	if best_state and best_district:
		tehsils = active_locations.get(best_state, {}).get("districts", {}).get(best_district, {}).get("tehsils", [])
		best_tehsil = tehsils[0] if tehsils else None
		print(f"[CoreStack Debug] Partial match (first tehsil): {best_state} > {best_district} > {best_tehsil}")
		return {
			"state": best_state,
			"district": best_district,
			"tehsil": best_tehsil
		}

	# Pass 3: match district or tehsil anywhere (no state in query)
	for state_name, state_data in active_locations.items():
		districts = state_data.get("districts", {})
		for district_name, district_data in districts.items():
			district_words = _lc_words(district_name)
			if district_words & query_words or district_name.lower() in query_lc:
				best_state = state_name
				best_district = district_name
				tehsils = district_data.get("tehsils", [])
				for tehsil_name in tehsils:
					tehsil_words = _lc_words(tehsil_name)
					if tehsil_words & query_words or tehsil_name.lower() in query_lc:
						best_tehsil = tehsil_name
						break
				if not best_tehsil:
					best_tehsil = tehsils[0] if tehsils else None
				print(f"[CoreStack Debug] Fallback match: {best_state} > {best_district} > {best_tehsil}")
				return {
					"state": best_state,
					"district": best_district,
					"tehsil": best_tehsil
				}

	# Pass 4: match just a tehsil name anywhere in any state/district
	for state_name, state_data in active_locations.items():
		districts = state_data.get("districts", {})
		for district_name, district_data in districts.items():
			tehsils = district_data.get("tehsils", [])
			for tehsil_name in tehsils:
				tehsil_words = _lc_words(tehsil_name)
				if tehsil_words & query_words or tehsil_name.lower() in query_lc:
					print(f"[CoreStack Debug] Tehsil-only match: {state_name} > {district_name} > {tehsil_name}")
					return {
						"state": state_name,
						"district": district_name,
						"tehsil": tehsil_name
					}

	print(f"[CoreStack Debug] No text match found, falling back to geocoding")
	return {"state": None, "district": None, "tehsil": None}


def _get_generated_layer_urls(state: str, district: str, tehsil: str) -> Any:
	params = {
		"state": state,
		"district": district,
		"tehsil": tehsil
	}
	payload = _corestack_request("get_generated_layer_urls/", params=params)
	print("\n[CoreStack Debug] Generated layer URLs payload type:", type(payload).__name__)
	print("[CoreStack Debug] Generated layer URLs payload (truncated):", str(payload)[:1000])
	return payload


def _normalize_layer_payload(layer_payload: Any, location: Dict[str, str]) -> Dict[str, Any]:
	if not isinstance(layer_payload, list):
		raise RuntimeError("CoreStack API returned unexpected layer payload")
	vector_layers = []
	raster_layers = []
	for layer in layer_payload:
		layer_type = str(layer.get("layer_type", "")).lower()
		layer_name = layer.get("layer_name")
		dataset_name = layer.get("dataset_name") or layer.get("dataset")
		layer_url = layer.get("layer_url")
		if not layer_name or not layer_url:
			continue
		if dataset_name and dataset_name not in layer_name:
			layer_name = f"{dataset_name} ({layer_name})"
		entry = {
			"layer_name": layer_name,
			"layer_type": "vector" if "vector" in layer_type else "raster",
			"urls": [
				{
					"url": layer_url,
					"tehsil": location.get("tehsil"),
					"district": location.get("district"),
					"state": location.get("state")
				}
			]
		}
		if entry["layer_type"] == "vector":
			vector_layers.append(entry)
		else:
			raster_layers.append(entry)
	return {"vector": vector_layers, "raster": raster_layers}


def _build_local_layer_catalog() -> Dict[str, Any]:
	vector_layers = []
	raster_layers = []
	for name in CORESTACK_DATA_PRODUCTS.get("vector_layers", []):
		vector_layers.append({
			"layer_name": name,
			"layer_type": "vector",
			"urls": []
		})
	for name in CORESTACK_DATA_PRODUCTS.get("raster_layers", []):
		raster_layers.append({
			"layer_name": name,
			"layer_type": "raster",
			"urls": []
		})
	return {"vector": vector_layers, "raster": raster_layers}


def _parse_basic_query(query: str) -> Dict[str, Any]:
	lat_match = re.search(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)", query)
	lat = lon = None
	if lat_match:
		lat = float(lat_match.group(1))
		lon = float(lat_match.group(2))
	years = re.findall(r"(\d{4})", query)
	start_year = int(years[0]) if years else None
	end_year = int(years[-1]) if len(years) > 1 else None
	return {
		"location_name": None,
		"latitude": lat,
		"longitude": lon,
		"temporal": bool(years),
		"start_year": start_year,
		"end_year": end_year
	}


def _run_corestack_workflow(user_query: str) -> Dict[str, Any]:
	parsed = _parse_basic_query(user_query)
	if not CORE_STACK_API_KEY:
		return {
			"available_layers": _build_local_layer_catalog(),
			"location_info": {},
			"parsed": parsed,
			"error": "CORE_STACK_API_KEY is not set"
		}

	try:
		active_locations = _fetch_active_locations()
		location = {"state": None, "district": None, "tehsil": None}

		# Strategy 1: lat/lon from query -> admin API
		if parsed.get("latitude") is not None and parsed.get("longitude") is not None:
			admin_location = _get_admin_details_by_latlon(parsed["latitude"], parsed["longitude"])
			location = _normalize_location_for_active(admin_location, active_locations)
			if not all([location.get("state"), location.get("district"), location.get("tehsil")]):
				location = {"state": None, "district": None, "tehsil": None}

		# Strategy 2: text matching against active locations
		if not all([location.get("state"), location.get("district"), location.get("tehsil")]):
			location = _match_location_from_query(user_query, active_locations)

		# Strategy 3: geocode place name from query -> lat/lon -> admin API
		if not all([location.get("state"), location.get("district"), location.get("tehsil")]):
			location = _geocode_and_resolve(user_query, active_locations)

		if not all([location.get("state"), location.get("district"), location.get("tehsil")]):
			return {
				"available_layers": _build_local_layer_catalog(),
				"location_info": location,
				"parsed": parsed,
				"error": "Unable to resolve state/district/tehsil from query"
			}
		layer_payload = _get_generated_layer_urls(location["state"], location["district"], location["tehsil"])
		available_layers = _normalize_layer_payload(layer_payload, location)
		return {
			"available_layers": available_layers,
			"location_info": location,
			"parsed": parsed
		}
	except Exception as e:
		return {
			"available_layers": _build_local_layer_catalog(),
			"location_info": {},
			"parsed": parsed,
			"error": str(e)
		}


# ============================================================================
# CORESTACK AS A TOOL FOR CODEACT
# ============================================================================

@tool
def fetch_corestack_data(query: str) -> str:
	"""
	Fetches available CoreStack layers for a location from CoreStack workflow.
	Returns list of available layers with URLs for CodeAct to choose and analyze.

	Args:
		query: Natural language query with location (e.g., "Shirur village cropping intensity")

	Returns:
		JSON string with:
		- success: bool
		- spatial_data: dict with vector_layers[] and raster_layers[]
		- Each layer has: layer_name, layer_type, urls[] (multi-region support)
		- location_info: administrative details
	"""
	import json
	import sys
	import os

	# Add parent directory to path
	workspace_path = '/app/workspace'
	if workspace_path not in sys.path:
		sys.path.insert(0, workspace_path)

	print("\n" + "="*70)
	print("📊 CORESTACK LAYER FETCHER (via CoreStack workflow)")
	print(f"   Query: {query}")
	print("="*70)

	trace = _LANGFUSE_TRACE.get()
	span = _start_span(trace, "tool_execution", input_data={"query": query}, metadata={
		"tool_name": "fetch_corestack_data"
	})

	try:
		# Run workflow
		result_state = _run_corestack_workflow(query)

		# Check for errors
		if "error" in result_state:
			response = json.dumps({
				"success": False,
				"error": result_state["error"]
			}, indent=2)
			_end_span(span, output_data=response, metadata={"success": False})
			return response

		# Extract available layers
		available_layers = result_state.get("available_layers", {})
		location_info = result_state.get("location_info", {})
		parsed = result_state.get("parsed", {})

		# Return layer catalog for CodeAct to choose from
		response_obj = {
			"success": True,
			"data_type": "spatial",
			"spatial_data": {
				"vector_layers": available_layers.get('vector', []),
				"raster_layers": available_layers.get('raster', [])
			},
			"location_info": location_info,
			"parsed_query": {
				"location_name": parsed.get('location_name'),
				"latitude": parsed.get('latitude'),
				"longitude": parsed.get('longitude'),
				"temporal": parsed.get('temporal'),
				"start_year": parsed.get('start_year'),
				"end_year": parsed.get('end_year')
			}
		}

		print(f"\n✅ FETCH COMPLETE:")
		print(f"   Vector layers: {len(available_layers.get('vector', []))}")
		print(f"   Raster layers: {len(available_layers.get('raster', []))}")

		response = json.dumps(response_obj, default=str)
		_end_span(span, output_data=response, metadata={"success": True})
		return response

	except Exception as e:
		print(f"\n❌ ERROR: {str(e)}")
		traceback.print_exc()
		response = json.dumps({
			"success": False,
			"error": str(e)
		}, indent=2)
		_end_span(span, output_data=response, metadata={"success": False, "error": str(e)})
		return response


# ============================================================================
# CODEACT PROMPT (ADAPTED FROM agent.py)
# ============================================================================

def create_corestack_prompt(task: str) -> str:
	"""
	Creates the CodeAct prompt, with CoreStack as primary data source.

	Key features:
	- Emphasizes CoreStack as primary data source for India-specific queries
	- Includes Earth Engine as supplementary tool for global/non-India data
	- Lists all correct libraries and proper export formats
	- Specifies expected output types
	"""
	return f"""
You are a geospatial analysis agent, expert at writing python code to perform geospatial analysis. You will use the following python libraries:
osmnx, geopandas, shapely, matplotlib, numpy, pandas, rasterio, ee, geemap, geedim, geopy, requests, json

You will be given a task, you will write python code to perform the task and export outputs to local machine.

If your code has errors, you will search for tutorials and documentations of earlier mentioned libraries to fix the errors.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL UNDERSTANDING: SPATIAL vs TIMESERIES DATA (CoreStack Architecture)
═══════════════════════════════════════════════════════════════════════════════

CoreStack has TWO FUNDAMENTALLY DIFFERENT data structures you MUST understand:

**1️⃣ SPATIAL DATA (Vectors/Rasters)** - For location-specific analysis:
   - Geographic features with SPATIAL VARIATION across a region
   - Data varies BY LOCATION (different patches/polygons have different values)
   - Temporal data stored as YEARLY ATTRIBUTE COLUMNS, NOT timeseries arrays
   - Example: cropping_intensity_vector has columns: cropping_intensity_2017, cropping_intensity_2018, ..., cropping_intensity_2023
   - Use for: Village/tehsil-level analysis where spatial variation matters

**2️⃣ TIMESERIES DATA (Watershed MWS)** - For temporal water budget analysis:
   - Single aggregated value per watershed per time period
   - NO spatial variation (entire watershed = 1 value)
   - Fortnightly measurements (water balance components)
   - Example: water_balance timeseries has arrays: year[], fortnight[], runoff[], precipitation[]
   - Use for: Watershed water budget analysis ONLY

═══════════════════════════════════════════════════════════════════════════════
LAYER SELECTION DECISION FRAMEWORK (WITH EXPLANATIONS)
═══════════════════════════════════════════════════════════════════════════════

**Query Type 1: "Cropping Intensity in [Village] Over Years"**
✅ CORRECT CHOICE: cropping_intensity_vector (spatial vector)
❌ WRONG CHOICE: watershed timeseries

WHY SPATIAL?
- Cropping intensity is SPATIALLY VARIABLE (different fields = different intensity)
- You want: total area under crops in village, which varies by location
- Data structure: GeoDataFrame with polygon features
- Temporal aspect: Stored as yearly columns (cropping_intensity_2017, 2018, ..., 2023)
- Analysis approach: Sum/average these columns across village polygons, plot trend

WHY NOT TIMESERIES?
- Watershed timeseries measures aggregate WATER BALANCE (runoff, precip, ET)
- It's NOT about cropping patterns or land use
- Timeseries is fortnightly, not yearly
- Timeseries has NO spatial variation (1 value per watershed)

**Query Type 2: "Surface Water Availability Over Years in [Village]"**
✅ PRIMARY: surface_water_bodies_vector (spatial vector)
⚠️  OPTIONAL CONTEXT: water_balance (timeseries) for watershed-level trends

WHY SPATIAL?
- Surface water bodies are PHYSICAL FEATURES with geometry (lakes, ponds, reservoirs)
- You want: total area of water bodies within village boundaries
- Data structure: Polygon features with seasonal attributes (Kharif/Rabi/Zaid flags)
- Analysis: Clip polygons to village, sum area per season

CAVEAT: surface_water_bodies_vector may be a SINGLE SNAPSHOT per year
- For multi-year TRENDS, derive from land_use_land_cover_raster
- Workaround: Count water pixels (classes 2-4) in LULC for years 2017-2024

**Query Type 3: "Tree Cover Loss in [Village] Since [Year]"**
✅ CORRECT: change_tree_cover_loss_raster (change raster)

WHY CHANGE RASTER?
- Pre-computed transition matrix: trees (class 6) → other classes
- Period: 2017-2022 composite (mode of 2017-2019 vs 2020-2022)
- Classes: Trees→Built-up, Trees→Barren, Trees→Crops, Trees→Shrubs
- Analysis: Mask to loss classes, count pixels, convert to hectares
- SAVES COMPUTATION vs manually comparing 2017 vs 2022 LULC

FALLBACK: If custom time period (e.g., "2018-2024"):
- Use land_use_land_cover_raster for specific years
- Compare class 6 (trees) presence across years manually

**Query Type 4: "Change in Urbanization / Land Cover Change in [Village]"**
✅ CORRECT APPROACH: Compare LULC rasters across two years (LULC comparison)
⛔ NEVER use any layer with "Urbanization" or "change_urbanization" in the name. Those layers return 404.
⛔ NEVER search for "change_urbanization_raster". It does NOT work.

INSTEAD, you MUST:
1. From the fetched CoreStack data, find LULC raster layers (layer_name contains "LULC" or dataset_name contains "LULC_level_3")
   - Example layer names: LULC_17_18_dharwad_navalgund_level_3, LULC_24_25_dharwad_navalgund_level_3
2. Pick the EARLIEST available LULC (e.g., LULC_17_18) and the MOST RECENT (e.g., LULC_24_25)
3. Download BOTH rasters using requests.get() and save to ./exports/
4. Open both with rasterio, read band 1
5. LULC class codes:
   - 0: Background, 1: Built-up, 2: Water (Kharif), 3: Water (Kharif+Rabi)
   - 4: Water (Kharif+Rabi+Zaid), 6: Tree/Forests, 7: Barrenlands
   - 8: Single cropping cropland, 9: Single Non-Kharif cropping cropland
   - 10: Double cropping cropland, 11: Triple cropping cropland, 12: Shrub_Scrub
6. Reproject to EPSG:32643 for area calculation
7. Compute per-class pixel count and area (hectares) for BOTH years
8. Create a CHANGE MAP: where old != new, highlight changed pixels
9. For urbanization: count pixels that became class 1 (Built-up) in recent LULC but were NOT class 1 in old LULC
10. Export: change map GeoTIFF, area comparison CSV, visualization PNG

WHY LULC COMPARISON?
- Pre-computed change rasters return 404 and are unreliable
- LULC comparison gives you FULL CONTROL over time period and class transitions
- You get BOTH area statistics AND a spatial change map

**Query Type 5: "Microwatersheds with highest cropping sensitivity to drought"**
✅ CORRECT LAYERS: drought layer + cropping intensity layer (both spatial vectors from SAME fetch)
⛔ NEVER use np.corrcoef on w_sev columns (they are mostly zero → NaN correlation)
⛔ NEVER invent column names like 'mws_id_source', 'mws_id', 'watershed_id' — they DO NOT EXIST

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 4 below almost verbatim.
   The ONLY column to group by after sjoin is `id_drought` (the drought GDF's `id` column with lsuffix).
   The sjoin MUST use: `lsuffix='drought', rsuffix='crop'`
   After sjoin, print columns: `print(joined.columns.tolist())` to verify.

⚠️ CRITICAL LAYER NAME MATCHING:
- The drought layer name is `"Drought (dharwad_navalgund_drought)"` — match with `'drought' in name.lower()`
- The cropping layer name is `"Cropping Intensity (dharwad_navalgund_intensity)"` — match with `'cropping' in name.lower() and 'intensity' in name.lower()`
- ALWAYS print ALL vector layer names FIRST: `for l in vector_layers: print(l['layer_name'])`
- DO NOT look for exact strings like 'drought_frequency' — the API returns different names

DROUGHT DATA COLUMN REFERENCE:
- `w_sev_YYYY`: Weeks of SEVERE drought in year YYYY (integer, often 0)
- `w_mod_YYYY`: Weeks of MODERATE drought
- `w_mld_YYYY`: Weeks of MILD drought
- `w_no_YYYY`: Weeks of NO drought
- `drysp_YYYY`: Dry spell length in weeks
- `avg_dryspell`: Average dry spell across years
- `t_wks_YYYY`: Total weeks in season for year YYYY

CROPPING DATA COLUMN REFERENCE:
- `cropping_intensity_YYYY`: Cropping intensity value for year YYYY (2017-2024)

CORRECT METHODOLOGY (drought-cropping sensitivity analysis):
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names: `for l in vector_layers: print(l['layer_name'])`
3. Find drought layer: `'drought' in layer['layer_name'].lower()` (NOT 'drought_frequency')
4. Find cropping layer: `'cropping' in layer['layer_name'].lower() and 'intensity' in layer['layer_name'].lower()`
5. Load both as GeoDataFrames
6. Spatial join drought_gdf with cropping_gdf (inner join, intersects)
7. For EACH microwatershed (group by drought MWS id):
   a) Classify each year (2017-2022) as DROUGHT or NON-DROUGHT:
      - A year is a DROUGHT YEAR if `drysp_YYYY > 0` OR `w_sev_YYYY + w_mod_YYYY + w_mld_YYYY > 0`
      - Otherwise it is a NON-DROUGHT YEAR
   b) Compute mean cropping intensity across DROUGHT years
   c) Compute mean cropping intensity across NON-DROUGHT years
   d) sensitivity_score = mean_non_drought - mean_drought (positive = drop during drought)
8. Rank microwatersheds by sensitivity_score DESCENDING (highest drop = most sensitive)
9. Take top N, export as GeoJSON with sensitivity scores
10. ALSO resolve to village names using admin_boundaries_vector spatial join with vill_name column

WHY THIS APPROACH?
- w_sev columns are mostly 0 (severe drought is rare) → correlation fails with NaN
- drysp (dry spell) columns are more reliable indicators of drought occurrence
- Comparing mean cropping intensity between drought vs non-drought years gives a clear sensitivity signal
- Positive difference means cropping drops during drought = high sensitivity

═══════════════════════════════════════════════════════════════════════════════

**Query Type 6: "Microwatersheds with highest surface water availability sensitivity to drought"**
✅ CORRECT LAYERS: drought layer + surface water bodies layer (both from SAME fetch)
⛔ NEVER use np.corrcoef on w_sev columns (they are mostly zero → NaN)
⛔ NEVER invent column names — print columns FIRST to verify

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 5 below almost verbatim.
   Use TABULAR merge (not spatial join): merge drought_gdf with aggregated surface water on `uid` == `MWS_UID`.
   After merge, iterate rows (one per MWS) and compare surface water area in drought vs non-drought years.

⚠️ CRITICAL LAYER NAME MATCHING:
- Drought layer: `"Drought (dharwad_navalgund_drought)"` — match with `'drought' in name.lower()` AND `'causality' not in name.lower()`
- Surface water layer: `"Surface Water Bodies (surface_waterbodies_dharwad_navalgund)"` — match with `'surface water' in name.lower()` AND `'zoi' not in name.lower()`
- There are TWO surface water layers — use the one WITHOUT 'zoi' in its name
- ALWAYS print ALL vector layer names FIRST

SURFACE WATER DATA COLUMN REFERENCE:
- `area_YY-YY`: Water spread area in hectares for hydro-year (e.g., `area_17-18` = 2017-18 hydro-year)
- `k_YY-YY`: Kharif season water availability (%)
- `kr_YY-YY`: Kharif+Rabi season water availability (%)
- `krz_YY-YY`: Kharif+Rabi+Zaid season water availability (%)
- `MWS_UID`: Microwatershed UID (matches `uid` in drought layer)
- `Village Name`: Village name (already present — no need for admin boundary join!)
- Each row is ONE water body (2798 total) — multiple water bodies per MWS

YEAR MAPPING (drought calendar year → surface water hydro-year):
- drysp_2017 → area_17-18
- drysp_2018 → area_18-19
- drysp_2019 → area_19-20
- drysp_2020 → area_20-21
- drysp_2021 → area_21-22
- drysp_2022 → area_22-23

CORRECT METHODOLOGY (surface water sensitivity analysis):
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names
3. Find drought layer: `'drought' in name.lower()` AND `'causality' not in name.lower()`
4. Find surface water layer: `'surface water' in name.lower()` AND `'zoi' not in name.lower()`
5. Load both as GeoDataFrames
6. Print columns of both GDFs
7. Aggregate surface water area by MWS: `sw_by_mws = sw_gdf.groupby('MWS_UID')[area_cols].sum().reset_index()`
8. Merge drought_gdf with sw_by_mws on `uid` == `MWS_UID` (tabular merge, NOT spatial join)
9. For EACH MWS (iterate merged rows):
   a) Map drought years to hydro-years: drysp_2017 → area_17-18, etc.
   b) If drysp_YYYY > 0 → DROUGHT YEAR: collect area_YY-(YY+1)
   c) If drysp_YYYY == 0 → NON-DROUGHT YEAR: collect area_YY-(YY+1)
   d) sensitivity_score = mean_non_drought_area - mean_drought_area (positive = water drops during drought)
10. Rank by sensitivity_score DESCENDING (highest drop = most sensitive)
11. Take top N, export as GeoJSON with sensitivity scores

WHY TABULAR MERGE (not spatial join)?
- Both drought and surface water layers share MWS identifiers (uid / MWS_UID)
- Surface water has ~2798 water bodies → must aggregate by MWS_UID first
- Spatial join would create many-to-many duplicates; tabular merge is cleaner and faster
- After groupby + sum, we get one row per MWS with total water area per year

═══════════════════════════════════════════════════════════════════════════════

**Query Type 8: "Find microwatersheds most similar to a given MWS (by uid)"**
✅ CORRECT LAYERS: drought + cropping intensity + terrain vector + LULC vector (ALL from SAME fetch)
⛔ NEVER return layer-specific IDs like `dharwad_navalgund_drought.1` — ALWAYS return the `uid` column (e.g., `18_16157`)
⛔ NEVER use simple boolean equality — use proper distance metrics on numeric features

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 6 below almost verbatim.
   The `uid` column is the ACTUAL MWS identifier shared across ALL layers.
   Load 4 layers, merge on `uid`, build feature matrix, normalize, compute cosine similarity.

⚠️ CRITICAL LAYER NAME MATCHING:
- Drought layer: `'drought' in name.lower()` AND `'causality' not in name.lower()`
- Cropping Intensity: `'cropping' in name.lower() and 'intensity' in name.lower()`
- Terrain Vector: `'terrain' in name.lower() and 'vector' not in name.lower()` (it's `"Terrain Vector (dharwad_navalgund_cluster)"`)
- LULC Vector: `'lulc' in name.lower()` AND `'terrain' not in name.lower()` AND `'level' not in name.lower()` (it's `"LULC (lulc_vector_dharwad_navalgund)"`)
- ALWAYS print ALL vector layer names FIRST

LAYER SCHEMAS (what columns to use from each):
- **Drought** (115 MWS): `uid`, `drysp_2017..2022` (dry spell weeks), `w_sev_2017..2022`, `w_mod_2017..2022`, `w_mld_2017..2022`, `avg_dryspell`
- **Cropping Intensity** (115 MWS): `uid`, `cropping_intensity_2017..2024`
- **Terrain** (115 MWS): `uid`, `hill_slope`, `plain_area`, `ridge_area`, `slopy_area`, `valley_are`, `terrainClu`
- **LULC** (115 MWS, 101 cols): `uid`, plus per-year columns for each land use class (barrenland, built-up, cropland, tree_forest, shrub_scrub, etc.)
   Column names are truncated (e.g., `barrenla_1`, `built-up_2`, `cropland_3`)

CORRECT METHODOLOGY (MWS similarity analysis):
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names
3. Find and load: drought, cropping intensity, terrain, LULC — all as GeoDataFrames
4. Print columns of each to verify
5. Merge all 4 on `uid` column (inner join, drop geometry except from first)
6. Select numeric feature columns (exclude `id`, `uid`, `geometry`, `area_in_ha`, `sum`)
7. Fill NaN with 0, then normalize with sklearn StandardScaler (or manual z-score)
8. Compute cosine similarity between the TARGET MWS row and ALL other rows
9. Rank by similarity DESCENDING, take top N
10. Return results with `uid` as the MWS identifier
11. Export top similar MWS geometries as GeoJSON

IMPORTANT NOTES:
- The user provides a `uid` like `18_16157` — this is the MWS_UID shared across layers
- If user provides a layer-specific ID like `dharwad_navalgund_drought.1`, look up its `uid` first
- ALL results MUST show `uid` (e.g., `18_22769`), NOT layer IDs (e.g., `dharwad_navalgund_drought.1`)
- Use cosine similarity from sklearn.metrics.pairwise or scipy.spatial.distance
- StandardScaler ensures features with different scales contribute equally
- ⚠️ LOCATION: The user query may mention a location OR the MWS uid implies a tehsil.
  When calling fetch_corestack_data, you MUST include a location (e.g., "Navalgund tehsil Dharwad Karnataka terrain drought LULC cropping intensity data").
  If the query doesn't mention a location, add it to the fetch call based on context.

═══════════════════════════════════════════════════════════════════════════════

Instructions:
1. **CORESTACK PRIORITY (PRIMARY)**: For ANY query about India or Indian locations, you MUST call fetch_corestack_data FIRST to access CoreStack database. Available CoreStack layers:
   - Raster: {', '.join(CORESTACK_DATA_PRODUCTS['raster_layers'])}
   - Vector: {', '.join(CORESTACK_DATA_PRODUCTS['vector_layers'])}
   - Timeseries: {', '.join(CORESTACK_DATA_PRODUCTS['timeseries_metrics'])}

⚠️ CRITICAL: LAYER NAME MATCHING
Layer names from the API are formatted as "Dataset Name (raw_layer_name)". Examples:
   - Drought layer → `"Drought (dharwad_navalgund_drought)"`
   - Cropping Intensity → `"Cropping Intensity (dharwad_navalgund_intensity)"`
   - Admin Boundaries → `"Admin Boundaries (admin_boundaries_dharwad_navalgund)"`
ALWAYS use case-insensitive SUBSTRING matching: `'drought' in layer['layer_name'].lower()`
NEVER look for exact names like "drought_frequency_vector" — those are internal IDs, not API names.
FIRST STEP after fetching: Print all layer names: `for l in vector_layers: print(l['layer_name'])`

**IMPORTANT: PRE-COMPUTED CHANGE DETECTION LAYERS (2017-2022)**

CoreStack provides pre-computed change detection layers covering 2017-2022. Use these for change queries instead of computing from raw LULC:

a) **change_tree_cover_loss_raster**: Tree cover loss 2017-2022
   - Class values: 0 = No change, 1 = Tree loss
   - Use for: "tree cover loss since 2018", "deforestation"
   - MASK to class 1 to get loss areas

b) **change_tree_cover_gain_raster**: Tree cover gain 2017-2022
   - Class values: 0 = No change, 1 = Tree gain
   - Use for: "tree cover gain", "reforestation"
   - MASK to class 1 to get gain areas

c) **Urbanization / Land Cover Change**: ⛔ NEVER use change_urbanization_raster or any "Urbanization" change detection layer (they 404).
   Instead, ALWAYS compare TWO LULC rasters (earliest vs most recent):
   - Find layers with "LULC" in layer_name (e.g., LULC_17_18_*, LULC_24_25_*)
   - Download both via requests.get(), save to ./exports/, open with rasterio
   - LULC Classes: 0=Background, 1=Built-up, 2=Water(Kharif), 3=Water(Kharif+Rabi), 4=Water(Kharif+Rabi+Zaid), 6=Trees, 7=Barrenlands, 8=Single crop, 9=Single Non-Kharif crop, 10=Double crop, 11=Triple crop, 12=Shrub_Scrub
   - Reproject to EPSG:32643 for area in hectares
   - Compare pixel classes, compute per-class area for both years
   - Export: change GeoTIFF + area stats CSV + PNG visualization

d) **change_cropping_reduction_raster**: Cropland degradation 2017-2022
   - Shows areas where cropping intensity decreased

e) **change_cropping_intensity_raster**: Cropping intensity transitions 2017-2022
   - Shows how cropping patterns changed (single→double, double→triple, etc.)

f) **cropping_intensity_vector**: Vector layer with yearly attributes
   - Contains cropping intensity values for EACH YEAR (2017-2022+)
   - Use for: "cropping intensity over years", "temporal trends in cropping"
   - IMPORTANT: Year data in columns like cropping_intensity_2017, cropping_intensity_2018, etc.
   - ALWAYS extract year using regex from column names (search for 4-digit numbers)

g) **surface_water_bodies_vector**: Water bodies with temporal attributes
   - Has seasonal availability and area over years
   - Use for: "surface water over years", "water availability trends"

h) **drought layer** (also known as drought_frequency_vector): Drought severity mapping
   - ⚠️ The ACTUAL layer name from the API is `"Drought (dharwad_navalgund_drought)"` — NOT "drought_frequency"
   - Match with: `'drought' in layer['layer_name'].lower()` (case-insensitive substring match)
   - Use for: "drought affected areas", "drought frequency", "drought sensitivity"
   - ⚠️ CRITICAL: You MUST call fetch_corestack_data ONCE and use ALL needed layers from the SAME response.
   - The response already contains ALL layers (29+ vectors). Do NOT call fetch_corestack_data twice.
   - Workflow (follow EXAMPLE 3 in the code examples below):
     1. Call fetch_corestack_data ONCE → get vector_layers list
     2. Find the layer with 'drought' in layer_name → load as drought_gdf
     3. Find the layer with 'admin' AND 'boundar' in layer_name → load as admin_gdf
     4. Print admin_gdf.columns to discover the village name column
     5. The village name column is **vill_name** (text names). NEVER use vill_ID (numeric IDs) or geometry columns.
     6. Spatial join: `gpd.sjoin(drought_gdf, admin_gdf[['vill_name', 'geometry']], how='left', predicate='intersects')`
     7. Get unique village names: `sorted(joined['vill_name'].unique().tolist())`
     8. Dissolve by village name for GeoJSON export
   - Output: list of unique village names + a GeoJSON with one polygon per village (dissolved from microwatersheds)

**IMPORTANT: MICROWATERSHED-LEVEL DATA**:
CoreStack data is provided at **microwatershed (MWS) level**, NOT village level. Each polygon represents a small watershed area within the tehsil. When analyzing a village:
1. The data contains ALL microwatersheds in the tehsil covering the village area
2. NO village name column exists in drought/cropping layers - data is at finer granularity
3. To get village names: Load **admin_boundaries_vector** separately and do a spatial join
4. Use `uid` column for microwatershed identification
5. For drought queries: ALWAYS resolve to village names via spatial join with admin_boundaries_vector

**MULTI-REGION DATA SUPPORT**:
When village spans multiple tehsils, layers will have MULTIPLE URLs in the 'urls' array.
Each layer has structure: {{'layer_name': str, 'layer_type': str, 'urls': [{{url, tehsil, district, state}}, ...]}}
For multi-region layers: Read ALL URLs, concat GeoDataFrames, then analyze.

2. **CORESTACK USAGE** (for India queries):
	```python
	import json
	import os
	import geopandas as gpd
	import pandas as pd

	# ALWAYS create exports directory first
	os.makedirs('./exports', exist_ok=True)

	result = fetch_corestack_data("your query about India")
	data = json.loads(result)

	if data['success'] and data['data_type'] == 'spatial':
		# Access layer data from result
		vector_layers = data['spatial_data']['vector_layers']
		raster_layers = data['spatial_data']['raster_layers']

		# EXAMPLE 1: TEMPORAL VECTOR (Cropping Intensity Over Years)
		for layer in vector_layers:
			if 'Cropping Intensity' in layer['layer_name']:
				# Handle multi-region: Read all URLs and concat
				all_gdfs = []
				for url_info in layer['urls']:
					gdf = gpd.read_file(url_info['url'])
					all_gdfs.append(gdf)

				# Merge all regions
				merged_gdf = pd.concat(all_gdfs, ignore_index=True)

				# CRITICAL: Extract years from column names (e.g., 'cropping_intensity_2017')
				import re
				year_cols = [col for col in merged_gdf.columns if 'cropping_intensity_' in col and re.search(r'\\d{{4}}', col)]

				# Parse year from column name: 'cropping_intensity_2017' -> 2017
				years_data = []
				for col in sorted(year_cols):
					year_match = re.search(r'(\\d{{4}})', col)
					if year_match:
						year = int(year_match.group(1))
						avg_value = merged_gdf[col].mean()
						years_data.append((year, avg_value))

				# Sort by year and plot
				years_data.sort()
				years_list = [y[0] for y in years_data]
				values_list = [y[1] for y in years_data]

				import matplotlib.pyplot as plt
				plt.figure(figsize=(10, 6))
				plt.plot(years_list, values_list, marker='o')
				plt.xlabel('Year')
				plt.ylabel('Average Cropping Intensity')
				plt.title('Cropping Intensity Over Years')
				plt.savefig('./exports/cropping_intensity_over_years.png')
				print(f"Years: {{years_list}}")
				print(f"Values: {{values_list}}")

		# EXAMPLE 2: CHANGE DETECTION RASTER (Tree Cover Loss)
		for layer in raster_layers:
			if 'Change Tree Cover Loss' in layer['layer_name']:
				import rasterio
				# For rasters, typically use first URL (single region or mosaic)
				url = layer['urls'][0]['url']
				with rasterio.open(url) as src:
					loss_data = src.read(1)
					# Mask to class 1 (loss areas)
					loss_mask = (loss_data == 1)
					loss_area_pixels = loss_mask.sum()
					# Convert to hectares using pixel size
					pixel_area = src.transform[0] * abs(src.transform[4])
					loss_area_ha = loss_area_pixels * pixel_area / 10000
					print(f"Tree cover loss: {{loss_area_ha:.2f}} hectares")

		# EXAMPLE 3: DROUGHT + VILLAGE NAMES (Spatial Join with Admin Boundaries)
		# Step 1: Find and load drought_frequency_vector
		drought_gdf = None
		for layer in vector_layers:
			if 'drought' in layer['layer_name'].lower():
				all_gdfs = []
				for url_info in layer['urls']:
					gdf = gpd.read_file(url_info['url'])
					all_gdfs.append(gdf)
				drought_gdf = pd.concat(all_gdfs, ignore_index=True).to_crs('EPSG:4326')
				break

		# Step 2: Find and load admin_boundaries_vector
		admin_gdf = None
		for layer in vector_layers:
			if 'admin' in layer['layer_name'].lower() and 'boundar' in layer['layer_name'].lower():
				all_gdfs = []
				for url_info in layer['urls']:
					gdf = gpd.read_file(url_info['url'])
					all_gdfs.append(gdf)
				admin_gdf = pd.concat(all_gdfs, ignore_index=True).to_crs('EPSG:4326')
				break

		if drought_gdf is not None and admin_gdf is not None:
			# Step 3: Use vill_name column (the village name column in admin_boundaries_vector)
			print("Admin boundary columns:", admin_gdf.columns.tolist())
			village_col = 'vill_name'  # This is the correct column with village names
			print(f"Using village column: {{village_col}}")

			# Step 4: Spatial join
			joined = gpd.sjoin(drought_gdf, admin_gdf[[village_col, 'geometry']], how='left', predicate='intersects')
			joined = joined.dropna(subset=[village_col])

			# Step 5: Get unique village names
			unique_villages = sorted(joined[village_col].unique().tolist())
			print(f"Unique drought-affected villages ({{len(unique_villages)}}): {{unique_villages}}")

			# Step 6: Dissolve by village for export GeoJSON
			village_dissolved = joined.dissolve(by=village_col, aggfunc='first').reset_index()
			village_dissolved.to_file('./exports/drought_affected_villages.geojson', driver='GeoJSON')

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 4: DROUGHT-CROPPING SENSITIVITY                      ║
		# ║ ⚠️ For Query Type 5, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ The key column after sjoin is 'id_drought' (NOT mws_id etc.) ║
		# ╚══════════════════════════════════════════════════════════════╝
		# FIRST: Print all vector layer names to find the right ones
		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 1: Find drought layer (match 'drought' in name) and cropping layer (match 'cropping' + 'intensity')
		drought_gdf = None
		crop_gdf = None
		for layer in vector_layers:
			if 'drought' in layer['layer_name'].lower() and drought_gdf is None:
				print(f"Found drought layer: {{layer['layer_name']}}")
				drought_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')
			if 'cropping' in layer['layer_name'].lower() and 'intensity' in layer['layer_name'].lower() and crop_gdf is None:
				print(f"Found cropping intensity layer: {{layer['layer_name']}}")
				crop_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')

		if drought_gdf is not None and crop_gdf is not None:
			import numpy as np
			# IMPORTANT: Print columns of both GDFs to understand the data
			print(f"Drought GDF columns: {{drought_gdf.columns.tolist()}}")
			print(f"Cropping GDF columns: {{crop_gdf.columns.tolist()}}")
			print(f"Drought GDF 'id' column sample: {{drought_gdf['id'].head().tolist()}}")

			# Step 2: Spatial join — lsuffix='drought', rsuffix='crop' makes 'id' → 'id_drought'
			joined = gpd.sjoin(drought_gdf, crop_gdf, how='inner', predicate='intersects', lsuffix='drought', rsuffix='crop')
			print(f"Joined columns: {{joined.columns.tolist()}}")
			print(f"Joined shape: {{joined.shape}}")

			# Step 3: Group by 'id_drought' — this IS the MWS identifier
			# ⚠️ DO NOT use 'mws_id_source', 'mws_id', 'watershed_id' — they do not exist!
			years = range(2017, 2023)
			results = []
			for mws_id, grp in joined.groupby('id_drought'):
				drought_ci = []
				non_drought_ci = []
				for yr in years:
					drysp_col = f'drysp_{{yr}}'
					ci_col = f'cropping_intensity_{{yr}}'
					if drysp_col not in grp.columns or ci_col not in grp.columns:
						continue
					drysp_val = grp[drysp_col].mean()
					ci_val = grp[ci_col].mean()
					if pd.isna(drysp_val) or pd.isna(ci_val):
						continue
					if drysp_val > 0:  # drought year
						drought_ci.append(ci_val)
					else:
						non_drought_ci.append(ci_val)
				if drought_ci and non_drought_ci:
					avg_drought = np.mean(drought_ci)
					avg_non_drought = np.mean(non_drought_ci)
					sensitivity = avg_non_drought - avg_drought  # positive = drops during drought
					results.append({{'id': mws_id, 'avg_ci_drought': avg_drought, 'avg_ci_non_drought': avg_non_drought, 'sensitivity_score': sensitivity}})

			# Step 4: Rank by sensitivity (descending = most sensitive first)
			sens_df = pd.DataFrame(results).sort_values('sensitivity_score', ascending=False)
			top_n = 10
			top = sens_df.head(top_n)
			print(f"Top {{top_n}} most drought-sensitive MWS:\\n{{top}}")

			# Step 5: Export top MWS geometries
			top_gdf = drought_gdf[drought_gdf['id'].isin(top['id'].tolist())].copy()
			top_gdf = top_gdf.merge(top[['id', 'sensitivity_score']], on='id')
			top_gdf.to_file('./exports/top_sensitive_microwatersheds.geojson', driver='GeoJSON')

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 5: SURFACE WATER SENSITIVITY TO DROUGHT             ║
		# ║ ⚠️ For Query Type 6, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ Uses TABULAR merge on uid == MWS_UID (NOT spatial join)      ║
		# ╚══════════════════════════════════════════════════════════════╝
		# FIRST: Print all vector layer names
		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 1: Find drought layer (NOT causality) and surface water layer (NOT zoi)
		drought_gdf = None
		sw_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'drought' in lname and 'causality' not in lname and drought_gdf is None:
				print(f"Found drought layer: {{layer['layer_name']}}")
				drought_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')
			if 'surface water' in lname and 'zoi' not in lname and sw_gdf is None:
				print(f"Found surface water layer: {{layer['layer_name']}}")
				sw_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')

		if drought_gdf is not None and sw_gdf is not None:
			import numpy as np
			# IMPORTANT: Print columns of both GDFs
			print(f"Drought GDF columns: {{drought_gdf.columns.tolist()}}")
			print(f"Surface water GDF columns: {{sw_gdf.columns.tolist()}}")
			print(f"Surface water shape: {{sw_gdf.shape}}")
			print(f"Drought GDF 'uid' sample: {{drought_gdf['uid'].head().tolist()}}")
			print(f"Surface water 'MWS_UID' sample: {{sw_gdf['MWS_UID'].head().tolist()}}")

			# Step 2: Aggregate surface water area by MWS_UID
			area_cols = [c for c in sw_gdf.columns if c.startswith('area_') and '-' in c]
			print(f"Area columns found: {{area_cols}}")
			sw_by_mws = sw_gdf.groupby('MWS_UID')[area_cols].sum().reset_index()
			print(f"Aggregated {{len(sw_by_mws)}} MWS from {{len(sw_gdf)}} water bodies")

			# Step 3: Tabular merge drought with aggregated surface water
			merged = drought_gdf.merge(sw_by_mws, left_on='uid', right_on='MWS_UID', how='inner')
			print(f"Merged shape: {{merged.shape}}")

			# Step 4: Year mapping — drought calendar year → surface water hydro-year
			year_map = {{
				2017: 'area_17-18', 2018: 'area_18-19', 2019: 'area_19-20',
				2020: 'area_20-21', 2021: 'area_21-22', 2022: 'area_22-23'
			}}

			# Step 5: For each MWS, compute surface water sensitivity
			results = []
			for idx, row in merged.iterrows():
				drought_sw = []
				non_drought_sw = []
				for yr, area_col in year_map.items():
					drysp_col = f'drysp_{{yr}}'
					if drysp_col not in merged.columns or area_col not in merged.columns:
						continue
					drysp_val = row[drysp_col]
					sw_val = row[area_col]
					if pd.isna(drysp_val) or pd.isna(sw_val):
						continue
					if drysp_val > 0:  # drought year
						drought_sw.append(sw_val)
					else:
						non_drought_sw.append(sw_val)
				if drought_sw and non_drought_sw:
					avg_drought = np.mean(drought_sw)
					avg_non_drought = np.mean(non_drought_sw)
					sensitivity = avg_non_drought - avg_drought  # positive = water drops during drought
					results.append({{'id': row['id'], 'uid': row['uid'], 'avg_sw_drought': avg_drought, 'avg_sw_non_drought': avg_non_drought, 'sensitivity_score': sensitivity}})

			# Step 6: Rank by sensitivity (descending = most sensitive first)
			sens_df = pd.DataFrame(results).sort_values('sensitivity_score', ascending=False)
			top_n = 10
			top = sens_df.head(top_n)
			print(f"Top {{top_n}} MWS by surface water sensitivity to drought:\n{{top}}")

			# Step 7: Export top MWS geometries
			top_gdf = drought_gdf[drought_gdf['id'].isin(top['id'].tolist())].copy()
			top_gdf = top_gdf.merge(top[['id', 'sensitivity_score']], on='id')
			top_gdf.to_file('./exports/top_sw_sensitive_microwatersheds.geojson', driver='GeoJSON')

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 6: MWS SIMILARITY CLUSTERING                         ║
		# ║ ⚠️ For Query Type 8, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ Loads 4 layers, merges on uid, cosine similarity             ║
		# ╚══════════════════════════════════════════════════════════════╝
		# FIRST: Print all vector layer names
		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 1: Load 4 layers — drought, cropping, terrain, LULC
		drought_gdf = None
		crop_gdf = None
		terrain_gdf = None
		lulc_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'drought' in lname and 'causality' not in lname and drought_gdf is None:
				print(f"Found drought layer: {{layer['layer_name']}}")
				drought_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)
			if 'cropping' in lname and 'intensity' in lname and crop_gdf is None:
				print(f"Found cropping layer: {{layer['layer_name']}}")
				crop_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)
			if 'terrain' in lname and 'lulc' not in lname and 'raster' not in lname and terrain_gdf is None:
				print(f"Found terrain layer: {{layer['layer_name']}}")
				terrain_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)
			if 'lulc' in lname and 'terrain' not in lname and 'level' not in lname and lulc_gdf is None:
				print(f"Found LULC layer: {{layer['layer_name']}}")
				lulc_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)

		# Step 2: Print columns
		for name, gdf in [('Drought', drought_gdf), ('Cropping', crop_gdf), ('Terrain', terrain_gdf), ('LULC', lulc_gdf)]:
			if gdf is not None:
				print(f"{{name}} columns ({{len(gdf.columns)}}): {{gdf.columns.tolist()[:15]}}...")

		# Step 3: Drop geometry from all except drought_gdf, then merge on 'uid'
		merged = drought_gdf.drop(columns=['geometry'])
		for gdf in [crop_gdf, terrain_gdf, lulc_gdf]:
			if gdf is not None:
				cols_to_drop = ['geometry', 'id', 'area_in_ha', 'sum']
				gdf_clean = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])
				merged = merged.merge(gdf_clean, on='uid', how='inner', suffixes=('', f'_{{gdf.columns[2][:4]}}'))
		print(f"Merged shape: {{merged.shape}}")

		# Step 4: Select numeric columns only (exclude id, uid, geometry, etc.)
		exclude_cols = ['id', 'uid', 'geometry', 'area_in_ha', 'sum', 'terrainClu']
		feature_cols = [c for c in merged.columns if c not in exclude_cols and merged[c].dtype in ['float64', 'int64', 'float32', 'int32']]
		print(f"Feature columns ({{len(feature_cols)}}): {{feature_cols[:20]}}...")

		# Step 5: Build feature matrix and normalize
		import numpy as np
		from sklearn.preprocessing import StandardScaler
		from sklearn.metrics.pairwise import cosine_similarity

		X = merged[feature_cols].fillna(0).values
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		# Step 6: Find the target MWS by uid
		target_uid = '18_16157'  # Replace with the uid from the query
		target_idx = merged.index[merged['uid'] == target_uid]
		if len(target_idx) == 0:
			print(f"Target uid {{target_uid}} not found! Available uids: {{merged['uid'].tolist()[:10]}}")
		else:
			target_idx = target_idx[0]
			target_vec = X_scaled[target_idx].reshape(1, -1)

			# Step 7: Cosine similarity between target and all
			sims = cosine_similarity(target_vec, X_scaled)[0]
			merged['similarity'] = sims

			# Exclude the target itself, sort descending
			others = merged[merged['uid'] != target_uid].sort_values('similarity', ascending=False)
			top_n = 10
			top = others.head(top_n)[['uid', 'similarity']]
			print(f"Top {{top_n}} most similar MWS to {{target_uid}}:")
			print(top.to_string())

			# Step 8: Export GeoJSON with geometries — CLEAN output
			top_uids = top['uid'].tolist()
			top_gdf = drought_gdf[drought_gdf['uid'].isin(top_uids + [target_uid])].copy()
			top_gdf = top_gdf.merge(merged[['uid', 'similarity']], on='uid', how='left')
			# Drop GeoServer 'id' column (e.g. dharwad_navalgund_drought.17) — it is NOT the MWS uid
			if 'id' in top_gdf.columns:
				top_gdf = top_gdf.drop(columns=['id'])
			# Keep only key summary columns + similarity + geometry (drop raw weekly rd/drlb columns)
			keep_cols = ['uid', 'similarity', 'area_in_ha', 'avg_dryspell', 'geometry']
			# Also keep yearly summary columns (drysp_, w_sev_, w_mod_, w_mld_, cropping_intensity_, hill_slope, etc.)
			for c in top_gdf.columns:
				if any(c.startswith(p) for p in ['drysp_', 'w_sev_', 'w_mod_', 'w_mld_', 'cropping_intensity', 'hill_slope', 'plain_area', 'ridge_area', 'slopy_area', 'valley_are']):
					keep_cols.append(c)
			keep_cols = [c for c in keep_cols if c in top_gdf.columns]
			top_gdf = top_gdf[keep_cols]
			top_gdf.to_file('./exports/similar_microwatersheds.geojson', driver='GeoJSON')

	elif data['success'] and data['data_type'] == 'timeseries':
		# Access timeseries data
		timeseries = data['timeseries_data']
		# Process timeseries for temporal analysis
	```
3. **EARTH ENGINE (SUPPLEMENTARY)**: ONLY use Earth Engine if CoreStack doesn't have the required data or for non-India queries. When using Earth Engine:
   - Initialize with: `ee.Initialize(project='apt-achievment-453417-h6')`
   - Use harmonized Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) and harmonized Landsat (LANDSAT/LC08/C02/T1_L2)
   - Export to local machine using geedim:
	```python
	import geedim as gd
	gd_image = gd.MaskedImage(ee_image)
	gd_image.download(filename=output_filename, scale=scale, region=ee_geom, crs='EPSG:4326')
	```
4. You can also use OpenStreetMap (osmnx) for additional context like roads, buildings, etc.
5. Then, you should write python code to perform the task.
6. When reading up vector data, always use to_crs method to convert to EPSG:4326.
7. **CRS HANDLING (CRITICAL)**:
   - CoreStack data: EPSG:4326 (WGS84 lat/lon)
   - India UTM Zone: EPSG:32643 (for area calculations)
   - For area: ALWAYS reproject to EPSG:32643 BEFORE calculating
   - For distance: Use geodesic calculations (geopy or shapely ops)
8. Pay extra attention to CRS of the data products, verify them manually before using them in analysis.
9. Always use actual data sources to perform analysis, do not use dummy/sample data.
10. **EXPORT FORMATS**:
	- All vector data should be exported in GeoJSON
	- All raster data should be exported in GeoTIFF
	- All visualizations should be exported in PNG
	- **CRITICAL**: All exports must be saved to `./exports/` directory (use relative path)
	- **FIRST STEP**: Always create exports directory: `import os; os.makedirs('./exports', exist_ok=True)`
	- Use paths like: `'./exports/output.png'`, `'./exports/data.geojson'`, etc.

**EXPECTED OUTPUT TYPES** (based on query type):
- Time series plots: Line charts showing temporal trends (e.g., cropping intensity over years, surface water over years)
- Change rasters: GeoTIFF files with change detection (e.g., tree cover change, cropland to built-up conversion) + total area statistics in hectares
- Filtered vectors: GeoJSON files with spatial filtering (e.g., villages with drought, high sensitivity microwatersheds)
- Village-level drought: Unique village names with drought stats, exported as GeoJSON (dissolved by village)
- Rankings: CSV or tables showing ranked microwatersheds/villages by various dimensions
- Similarity analysis: Top-K similar microwatersheds based on multiple attributes
- Scatterplots: 2D plots showing relationships between variables, with quadrant analysis where applicable

**IMPORTANT NOTES**:
- Access layer data directly from the tool's JSON response: `data['spatial_data']['vector_layers']`
- Get layer by name by iterating through the layers list and matching the name
- ALWAYS reproject to UTM (EPSG:32643) before area calculations
- **CRITICAL**: Save all exports to `./exports/` directory (relative path). Create directory first with `os.makedirs('./exports', exist_ok=True)`
- NEVER use `/app/exports/` path - that's for Docker only
- NEVER create dummy or fake data - ALWAYS use the actual data from fetch_corestack_data
- The tool returns real data in 'Execution logs' - parse and use that data directly

Make sure to wrap your final answer along with expected outputs as code block with a single-line string with \\n delimiters inside final_answer function. For example, your final response should look like:

```py
final_answer("The final answer is .... .\\n Exports:  \\n- export1: ./exports/export1.some_ext  \\n- export2: ./exports/export2.some_ext")
```

Task: {task}
"""


# ============================================================================
# HYBRID AGENT (CodeAct + CoreStack Tool)
# ============================================================================

def run_hybrid_agent(user_query: str, exports_dir: str = None):
	"""
	Run CodeAct agent with CoreStack tool + local execution (like agent.py).

	Uses Gemini model with local executor for full geospatial library support.
	Structure exactly mirrors agent.py but with CoreStack tool added.
	"""
	# Set up absolute exports directory
	if exports_dir is None:
		exports_dir = os.path.abspath("./exports")
	os.makedirs(exports_dir, exist_ok=True)

	# Get workspace directory
	workspace_dir = os.path.dirname(os.path.abspath(__file__))

	print("\n" + "="*70)
	print("🚀 HYBRID AGENT (CodeAct + CoreStack Tool)")
	print(f"📝 Query: {user_query}")
	print("="*70)

	# Use LiteLLM for Gemini (smolagents compatible)
	model = LiteLLMModel(
		model_id="gemini/gemini-2.5-flash-lite",
		api_key=os.getenv("GEMINI_API_KEY")
	)

	# Create tools list - NOTE: Only CoreStack tool, NO web_search to force using it
	tools = [fetch_corestack_data]

	# Use local Python executor
	print("✅ Using local Python executor")
	agent = CodeAgent(
		model=model,
		tools=tools,
		additional_authorized_imports=["*"]
	)

	trace = _start_trace(user_query, model.model_id)
	trace_token = _LANGFUSE_TRACE.set(trace)

	stdout_buffer = io.StringIO()
	tee_stdout = _TeeStdout(sys.stdout, stdout_buffer)

	try:
		# Generate prompt and run agent
		prompt = create_corestack_prompt(user_query)

		model_span = _start_span(trace, "model_generation", input_data=prompt, metadata={
			"model_id": model.model_id
		})

		exec_span = _start_span(trace, "code_execution", input_data={"user_query": user_query})

		with redirect_stdout(tee_stdout):
			result = agent.run(prompt)

		execution_logs = stdout_buffer.getvalue()
		generated_code = _extract_code_blocks(execution_logs)

		_end_span(model_span, output_data=result, metadata={
			"prompt": prompt,
			"generated_code": generated_code
		})
		_end_span(exec_span, output_data=result, metadata={
			"execution_logs": execution_logs
		})

		print("\n" + "="*70)
		print("✅ AGENT COMPLETED")
		print("="*70)
		print(result)
		print("="*70)

		_update_trace(trace, output_data=result, metadata={
			"execution_logs": execution_logs,
			"generated_code": generated_code
		})

		return result

	except Exception as e:
		error_msg = f"Agent execution failed: {str(e)}"
		print(f"\n❌ ERROR: {error_msg}")
		error_details = traceback.format_exc()
		_update_trace(trace, output_data=error_msg, metadata={
			"error": str(e),
			"traceback": error_details
		})
		raise e
	finally:
		_LANGFUSE_TRACE.reset(trace_token)


# ============================================================================
# EXAMPLE USAGE & COMPARISON
# ============================================================================

if __name__ == "__main__":
	"""
	Example usage of the hybrid agent with a test query.
	"""

	print("Bot TEST")
	print("="*70)


	# print("Running query #1 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Could you show how cropping intensity has changed over the years in Navalgund, Dharwad, Karnataka?")

	# print("Running query #2 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Could you show how surface water availability has changed over the years in Navalgund, Dharwad, Karnataka?")

	# print("Running query #3 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Can you show me areas that have lost tree cover in Navalgund, Dharwad, Karnataka since 2018? also hectares of degraded area?")

    # print("Running query #4 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("How much cropland in Navalgund, Dharwad, Karnataka has turned into built up since 2018? can you show me those regions?")

    # print("Running query #5 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Which villages in Navalgund, Dharwad Karnataka among the ones available on Core Stack have experienced droughts? ")

	# print("Running query #6 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest cropping senstivity to drought")

	# print("Running query #7 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest surface water availability senstivity to drought")


	print("Running query #8 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	print("="*70)
	run_hybrid_agent("find me microwatersheds in Navalgund tehsil, Dharwad district, Karnataka most similar to 18_16157 uid microwatershed, based on its terrain, drought frequency, LULC and cropping intensity")


	# print("Running query #7 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest surface water availability senstivity to drought")


	# print("Running query #7 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest surface water availability senstivity to drought")

