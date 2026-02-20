"""
Hybrid Architecture: CodeAct + CoreStack Tool
==============================================
"""

import os
import io
import re
import sys
import json
import uuid
import traceback
import warnings
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import redirect_stdout

# Suppress GeoPandas/Shapely geographic CRS centroid warning
# (agent-generated code computes centroids in WGS84 for GEE point sampling — acceptable accuracy)
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*centroid.*")

from dotenv import load_dotenv

# Import CodeAct from smolagents (structure from agent.py)
from smolagents import CodeAgent, tool, LiteLLMModel, DuckDuckGoSearchTool

# No need to import executor - it's specified via executor_type parameter
DOCKER_AVAILABLE = True  # Assume Docker is available like in agent.py

# Import Earth Engine
import ee

load_dotenv()

# Get API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CORE_STACK_API_KEY = os.getenv("CORE_STACK_API_KEY")
CORESTACK_BASE_URL = os.getenv("CORESTACK_BASE_URL", "https://geoserver.core-stack.org/api/v1")

# Langfuse keys (read from .env via LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)
# These are auto-read by the Langfuse SDK from env vars — no manual assignment needed.

# Initialize Earth Engine
GEE_PROJECT = os.getenv("GEE_PROJECT")
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
		"change_detection_deforestation_raster",
		"change_detection_afforestation_raster",
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
# LANGFUSE OBSERVABILITY (full lifecycle tracing)
# ============================================================================
#
# Trace hierarchy visible in the Langfuse dashboard:
#
#   Session (session_id)                     ← groups user turns
#   └─ Trace: run_hybrid_agent               ← root per user request
#        ├─ Span: corestack_api_request      ← each CoreStack HTTP call
#        ├─ Span: location_resolution        ← geo-resolution workflow
#        ├─ Span: fetch_corestack_data       ← tool execution
#        ├─ Generation: llm_call             ← each model inference
#        ├─ Span: agent_reasoning_step       ← code execution steps
#        └─ Span: final_response             ← answer packaging
#
# SmolagentsInstrumentor (OpenTelemetry) auto-traces every LLM call, tool
# execution, code block, and error inside CodeAgent.  The module below adds
# structured spans, session tracking, feedback scoring, cost monitoring,
# prompt versioning, and error capture on top of that.
# ============================================================================

from langfuse_observability import (
	lf_client,
	observe,
	is_enabled as _langfuse_is_enabled,
	generate_session_id,
	generate_trace_id,
	set_trace_metadata,
	set_trace_output,
	trace_llm_call,
	trace_tool_call,
	trace_agent_execution,
	span as lf_span,
	generation as lf_generation,
	score_trace,
	score_trace_by_id,
	log_error_to_trace,
	log_error_to_trace_root,
	tag_prompt_version,
	set_prompt_version,
	get_prompt_version,
	record_generation_cost,
	flush as lf_flush,
	shutdown as lf_shutdown,
)

_LANGFUSE_ENABLED = _langfuse_is_enabled()

# ── SmolagentsInstrumentor (OpenTelemetry auto-tracing) ──
try:
	from openinference.instrumentation.smolagents import SmolagentsInstrumentor
	SmolagentsInstrumentor().instrument()  # must run BEFORE any CodeAgent
	print("✅ SmolagentsInstrumentor active — auto-tracing LLM / tool / code steps")
except Exception as _otel_err:
	print(f"⚠️  SmolagentsInstrumentor unavailable: {_otel_err}")


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


@observe(name="corestack_api_request", as_type="span")
def _corestack_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
	"""HTTP GET to CoreStack API — traced as a child span in Langfuse."""
	import time as _time
	import requests

	if not CORE_STACK_API_KEY:
		raise RuntimeError("CORE_STACK_API_KEY is not set")

	url = f"{CORESTACK_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
	headers = {"X-API-Key": CORE_STACK_API_KEY}

	# ── capture request metadata on the span ──
	lf_client.update_current_span(
		input={"endpoint": endpoint, "params": params, "url": url},
		metadata={"http_method": "GET"},
	)

	t0 = _time.perf_counter()
	try:
		response = requests.get(url, headers=headers, params=params, timeout=30)
		response.raise_for_status()
		payload = response.json()
	except ValueError as exc:
		log_error_to_trace(exc, context=f"Non-JSON from {endpoint}")
		raise RuntimeError(f"CoreStack API returned non-JSON response from {endpoint}") from exc
	except Exception as exc:
		log_error_to_trace(exc, context=f"API call to {endpoint}")
		raise

	elapsed_ms = (_time.perf_counter() - t0) * 1000

	if isinstance(payload, dict) and payload.get("success") is False:
		message = payload.get("message") or payload.get("error") or "Unknown error"
		err = RuntimeError(f"CoreStack API error from {endpoint}: {message}")
		log_error_to_trace(err, context=endpoint)
		raise err

	# ── record response metadata ──
	lf_client.update_current_span(
		output={"status_code": response.status_code, "payload_type": type(payload).__name__},
		metadata={"latency_ms": round(elapsed_ms, 2), "status_code": response.status_code},
	)
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


@observe(name="corestack_workflow", as_type="span")
def _run_corestack_workflow(user_query: str) -> Dict[str, Any]:
	"""Full CoreStack location-resolution + layer-fetch workflow (traced)."""
	lf_client.update_current_span(input={"user_query": user_query})
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

	import time as _time
	t0 = _time.perf_counter()

	try:
		# Run workflow (traced via @observe on _run_corestack_workflow)
		result_state = _run_corestack_workflow(query)

		# Check for errors
		if "error" in result_state:
			response = json.dumps({
				"success": False,
				"error": result_state["error"]
			}, indent=2)
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

		elapsed_ms = (_time.perf_counter() - t0) * 1000
		n_vector = len(available_layers.get('vector', []))
		n_raster = len(available_layers.get('raster', []))

		print(f"\n✅ FETCH COMPLETE:")
		print(f"   Vector layers: {n_vector}")
		print(f"   Raster layers: {n_raster}")

		# ── Record tool output on current observation ──
		lf_client.update_current_span(
			output={"vector_layers": n_vector, "raster_layers": n_raster, "location": location_info},
			metadata={"tool_name": "fetch_corestack_data", "latency_ms": round(elapsed_ms, 2)},
		)

		response = json.dumps(response_obj, default=str)
		return response

	except Exception as e:
		print(f"\n❌ ERROR: {str(e)}")
		traceback.print_exc()
		log_error_to_trace(e, context="fetch_corestack_data")
		response = json.dumps({
			"success": False,
			"error": str(e)
		}, indent=2)
		return response


# ============================================================================
# CODEACT PROMPT (ADAPTED FROM agent.py)
# ============================================================================

def create_corestack_prompt(task: str) -> str:
	"""Compressed CodeAct prompt — CoreStack primary, EE supplementary."""
	return f"""
You are a geospatial analysis agent. Libraries: osmnx, geopandas, shapely, matplotlib, numpy, pandas, rasterio, ee, geemap, geedim, geopy, requests, json, sklearn.


═══ §1 CORE PRINCIPLES: SPATIAL vs TIMESERIES ═══

CoreStack has TWO data structures:
1. SPATIAL (Vectors/Rasters): Location-specific features with SPATIAL VARIATION. Temporal data as YEARLY COLUMNS (e.g., cropping_intensity_2017..2023). Use for village/tehsil analysis.
2. TIMESERIES (Watershed MWS): Single aggregated value per watershed per time period. Fortnightly. NO spatial variation. Use ONLY for watershed water budget.

DECISION: If query is about cropping, land use, water bodies, drought, terrain → use SPATIAL. If about watershed water balance (runoff, precip, ET fortnightly) → TIMESERIES.

═══ §2 GLOBAL MANDATORY RULES ═══

G1. SINGLE FETCH: Call fetch_corestack_data() ONCE per task. Response contains ALL layers (29+ vectors, multiple rasters). NEVER call it twice.
G2. LAYER MATCHING: ALWAYS use case-insensitive SUBSTRING: `'keyword' in layer['layer_name'].lower()`. NEVER use exact string matching. NEVER search for internal IDs. Example:
```python
admin_layer = None
nrega_layer = None
for layer in vector_layers:
    n = layer['layer_name'].lower()
    if 'admin' in n: admin_layer = layer
    if 'nrega' in n: nrega_layer = layer
admin_gdf = pd.concat([gpd.read_file(u['url']) for u in admin_layer['urls']], ignore_index=True)
```

G3. PRINT FIRST: After fetching, ALWAYS print all layer names: `for l in layers: print(l['layer_name'])`
G4. LOAD PATTERN: `gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)`
G5. CRS: Always `.to_crs('EPSG:4326')` after loading. CoreStack data is EPSG:4326.
G6. EXPORTS DIR: First step: `import os; os.makedirs('./exports', exist_ok=True)`. All outputs to `./exports/`. NEVER `/app/exports/`.
G7. EXPORT FORMATS: Vectors→GeoJSON, Rasters→GeoTIFF, Visualizations→PNG.
G8. REAL DATA ONLY: NEVER create dummy/fake data. ALWAYS use actual data from fetch_corestack_data.
G9. MULTI-REGION: When layer has multiple URLs, read ALL, concat GDFs: `pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)`
G10. MWS IDENTIFIER: Use `uid` column (e.g., '18_16157') for microwatershed ID. NEVER use GeoServer `id` (e.g., 'dharwad_navalgund_drought.1'). Drop `id` from exports.
G11. FINAL ANSWER: `final_answer("Result...\\\\nExports:\\\\n- ./exports/file.ext")`
G12. PRINT COLUMNS: After loading ANY GeoDataFrame, ALWAYS `print(gdf.columns.tolist())` BEFORE accessing columns. NEVER guess column names.

⛔ SANDBOX CONSTRAINTS (violations = instant failure):
S1. NEVER use bare `open()` for rasters — use `rasterio.MemoryFile(bytes_data)` to read raster bytes.
S2. NEVER reproject rasters to UTM — LULC rasters are 4911×5826 px, UTM reproject causes 300+PB memory crash.
S3. DEGREE-BASED AREA: `px_area_ha = (dx_deg * 111320 * cos(lat_rad)) * (dy_deg * 110540) / 10000`
S4. For raster clipping: MUST `from rasterio.mask import mask` then `mask(src, geoms, crop=True)`. Also `from shapely.geometry import mapping as shapely_mapping; geoms = [shapely_mapping(gdf.geometry.iloc[0])]` — NEVER pass raw function or `.__geo_interface__`.
S5. `rasterio.open(path, 'w', ...)` IS allowed (module method). `plt.savefig(...)` IS allowed.

═══ §3 COMMON PATTERNS (reference by name) ═══

**VILLAGE_BOUNDARY** — Get village boundary from OSM with fallback:
```python
import osmnx as ox
geocode_queries = [
    f"{{village}}, {{tehsil}}, {{district}}, {{state}}, India",  # full
    f"{{village}}, {{district}}, {{state}}, India",              # skip tehsil (OSM spellings differ)
    f"{{village}}, {{state}}, India",                             # last resort
]
village_gdf = None
for gq in geocode_queries:
    try:
        village_gdf = ox.geocode_to_gdf(gq)
        village_gdf = village_gdf.to_crs(data_gdf.crs)
        break
    except Exception:
        continue
if village_gdf is None:  # geopy fallback → ~2km buffer
    from geopy.geocoders import Nominatim; from shapely.geometry import Point
    geolocator = Nominatim(user_agent="corestack_agent")
    for gq in geocode_queries:
        try:
            loc = geolocator.geocode(gq)
            if loc:
                village_gdf = gpd.GeoDataFrame(geometry=[Point(loc.longitude, loc.latitude).buffer(0.02)], crs="EPSG:4326").to_crs(data_gdf.crs)
                break
        except Exception:
            continue
if village_gdf is None:
    print("WARNING: geocode failed. Using full tehsil."); village_name = "Tehsil"
```
Apply for vectors: `filtered_gdf = gpd.overlay(data_gdf, village_gdf[['geometry']], how='intersection')`
Apply for rasters: Always buffer the village boundary by ~2km before clipping: `village_gdf['geometry'] = village_gdf.geometry.buffer(0.02)`

**RASTER_CLIP** — Read + clip raster (sandbox-safe):
⚠️ CRITICAL: ALWAYS pass `nodata=0` to `rasterio_mask()` — rasters may have nodata=-9999 which overflows uint8.
```python
import math, requests, numpy as np
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping as shapely_mapping
# Buffer village boundary ~2km for raster clipping
village_gdf['geometry'] = village_gdf.geometry.buffer(0.02)
r_bytes = requests.get(url, timeout=120).content
with rasterio.MemoryFile(r_bytes) as memfile:
    with memfile.open() as src:
        if village_gdf is not None:
            geoms = [shapely_mapping(village_gdf.geometry.iloc[0])]
            # ⚠️ MUST pass nodata=0 — native nodata (-9999) crashes uint8
            data, transform = rasterio_mask(src, geoms, crop=True, nodata=0)
            data = data[0]
        else:
            data = src.read(1); transform = src.transform
        crs = src.crs
# ALWAYS inspect unique values first:
print(f"Unique raster values: {{np.unique(data)}}")
# Area: degree-based (NEVER UTM)
dx, dy = abs(transform[0]), abs(transform[4])
bbox = village_gdf.total_bounds  # [minx, miny, maxx, maxy]
center_lat = (bbox[1] + bbox[3]) / 2
px_area_ha = (dx * 111320 * math.cos(math.radians(center_lat))) * (dy * 110540) / 10000
# Count change pixels and compute area:
change_mask = (data == 1)  # or whatever change class
change_pixel_count = int(np.count_nonzero(change_mask))  # returns int, NOT tuple
total_area_ha = change_pixel_count * px_area_ha
print(f"Change pixels: {{change_pixel_count}}, Area: {{total_area_ha:.2f}} ha")
```


**GEE_CENTROID_SAMPLE** — Sample GEE image at MWS centroids:
```python
gdf['centroid_lon'] = gdf.geometry.centroid.x
gdf['centroid_lat'] = gdf.geometry.centroid.y
ee.Initialize(project='corestack-gee')
ee_roi = ee.Geometry.Rectangle([*map(float, gdf.total_bounds)])
points = [ee.Feature(ee.Geometry.Point([float(r['centroid_lon']), float(r['centroid_lat'])]), {{'uid': str(r['uid'])}}) for _, r in gdf.iterrows()]
fc = ee.FeatureCollection(points)
# ⚠️ CRITICAL: sampleRegions ONLY works on ee.Image, NOT ee.ImageCollection!
# You MUST reduce the collection to a single image FIRST:
image = collection.filterBounds(ee_roi).map(cloud_mask).map(compute_band).mean()  # .mean() reduces ImageCollection → Image
sampled = image.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()
```
⚠️ NEVER pass polygon geometries to GEE — use POINT centroids only.
⚠️ NEVER call `.sampleRegions()` on an `ee.ImageCollection` — ALWAYS reduce to `ee.Image` first with `.mean()` or `.median()`.


═══ §4 LAYER MATCHING RULES ═══

| Layer | Match Rule | Negative Filter |
|---|---|---|
| Cropping Intensity | `'cropping' in n and 'intensity' in n` | — |
| Drought | `'drought' in n` | `'causality' not in n` |
| Surface Water Bodies | `'surface water' in n` | `'zoi' not in n` |
| Admin Boundaries | `'admin' in n` or `'panchayat' in n` | — |
| NREGA Assets | `'nrega' in n` | — |
| Terrain Vector | `'terrain' in n` | `'lulc' not in n and 'raster' not in n` |
| LULC Vector | `'lulc' in n` | `'terrain' not in n and 'level' not in n` |
| LULC Raster (level_3) | `'lulc' in n and 'level_3' in n` | — |
| Deforestation Raster | `'deforestation' in n` | — |
| Afforestation Raster | `'afforestation' in n` | — |
| Degradation Raster | `'degradation' in n` | — |
| Tree Overall Change | `'tree overall' in n` | — |

Where `n = layer['layer_name'].lower()`

⛔ NEVER search for 'tree_cover_loss', 'change_tree_cover_loss', 'change_urbanization', 'drought_frequency' — these names do NOT exist in API.
⛔ NEVER use any "Urbanization" change detection layer (returns 404). For crop→built-up, compare TWO LULC rasters.

Change detection layer naming: `"Change Detection Raster (change_<district>_<tehsil>_<Type>)"`.  E.g., `change_dharwad_kundgol_Deforestation`.

═══ §5 COLUMN REFERENCE ═══

**Cropping Intensity**: `cropping_intensity_YYYY` (2017-2024). 115 MWS, 54 cols.
**Drought**: `drysp_YYYY` (dry spell weeks), `w_sev_YYYY`/`w_mod_YYYY`/`w_mld_YYYY`/`w_no_YYYY` (severity weeks), `avg_dryspell`, `t_wks_YYYY`. 115 MWS, 258 cols. Weekly `rd` columns: `rdYY-M-D` (e.g., rd19-5-21). Parse: year=2000+int(YY), month=int(M). Group by `c[2:4]` for year.
**Surface Water**: ⚠️ Columns use 2-DIGIT year pairs ONLY (NEVER 4-digit): `area_YY-YY` (e.g., `area_17-18`, `area_22-23`). NEVER `area_2017-18`. Find area cols: `[c for c in cols if re.match(r'area_\d{{2}}-\d{{2}}', c)]`. Also: `k_YY-YY` (kharif%), `kr_YY-YY`, `krz_YY-YY`. `MWS_UID` for groupby. Each row = one water body (2798 total). Year range: area_17-18 through area_22-23.
**Terrain**: `uid`, `hill_slope`, `plain_area`, `ridge_area`, `slopy_area`, `valley_are`, `terrainClu`. 115 MWS.
**LULC Vector**: `uid`, 101 cols. Truncated names: `triply_cro`/`triply_c_1..7`, `doubly_cro`/`doubly_c_1..7`, `single_kha`/`single_k_1..7`, `single_non`/`single_n_1..7`, `krz_water_`/`krz_wate_1..7`, `kr_water_a`/`kr_water_1..7`, `k_water_ar`/`k_water__1..7`. Base col=2017, suffixes _1.._7=2018-2024.
**Admin Boundaries**: `vill_name` (village name text), `vill_ID` (numeric), `P_SC`, `P_ST`, `TOT_P`. 138 village polygons.
**NREGA**: `Gram_panch`, `Panchayat`, `Work Name`, `Work Type`, `WorkCatego`, `Total_Expe`, geometry (points). 10797 works.

**LULC Raster Class Codes (level_3)**: 1=Built-up, 2=Water(K), 3=Water(KR), 4=Water(KRZ), 6=Trees, 7=Barrenlands, 8=Single crop, 9=Single non-K crop, 10=Double crop, 11=Triple crop, 12=Shrub_Scrub. Cropland={{8,9,10,11}}.
LULC raster names: `LULC_YY_YY_<district>_<tehsil>_level_3`. Float64 with NaN nodata → `np.nan_to_num(data, nan=0).astype(int)`. Same grid/CRS for comparison.

**Year Mapping (drought→surface water)**: drysp_2017→area_17-18, drysp_2018→area_18-19, ..., drysp_2022→area_22-23.

═══ §6 QUERY TYPE ROUTING ═══

For each query type below: use VILLAGE_BOUNDARY pattern for village filtering, LOAD_LAYER pattern (G4) for loading, and follow the specified steps.

───────────────────────────────────────
**QT1: Cropping Intensity in [Village] Over Years**
Layers: cropping_intensity_vector (spatial)
Steps:
1. Fetch and load CI vector at tehsil level
2. Get village boundary (VILLAGE_BOUNDARY pattern) → spatially intersect: `gpd.overlay(crop_gdf, village_gdf[['geometry']], how='intersection')`
3. Extract year cols: `[c for c in cols if 'cropping_intensity_' in c.lower() and re.search(r'\\d{{4}}', c)]`
4. Compute avg CI per year, plot time-series line chart
5. Compute per-MWS mean_ci, export GeoJSON
Outputs: `./exports/cropping_intensity_over_years_<village>.png`, `./exports/cropping_intensity_by_mws_<village>.geojson`

───────────────────────────────────────
**QT2: Surface Water Availability Over Years in [Village]**
Layers: surface_water_bodies_vector (NOT zoi)
Steps:
1. Load SW layer, get village boundary, spatially intersect
2. Find area columns: `re.match(r'area_\\d{{2}}-\\d{{2}}', col)`
3. Convert to numeric, aggregate by MWS_UID: `sw_gdf.groupby('MWS_UID')[area_cols].sum()`
4. Sum across MWS per year → total water area trend
5. Plot time-series, export per-MWS GeoJSON (dissolve by MWS_UID)
Outputs: `./exports/surface_water_trend.png`, `./exports/surface_water_availability_by_mws.geojson`

───────────────────────────────────────
**QT3: Tree Cover Loss / Deforestation in [Village]**
Layers: deforestation raster + degradation raster (both in raster_layers). Fallback: 'tree overall change' raster.
⚠️ TEHSIL MATCHING: Layer name contains tehsil (e.g., change_dharwad_kundgol_Deforestation). Include CORRECT tehsil in fetch query.
Steps:
1. Find deforestation layer: `'deforestation' in layer['layer_name'].lower()`. Also find degradation: `'degradation' in layer['layer_name'].lower()`
2. Get village boundary (VILLAGE_BOUNDARY pattern), then buffer it: `village_gdf['geometry'] = village_gdf.geometry.buffer(0.02)`
3. For EACH raster (deforestation, degradation):
   a. Download bytes: `r_bytes = requests.get(layer['layer_url'], timeout=120).content`
   b. Use RASTER_CLIP pattern to clip to village
   c. Mask to loss class: `change_mask = (data == 1)`
   d. Count change pixels: `change_count = int(np.count_nonzero(change_mask))` — returns an int, NOT a tuple
   e. Compute area: `area_ha = change_count * px_area_ha`
   f. Save clipped GeoTIFF using `rasterio.open('./exports/...tif', 'w', driver='GTiff', ...)`
4. Print total loss/degradation in hectares for each
Outputs: `./exports/deforestation_change.tif`, `./exports/degradation_change.tif` + print area
FALLBACK (custom years): Use LULC rasters, compare class 6 (trees) across years.

───────────────────────────────────────
**QT4: Cropland → Built-up / Land Cover Change**
Layers: TWO LULC level_3 rasters (earliest + most recent)
⛔ NEVER use "Urbanization" or "change_urbanization" layers (404 error).
Steps:
1. Find LULC level_3 layers, sort by year, pick oldest matching user's year + newest
2. Download BOTH rasters (requests.get), use RASTER_CLIP for each
3. `np.nan_to_num(data, nan=0).astype(int)`
4. Change mask: `np.isin(old, [8,9,10,11]) & (new == 1)` (cropland→built-up)
5. Compute area (degree-based, S3), save change GeoTIFF, print hectares
Outputs: `./exports/crop_to_builtup_change.tif` + print area

───────────────────────────────────────
**QT5: MWS with Highest Cropping Sensitivity to Drought**
Layers: drought + cropping_intensity (both vectors, SAME fetch)
⛔ NEVER use np.corrcoef on w_sev columns (mostly zero → NaN)
⛔ NEVER invent column names like 'mws_id_source', 'mws_id', 'watershed_id'
Steps:
1. Load both layers as GDFs
2. Spatial join: `gpd.sjoin(drought_gdf, crop_gdf, how='inner', predicate='intersects', lsuffix='drought', rsuffix='crop')`
3. Group by `id_drought` (the drought GDF's `id` with lsuffix). Print `joined.columns.tolist()` to verify.
4. Per MWS per year (2017-2022): if `drysp_YYYY > 0` → DROUGHT year, else NON-DROUGHT
5. sensitivity_score = mean_ci_non_drought - mean_ci_drought (positive = drop during drought)
6. Rank descending, take top N, resolve to village names via admin_boundaries sjoin (vill_name col)
7. Export top MWS geometries as GeoJSON
Output: `./exports/top_sensitive_microwatersheds.geojson`

───────────────────────────────────────
**QT6: MWS with Highest Surface Water Sensitivity to Drought**
Layers: drought (NOT causality) + surface_water_bodies (NOT zoi)
Steps:
1. Load both. Print columns.
2. Aggregate SW by MWS: `sw_gdf.groupby('MWS_UID')[area_cols].sum()`
3. TABULAR merge (NOT spatial join): `drought_gdf.merge(sw_by_mws, left_on='uid', right_on='MWS_UID', how='inner')`
4. Year mapping: drysp_2017→area_17-18, ..., drysp_2022→area_22-23
5. Per MWS: if drysp>0 → drought year, collect area; else non-drought. sensitivity = mean_non_drought - mean_drought
6. Rank descending, export top N
Why tabular? Both share MWS IDs (uid/MWS_UID). SW has ~2798 bodies → aggregate first. Spatial join creates many-to-many duplicates.
Output: `./exports/top_sw_sensitive_microwatersheds.geojson`

───────────────────────────────────────
**QT8: Find Similar MWS (Cosine Similarity)**
Layers: drought + cropping_intensity + terrain + LULC (ALL vectors, SAME fetch)
⛔ NEVER return layer-specific IDs — ALWAYS return `uid`
Steps:
1. Load all 4 layers. Print columns.
2. Drop geometry from all except first, merge on `uid` (inner join)
3. Select numeric feature cols (exclude id, uid, geometry, area_in_ha, sum, terrainClu)
4. `StandardScaler().fit_transform(X.fillna(0))` 
5. `cosine_similarity(target_vec, X_scaled)` from sklearn
6. Rank descending (exclude target), take top N
7. Export with uid + similarity score, drop GeoServer `id`
Keep summary cols: drysp_, w_sev_, w_mod_, w_mld_, cropping_intensity, hill_slope, plain_area, ridge_area, slopy_area, valley_are
⚠️ If query doesn't mention location, add it to fetch call based on context.
Output: `./exports/similar_microwatersheds.geojson`

───────────────────────────────────────
**QT9: Find Similar MWS (Propensity Score Matching)**
Layers: same 4 as QT8
⛔ NEVER use cosine_similarity for this — use LogisticRegression PSM
Steps:
1-3. Same as QT8 (load, merge, select features)
4. Create treatment col: `treatment = (uid == target_uid).astype(int)`
5. `StandardScaler`, then `LogisticRegression(max_iter=1000, solver='lbfgs').fit(X_scaled, y)`
6. `propensity_scores = model.predict_proba(X_scaled)[:, 1]`
7. `ps_distance = abs(target_ps - other_ps)`. Rank ascending (smallest = most similar).
8. Export with uid + propensity_score + ps_distance
Output: `./exports/psm_matched_microwatersheds.geojson`

───────────────────────────────────────
**QT10: Rank Top-K MWS by Cropping Intensity & Water Availability**
Inputs: Past exports from QT5/QT6 GeoJSON files → extract UIDs
Layers: LULC vector
Steps:
1. Read `./exports/top_drought_sensitive_microwatersheds.geojson` → drought UIDs
2. Read `./exports/top_sw_sensitive_microwatersheds.geojson` → SW UIDs
3. Fetch LULC vector, filter to union of UIDs
4. CI score: `3*avg(triply_cro..c_7) + 2*avg(doubly_cro..c_7) + 1*avg(single_kha..k_7 + single_non..n_7)`
5. SW score: `3*avg(krz_water_..wate_7) + 2*avg(kr_water_a..water_7) + 1*avg(k_water_ar..water__7)`
6. Rank drought UIDs by ci_score DESC, SW UIDs by sw_score DESC
7. Export combined GeoJSON with both scores and is_drought_sensitive/is_sw_sensitive flags
Output: `./exports/ranked_mws_by_ci_and_sw.geojson`

**QT11: SC/ST% vs NREGA Works Scatter Plot**
Match layers using §4: Admin → `'admin' in n` (e.g., "Admin Boundary (dharwad_navalgund)"), NREGA → `'nrega' in n` (e.g., "NREGA Assets (dharwad_navalgund)")
Steps:
1. Iterate vector_layers. Match admin: `'admin' in layer['layer_name'].lower()`. Match NREGA: `'nrega' in layer['layer_name'].lower()`. Load both using G4 pattern.
2. Admin has vill_name, P_SC, P_ST, TOT_P. NREGA has point geometry.
3. Aggregate admin by vill_name: sum P_SC, P_ST, TOT_P → sc_st_pct = (P_SC+P_ST)/TOT_P*100
4. Dissolve admin by vill_name → one polygon per village
5. Spatial join NREGA points to admin polygons: `gpd.sjoin(nrega_gdf, admin_dissolved, how='left', predicate='within')`
6. Count works per village, merge with sc_st_pct
7. Scatter plot: X=sc_st_pct, Y=nrega_count, annotate village names, add trend line + Pearson r

8. Export PNG + GeoJSON
Outputs: `./exports/scst_vs_nrega_scatter.png`, `./exports/scst_vs_nrega_villages.geojson`

───────────────────────────────────────
**QT12: Cropping Intensity vs Rainfall Runoff — 4 Quadrant Scatter**
Layers: cropping_intensity + drought (both vectors)
Steps:
1. Load both. Compute mean CI per MWS (avg cropping_intensity_YYYY)
2. Compute runoff: find rd columns (`c.startswith('rd') and '-' in c`), group by year (c[2:4]), sum per year, average across years → mean_annual_surplus_mm
3. runoff_volume_m3 = mean_annual_surplus * area_in_ha * 10. Clip negative to 1.
4. Merge on uid. Thresholds = median of each axis.
5. Quadrants: Red=HighCI+HighRunoff, Blue=LowCI+HighRunoff, Orange=HighCI+LowRunoff, Gray=LowCI+LowRunoff
6. Scatter with log Y-scale if max/min>10, annotate uid, quadrant lines at medians
7. Export PNG + GeoJSON with quadrant assignments
Outputs: `./exports/ci_vs_runoff_quadrant_scatter.png`, `./exports/ci_vs_runoff_quadrants.geojson`

───────────────────────────────────────
**QT13: Monsoon LST vs Cropping Intensity Scatter**
Layers: cropping_intensity (CoreStack) + Landsat 8 LST (GEE)
⛔ CoreStack has NO LST — GEE is mandatory. NEVER fabricate temperatures.
Steps:
1. Load CI vector, compute mean_ci (2017-2023)
2. GEE_CENTROID_SAMPLE pattern with Landsat 8 C2L2:
   - Cloud mask: QA_PIXEL bits 3,4 must be 0
   - LST: `ST_B10 * 0.00341802 + 149.0 - 273.15` (K→°C)
   - Filter: months 6-9 (monsoon), 2017-2023, `.map(cloud_mask).map(compute_lst).mean()`
   - sampleRegions at centroids, scale=30
3. Merge LST with CI on uid, scatter with trend line + Pearson r
Outputs: `./exports/lst_vs_ci_scatter.png`, `./exports/lst_vs_cropping_intensity.geojson`

───────────────────────────────────────
**QT14: Phenological Stage Detection & Similarity**
Layers: cropping_intensity vector (for MWS uid+geometry) + Sentinel-2 NDVI (GEE)
⛔ NEVER pass polygon geometries to GEE — centroids only. NEVER fabricate NDVI.
⚠️ SKIP months 7,8 (July/Aug) — Indian monsoon = zero cloud-free Sentinel-2.
Steps:
1. Load CI vector (has uid for MWS identification, NOT admin boundary)
2. Verify target uid exists
3. GEE: For EACH month in each year (Sentinel-2 SR Harmonized):
   - Pre-filter: `ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)`
   - DO NOT use `.map()` with custom Python functions (fails in executor)
   - `collection.median().normalizedDifference(['B8','B4']).rename(band_name)`
   - sampleRegions EACH month independently (NOT multi-band — drops nulls)
4. Parse results → DataFrame uid×year×month×ndvi
5. Classify phenological stage per month:
   - Bare/Fallow: NDVI<0.15
   - Dormant: 0.15≤NDVI<0.25 and |delta|≤0.03
   - Green-up: delta>0.03
   - Peak Vegetation: NDVI≥0.45 and |delta|≤0.03
   - Maturity: 0.25≤NDVI<0.45 and |delta|≤0.03
   - Senescence: delta<-0.03
6. Compare each MWS with target per month → similarity_pct
7. Export GeoJSON + heatmap PNG (stage colors: Bare=#8B4513, Dormant=#D2B48C, Green-up=#90EE90, Maturity=#FFD700, Peak=#006400, Senescence=#FF8C00)
Outputs: `./exports/phenological_stages.geojson`, `./exports/phenological_stages_heatmap.png`

───────────────────────────────────────
**QT15: Runoff per Phenological Stage vs Cropping Intensity Scatter**
Inputs: `./exports/phenological_stages.geojson` (QT14 output)
Layers: drought + cropping_intensity (CoreStack)
Steps:
1. Load phenological GeoJSON → extract stage_YYYY_MM columns → long format (uid,year,month,stage)
2. Load drought vector, parse rd columns: rdYY-M-D → (year=2000+YY, month=M) → group by (year,month) → sum per MWS per month
3. Merge pheno stages with monthly runoff on uid+year+month
4. Accumulate runoff per (uid, year, stage): sum runoff_mm across all months of that stage
5. Get CI per MWS per year from CI vector
6. Scatter: X=runoff_accum(mm), Y=CI(%), color by stage
Outputs: `./exports/runoff_vs_ci_by_phenostage.png`, `./exports/runoff_vs_ci_by_phenostage.geojson`

───────────────────────────────────────
**QT16: Hypothesis Test — LST vs Cropping Intensity**
Layers: cropping_intensity (CoreStack) + Landsat 8 LST (GEE, ALL months 2017-2023)
⚠️ DO NOT add try/except. Keep FLAT code structure.
Steps:
1. Load CI, compute mean_ci (2017-2023)
2. GEE_CENTROID_SAMPLE: Landsat 8 C2L2, ALL months (not just monsoon), cloud-masked, .mean()
3. Merge LST+CI on uid
4. Hypothesis testing:
   a. `pearsonr(LST, CI)` → r, p
   b. Split at median LST → Hot/Cool groups
   c. `ttest_ind(hot_ci, cool_ci, equal_var=False)` → Welch's t
   d. Cohen's d = (mean_hot - mean_cool) / pooled_std
5. Scatter colored by Hot/Cool, regression line, vertical line at median LST, annotation box with r,p,t,d
Outputs: `./exports/lst_ci_hypothesis_test.png`, `./exports/lst_ci_hypothesis_test.geojson`

───────────────────────────────────────
**QT17: Agricultural Suitability Index (ASI) Ranking**
Layers: cropping_intensity + surface_water_bodies (CoreStack) + Landsat 8 LST (GEE)
⚠️ DO NOT add try/except. Keep FLAT code structure.
Steps:
1. Load CI → mean_ci. Load SW bodies (NOT zoi) → identify kharif cols (`re.match(r'^k_\\d', c)`, NOT kr/krz) → aggregate by MWS_UID → mean_kharif_sw
2. GEE_CENTROID_SAMPLE: Landsat 8 LST (all months, 2017-2023)
3. Merge CI + LST + SW on uid/MWS_UID
4. Min-max normalize all three
5. ASI = 0.40×CI_norm + 0.30×(1−LST_norm) + 0.30×SW_norm (range 0-1)
6. Rank ASI descending. Horizontal bar chart (top 25) with stacked components (CI=#2ecc71, Thermal=#e67e22, Water=#3498db)
Outputs: `./exports/agricultural_suitability_index.png`, `./exports/agricultural_suitability_index.geojson`

═══ §7 CORESTACK DATA ACCESS ═══

Available CoreStack layers:
- Raster: {', '.join(CORESTACK_DATA_PRODUCTS['raster_layers'])}
- Vector: {', '.join(CORESTACK_DATA_PRODUCTS['vector_layers'])}
- Timeseries: {', '.join(CORESTACK_DATA_PRODUCTS['timeseries_metrics'])}

Usage:
```python
import json, os, geopandas as gpd, pandas as pd
os.makedirs('./exports', exist_ok=True)
result = fetch_corestack_data("your query about India")
data = json.loads(result)
if data['success'] and data['data_type'] == 'spatial':
    vector_layers = data['spatial_data']['vector_layers']
    raster_layers = data['spatial_data']['raster_layers']
    for l in vector_layers: print(l['layer_name'])
elif data['success'] and data['data_type'] == 'timeseries':
    timeseries = data['timeseries_data']
```

Earth Engine (supplementary — ONLY if CoreStack lacks needed data):
- Initialize: `ee.Initialize(project='corestack-gee')`
- Sentinel-2: `COPERNICUS/S2_SR_HARMONIZED`
- Landsat 8: `LANDSAT/LC08/C02/T1_L2`
- Export: `geedim.MaskedImage(ee_image).download(filename=path, scale=s, region=geom, crs='EPSG:4326')`

MWS-level data: CoreStack is at microwatershed level, NOT village level. To get village names: spatial join with admin_boundaries_vector using `vill_name` column (NOT vill_ID).

═══ §8 OUTPUT & EXECUTION ═══

Expected outputs by query type: time-series PNGs, change GeoTIFFs + area stats, filtered GeoJSONs, ranking tables, similarity GeoJSONs, scatter PNGs with regression/quadrants.

```py
final_answer("The final answer is .... .\\\\n Exports:  \\\\n- export1: ./exports/export1.some_ext  \\\\n- export2: ./exports/export2.some_ext")
```

Task: {task}
"""

# ============================================================================
# HYBRID AGENT (CodeAct + CoreStack Tool)
# ============================================================================

@observe(name="run_hybrid_agent")
def run_hybrid_agent(user_query: str, exports_dir: str = None, session_id: str = None, user_id: str = None):
	"""
	Run CodeAct agent with CoreStack tool + local execution.

	Trace structure (Langfuse dashboard):
	──────────────────────────────────────
	  Session (session_id)
	  └─ Trace: run_hybrid_agent
	       ├─ input  = user_query
	       ├─ metadata = model, prompt_version, …
	       ├─ Span: corestack_workflow        (from fetch_corestack_data → _run_...)
	       │    └─ Span: corestack_api_request (each HTTP call)
	       ├─ Generation: llm_call            (auto-traced by SmolagentsInstrumentor)
	       ├─ Span: agent_reasoning_step      (auto-traced)
	       └─ output = final agent answer

	Args:
		user_query:  The user's natural-language question.
		exports_dir: Directory for exported files.
		session_id:  Langfuse session ID to group traces.  Auto-generated if None.
		user_id:     Optional end-user identifier for the Langfuse trace.
	"""
	import time as _time
	t0 = _time.perf_counter()

	_MODEL_ID = "gemini/gemini-2.5-flash-lite"

	# ── 1. Session + trace metadata ──────────────────────────────────────────
	if session_id is None:
		session_id = generate_session_id()

	set_trace_metadata(
		session_id=session_id,
		user_id=user_id,
		user_query=user_query,
		tags=["corestack", "hybrid-agent"],
		version=get_prompt_version(),
		metadata={
			"model": _MODEL_ID,
			"prompt_version": get_prompt_version(),
			"exports_dir": exports_dir or "./exports",
		},
	)
	tag_prompt_version()  # attach prompt:v1.0.0 tag
	print(f"📊 Langfuse session: {session_id}")

	# ── 2. Set up directories ────────────────────────────────────────────────
	if exports_dir is None:
		exports_dir = os.path.abspath("./exports")
	os.makedirs(exports_dir, exist_ok=True)

	workspace_dir = os.path.dirname(os.path.abspath(__file__))

	print("\n" + "="*70)
	print("🚀 HYBRID AGENT (CodeAct + CoreStack Tool)")
	print(f"📝 Query: {user_query}")
	print("="*70)

	# ── 3. Model + agent construction ────────────────────────────────────────
	model = LiteLLMModel(
		model_id=_MODEL_ID,
		api_key=os.getenv("GEMINI_API_KEY")
	)

	tools = [fetch_corestack_data]

	print("✅ Using local Python executor")
	agent = CodeAgent(
		model=model,
		tools=tools,
		additional_authorized_imports=["*"],
		executor_kwargs={"timeout_seconds": 120}
	)

	stdout_buffer = io.StringIO()
	tee_stdout = _TeeStdout(sys.stdout, stdout_buffer)

	try:
		# ── 4. Prompt generation ──────────────────────────────────────────
		prompt = create_corestack_prompt(user_query)

		# ── 5. Agent execution (auto-traced by SmolagentsInstrumentor) ───
		with redirect_stdout(tee_stdout):
			result = agent.run(prompt)

		execution_logs = stdout_buffer.getvalue()
		elapsed_ms = (_time.perf_counter() - t0) * 1000

		# ── 6. Record output + metrics on the trace ─────────────────────
		set_trace_output({
			"answer": str(result)[:2000],
			"execution_duration_ms": round(elapsed_ms, 2),
		})

		print("\n" + "="*70)
		print("✅ AGENT COMPLETED")
		print(f"⏱️  Duration: {elapsed_ms/1000:.1f}s")
		print("="*70)
		print(result)
		print("="*70)

		# Flush Langfuse events after each run
		lf_flush()
		return result

	except Exception as e:
		elapsed_ms = (_time.perf_counter() - t0) * 1000
		error_msg = f"Agent execution failed: {str(e)}"
		print(f"\n❌ ERROR: {error_msg}")
		traceback.print_exc()

		# ── Record error on the root trace ─────────────────────────────
		log_error_to_trace_root(e, context="run_hybrid_agent")
		set_trace_output({
			"error": error_msg,
			"execution_duration_ms": round(elapsed_ms, 2),
		})
		lf_flush()
		raise e


# ============================================================================
# EXAMPLE USAGE & COMPARISON
# ============================================================================

if __name__ == "__main__":
	"""
	Example usage of the hybrid agent.

	All queries in this run share one Langfuse session so they appear as a
	single conversation in the dashboard.  Each run_hybrid_agent() call
	creates its own trace within that session.
	"""
	print("Bot TEST")
	print("="*70)

	# print("Running query #1 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Could you show how cropping intensity has changed over the years in Shirur Village, Kundgol, Dharwad, Karnataka?")

	# print("Running query #2 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Could you show how surface water availability has changed over the years in Shirur Village, Kundgol, Dharwad, Karnataka?")

	# print("Running query #3 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("Can you show me areas that have lost tree cover in Shirur Village, Kundgol, Dharwad, Karnataka since 2018? also hectares of degraded area?")

	# print("Running query #4 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("How much cropland in Shirur Village, Kundgol, Dharwad, Karnataka has turned into built up since 2018?")

# 	# print("Running query #5 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("Which villages in Navalgund, Dharwad Karnataka among the ones available on Core Stack have experienced droughts? ")

# 	# print("Running query #6 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest cropping senstivity to drought")

	# print("Running query #7 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	# print("="*70)
	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest surface water availability senstivity to drought")


# 	# print("Running query #8 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("find me microwatersheds in Navalgund tehsil, Dharwad district, Karnataka most similar to 18_16157 uid microwatershed, based on its terrain, drought frequency, LULC and cropping intensity")


	print("Running query #9 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
	print("="*70)
	run_hybrid_agent("find me microwatersheds most similar to 18_16157 id microwatershed in Navalgund, Dharwad, Karnataka, based on its terrain, drought frequency, and LULC using propensity score matching")

# 	# print("Running query #10 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("From the top-K earlier identified drought-sensitive and surface-water-sensitive microwatersheds in Navalgund, Dharwad, Karnataka, rank them based on their cropping intensity and surface water availability. Use weighted score: cropping_score = 3*triple_crop + 2*double_crop + 1*(single_kharif + single_non_kharif). Water score = 3*perennial_water + 2*winter_water + 1*monsoon_water. Read past exports from ./exports/ to get the MWS UIDs.")

	# print("Running query #11 from CSV (Navalgund, Dharwad, Karnataka)...")
	# print("="*70)
	# run_hybrid_agent("In my Navalgund tehsil of Dharwad, Karnataka, compare the SC/ST% population of villages against the number of NREGA works done in the villages. Build a scatter plot.")

# 	# print("Running query #12 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("My tehsil navalgund of Dharwad, Karnataka is quite groundwater stressed. Find the top micro-watersheds that have high cropping intensity as well as a large rainfall runoff volume that can be harvested. Similarly, find those micro-watersheds that have a low cropping intensity but high runoff volume. Essentially build a neat scatterplot split into four quadrants of high/low cropping intensity and high/low runoff")


# 	# print("Running query #13 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("For my tehsil Navalgund of Dharwad, Karnataka, can you create a scatterplot of average monsoon temperatures to cropping intensity?")

# 	# print("Running query #14 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("For my microwatershed 18_16157 in Navalgund, Dharwad, Karnataka, can you find out regions with similar phenological cycles during the years 2019 to 2020, and show per month which regions are in the same phenological stage? Use Sentinel-2 NDVI and MWS boundaries to compute NDVI time series and use phenological stage detection algorithm.")

# 	# print("Running query #15 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("For the microwatersheds in Navalgund, Dharwad, Karnataka identified in the phenological stage analysis, create a scatterplot of runoff accumulation per phenological stage vs cropping intensity. Use the Drought vector rd columns for weekly runoff data, sum them per month, then accumulate per phenological stage per MWS. Plot against cropping intensity from the Cropping Intensity vector for years 2019-2020. Color by phenological stage.")

	# print("Running query #16 from CSV (Navalgund, Dharwad, Karnataka)...")
	# print("="*70)
	# run_hybrid_agent("For my Navalgund tehsil in Dharwad, Karnataka, test the hypothesis that villages with higher average temperature have higher cropping intensity. Compute per-MWS average Land Surface Temperature from Landsat 8 and cropping intensity from CoreStack, build a scatterplot, and perform hypothesis testing (Pearson correlation + t-test on hot vs cool groups).")

# 	# print("Running query #17 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("For my Navalgund tehsil in Dharwad, Karnataka, rank microwatersheds by a composite Agricultural Suitability Index considering temperature (Landsat 8 LST), cropping intensity (CoreStack CI vector), and surface water availability during the growing phenological stage (kharif season from Surface Water Bodies vector). Use weighted linear combination: ASI = 0.40*CI_norm + 0.30*(1-LST_norm) + 0.30*SW_norm. Export ranked bar chart and GeoJSON.")

