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
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import redirect_stdout

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

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 1 below almost verbatim.
   The code fetches the Cropping Intensity vector, extracts yearly columns, computes averages,
   plots a time-series line chart PNG, AND exports a GeoJSON summary per MWS.

WHY SPATIAL?
- Cropping intensity is SPATIALLY VARIABLE (different fields = different intensity)
- You want: total area under crops in village, which varies by location
- Data structure: GeoDataFrame with polygon features
- Temporal aspect: Stored as yearly columns (cropping_intensity_2017, 2018, ..., 2023)
- Analysis approach: Sum/average these columns across village polygons, plot trend

⚠️ CRITICAL LAYER NAME MATCHING:
- Cropping Intensity layer: `"Cropping Intensity (dharwad_navalgund_intensity)"` — match with `'cropping' in layer['layer_name'].lower() and 'intensity' in layer['layer_name'].lower()`
- ALWAYS print ALL vector layer names FIRST: `for l in vector_layers: print(l['layer_name'])`

OUTPUTS (MANDATORY — produce ALL of these):
- PNG: `./exports/cropping_intensity_over_years.png` — time-series line chart of average CI per year
- GeoJSON: `./exports/cropping_intensity_by_mws.geojson` — each MWS polygon with yearly CI values and mean CI

WHY NOT TIMESERIES?
- Watershed timeseries measures aggregate WATER BALANCE (runoff, precip, ET)
- It's NOT about cropping patterns or land use
- Timeseries is fortnightly, not yearly
- Timeseries has NO spatial variation (1 value per watershed)

**Query Type 2: "Surface Water Availability Over Years in [Village]"**
✅ PRIMARY: surface_water_bodies_vector (spatial vector)
⚠️  OPTIONAL CONTEXT: water_balance (timeseries) for watershed-level trends

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 1b below almost verbatim.
   The code fetches the Surface Water Bodies vector, aggregates water area by MWS per hydro-year,
   plots a time-series line chart PNG, AND exports a GeoJSON summary per MWS.

WHY SPATIAL?
- Surface water bodies are PHYSICAL FEATURES with geometry (lakes, ponds, reservoirs)
- You want: total area of water bodies within village boundaries
- Data structure: Polygon features with seasonal attributes (Kharif/Rabi/Zaid flags)
- Analysis: Group by MWS_UID, sum area_YY-YY columns per year, plot trend

⚠️ CRITICAL LAYER NAME MATCHING:
- Surface water layer: `"Surface Water Bodies (surface_waterbodies_dharwad_navalgund)"` — match with `'surface water' in layer['layer_name'].lower()` AND `'zoi' not in layer['layer_name'].lower()`
- There are TWO surface water layers — use the one WITHOUT 'zoi' in its name
- ALWAYS print ALL vector layer names FIRST

SURFACE WATER DATA COLUMN REFERENCE:
- `area_YY-YY`: Water spread area in hectares for hydro-year (e.g., `area_17-18` = 2017-18 hydro-year)
- `k_YY-YY`: Kharif season water availability (percentage)
- `kr_YY-YY`: Kharif+Rabi season water availability (percentage)
- `krz_YY-YY`: Kharif+Rabi+Zaid season water availability (percentage)
- `MWS_UID`: Microwatershed UID (use for groupby aggregation)
- Each row is ONE water body — multiple water bodies per MWS

YEAR COLUMNS: area_17-18, area_18-19, area_19-20, area_20-21, area_21-22, area_22-23

OUTPUTS (MANDATORY — produce ALL of these):
- PNG: `./exports/surface_water_trend.png` — time-series line chart of total water area per year (summed across all MWS)
- GeoJSON: `./exports/surface_water_availability_by_mws.geojson` — each MWS with aggregated water area per year

**Query Type 3: "Tree Cover Loss / Deforestation in [Village] Since [Year]"**
✅ CORRECT: The deforestation change detection raster
⚠️ The ACTUAL API layer name is: `"Change Detection Raster (change_dharwad_navalgund_Deforestation)"`
   Match with: `'deforestation' in layer['layer_name'].lower()`
   For degradation: `'degradation' in layer['layer_name'].lower()`
⛔ NEVER search for 'tree_cover_loss' or 'change_tree_cover_loss' — those names do NOT exist in the API.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 2 below almost verbatim.
   The code fetches the Deforestation change detection raster, computes loss area in hectares,
   creates a visualization PNG, AND exports a deforestation summary GeoJSON.

ACTUAL CHANGE DETECTION LAYER NAMES FROM API (confirmed):
- `"Change Detection Raster (change_dharwad_navalgund_Deforestation)"` — tree cover loss
- `"Change Detection Raster (change_dharwad_navalgund_Afforestation)"` — tree cover gain
- `"Change Detection Raster (change_dharwad_navalgund_Degradation)"` — land degradation
- `"Change Detection Raster (change_dharwad_navalgund_Urbanization)"` — urbanization (may 404)
- `"Change Detection Raster (change_dharwad_navalgund_CropIntensity)"` — crop intensity change
- `"Tree Overall Change Raster (overall_change_raster_dharwad_navalgund)"` — overall tree change

WHY CHANGE RASTER?
- Pre-computed deforestation detection (2017-2022)
- Analysis: Mask to loss classes, count pixels, convert to hectares
- SAVES COMPUTATION vs manually comparing 2017 vs 2022 LULC

⚠️ SANDBOX CONSTRAINTS (CRITICAL — violations cause instant failure):
- NEVER use bare `open()` — use `rasterio.MemoryFile(bytes)` to read rasters from downloaded bytes.
- NEVER reproject to UTM — use degree-based math for area calculation.
- `rasterio.open(path, 'w', ...)` IS allowed (it's a module method, not bare open).
- `plt.savefig(...)` IS allowed.

OUTPUTS (MANDATORY — produce ALL of these):
- PNG: `./exports/deforestation_map.png` — spatial visualization of tree cover loss areas
- GeoJSON: `./exports/deforestation_navalgund.geojson` — vectorized loss polygons with area in hectares
- If user also asks for degradation: `./exports/degradation_navalgund.geojson`

FALLBACK: If custom time period (e.g., "2018-2024"):
- Use land_use_land_cover_raster for specific years
- Compare class 6 (trees) presence across years manually

**Query Type 4: "How much cropland turned into built-up / Land Cover Change in [Location]"**
✅ CORRECT APPROACH: Compare LULC level_3 rasters across two years
⛔ NEVER use any layer with "Urbanization" or "change_urbanization" in the name. Those layers return 404.
⛔ NEVER search for "change_urbanization_raster". It does NOT work.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 2b below almost verbatim.

⚠️ SANDBOX CONSTRAINTS (CRITICAL — violations cause instant failure):
- NEVER use bare `open()` — it is BLOCKED by the sandbox executor. Use `rasterio.MemoryFile(bytes)` to read rasters from downloaded bytes.
- NEVER reproject to UTM using `calculate_default_transform` — the LULC rasters are 4911x5826 pixels in EPSG:4326 (~0.00009° resolution). Reprojecting to UTM with degree-resolution causes a 300+ PB memory allocation crash.
- For area calculation, use degree-based math: `px_area = (dx_deg * 111320 * cos(lat_rad)) * (dy_deg * 110540) / 10000` hectares.
- `rasterio.open(path, 'w', ...)` IS allowed (it's a module method, not bare open).
- `plt.savefig(...)` IS allowed.

**KEY FACTS:**
- LULC rasters are in `raster_layers` from fetch_corestack_data, with dataset_name `LULC_level_3`
- Layer names follow: `LULC_YY_YY_<district>_<tehsil>_level_3` (e.g., LULC_18_19, LULC_24_25)
- Both rasters share the same grid (same shape, same CRS EPSG:4326) — NO reprojection needed for comparison
- Rasters are float64 with NaN as nodata. Cast to int or use np.nan_to_num()
- For area: each pixel is ~0.00009 degrees. At 15.5°N: 1 px ≈ 0.0095 ha.

**LULC Class Codes (level 3):**
- 1: Built-up, 2: Water (Kharif), 3: Water (Kharif+Rabi), 4: Water (Kharif+Rabi+Zaid)
- 6: Tree/Forests, 7: Barrenlands
- 8: Single cropping cropland, 9: Single Non-Kharif cropping cropland
- 10: Double cropping cropland, 11: Triple cropping cropland, 12: Shrub_Scrub
- Cropland classes = {8, 9, 10, 11}

**METHODOLOGY:**
1. Call fetch_corestack_data ONCE → get raster_layers list
2. Print ALL raster layer names: `for l in raster_layers: print(l['layer_name'])`
3. Find LULC level_3 layers: match `'lulc' in name.lower() and 'level_3' in name.lower()`
4. Sort by year: extract YY_YY from name, pick EARLIEST matching user's "since YYYY" and MOST RECENT
   - "since 2018" → use LULC_18_19 (old) vs LULC_24_25 (new)
5. Download BOTH rasters: `requests.get(url)` → save to `./exports/lulc_old.tif`, `./exports/lulc_new.tif`
6. Open with rasterio, read band 1, replace NaN with 0: `np.nan_to_num(data, nan=0).astype(int)`
7. Create change mask: cropland in old (classes 8,9,10,11) AND built-up in new (class 1)
   `crop_to_builtup = np.isin(old_data, [8,9,10,11]) & (new_data == 1)`
8. Compute area using degree-based math (NEVER reproject to UTM):
   `px_area_ha = (dx_deg * 111320 * cos(lat)) * (dy_deg * 110540) / 10000`
9. Create a change raster (0=no change, 1=crop→built-up) and save as GeoTIFF
10. Create visualization PNG with the change map overlaid
11. Print total conversion area in hectares
12. Export: change GeoTIFF + PNG + area stats

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

═══════════════════════════════════════════════════════════════════════════════

**Query Type 9: "Find similar microwatersheds using Propensity Score Matching (PSM)"**
✅ CORRECT LAYERS: drought + cropping intensity + terrain vector + LULC vector (ALL from SAME fetch)
⛔ NEVER return layer-specific IDs like `dharwad_navalgund_drought.1` — ALWAYS return the `uid` column (e.g., `18_16157`)
⛔ NEVER use cosine similarity for this query type — use PROPENSITY SCORE MATCHING

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 7 below almost verbatim.
   Propensity Score Matching (PSM) treats the target MWS as the "treated" unit (treatment=1)
   and all others as "control" (treatment=0). A logistic regression model predicts the
   propensity score (probability of being the target). MWS with the closest propensity
   scores to the target are the most similar.
   ⚠️ DO NOT write your own loading code — use EXACTLY the layer loading pattern from EXAMPLE 7.
   The pattern is: `drought_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)`
   ⚠️ DO NOT use cosine_similarity — this query MUST use LogisticRegression propensity scores.

⚠️ CRITICAL LAYER NAME MATCHING (same as Query Type 8):
- Drought layer: `'drought' in name.lower()` AND `'causality' not in name.lower()`
- Cropping Intensity: `'cropping' in name.lower() and 'intensity' in name.lower()`
- Terrain Vector: `'terrain' in name.lower() and 'lulc' not in name.lower() and 'raster' not in name.lower()`
- LULC Vector: `'lulc' in name.lower()` AND `'terrain' not in name.lower()` AND `'level' not in name.lower()`
- ALWAYS print ALL vector layer names FIRST

LAYER SCHEMAS (same as Query Type 8):
- **Drought** (115 MWS): `uid`, `drysp_2017..2022`, `w_sev_2017..2022`, `w_mod_2017..2022`, `w_mld_2017..2022`, `avg_dryspell`
- **Cropping Intensity** (115 MWS): `uid`, `cropping_intensity_2017..2024`
- **Terrain** (115 MWS): `uid`, `hill_slope`, `plain_area`, `ridge_area`, `slopy_area`, `valley_are`, `terrainClu`
- **LULC** (115 MWS, 101 cols): `uid`, plus per-year columns for each land use class

CORRECT METHODOLOGY (Propensity Score Matching):
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names
3. Find and load: drought, cropping intensity, terrain, LULC — all as GeoDataFrames
4. Print columns of each to verify
5. Merge all on `uid` column (inner join, drop geometry except from first)
6. Select numeric feature columns (exclude `id`, `uid`, `geometry`, `area_in_ha`, `sum`)
7. Create binary treatment column: treatment=1 for target MWS, treatment=0 for all others
8. Fill NaN with 0, normalize with StandardScaler
9. Fit logistic regression: treatment ~ features → get propensity scores for ALL MWS
10. Compute absolute difference in propensity score between target and each other MWS
11. Rank by ASCENDING propensity score difference (smallest diff = most similar)
12. Take top N matches
13. Return results with `uid` as the MWS identifier + propensity_score + ps_distance
14. Export top matched MWS geometries as GeoJSON

⚠️ IMPORTANT NOTES:
- Use `sklearn.linear_model.LogisticRegression` with `max_iter=1000` and `solver='lbfgs'`
- The propensity score is `model.predict_proba(X)[:, 1]` — the probability of being "treated"
- ps_distance = abs(target_ps - other_ps). Smallest distance = best match.
- ⚠️ LOCATION: Same as Query Type 8. ALWAYS include location in fetch_corestack_data call.
- Drop GeoServer `id` column from export. Use `uid` as identifier.

═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════

**Query Type 10: "Rank previously identified top-K MWS by cropping intensity AND surface water availability"**
This query takes MWS UIDs from PREVIOUSLY exported GeoJSON files (from earlier prompts 5/6/7) and
ranks them by two weighted composite scores:

✅ TWO PAST EXPORTS ARE USED:
- `./exports/top_drought_sensitive_microwatersheds.geojson` → UIDs for **cropping intensity ranking**
- `./exports/top_sw_sensitive_microwatersheds.geojson` → UIDs for **surface water ranking**

✅ TWO DATA LAYERS NEEDED (fetched from CoreStack):
- **LULC vector** → has per-MWS columns: `triply_cro/triply_c_1..7`, `doubly_cro/doubly_c_1..7`,
  `single_kha/single_k_1..7`, `single_non/single_n_1..7` (cropping areas by year)
  AND `krz_water_/krz_wate_1..7` (perennial), `kr_water_a/kr_water_1..7` (winter), `k_water_ar/k_water__1..7` (monsoon)
- **Surface Water Bodies vector** → has per-waterbody columns: `krz_YY-YY` (perennial), `kr_YY-YY` (winter), `k_YY-YY` (monsoon)
  Linked to MWS via `MWS_UID` column. Aggregate by `MWS_UID` using sum.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 8 below almost verbatim.

**WEIGHTED SCORE FORMULAS:**

1. **Cropping Intensity Score** (per MWS, averaged across years):
   `ci_score = 3 * avg(triply_cropped_area) + 2 * avg(doubly_cropped_area) + 1 * avg(single_kharif + single_non_kharif)`
   - Use LULC vector columns: triply_cro..triply_c_7 (8 years), doubly_cro..doubly_c_7, single_kha..single_k_7, single_non..single_n_7
   - Each suffix _1 through _7 represents years 2018-2024 (base col = 2017)
   - Average across all years, then apply weights

2. **Surface Water Availability Score** (per MWS, averaged across years):
   `sw_score = 3 * avg(perennial_water_area) + 2 * avg(winter_water_area) + 1 * avg(monsoon_water_area)`
   - Use LULC vector columns: krz_water_/krz_wate_1..7 (perennial), kr_water_a/kr_water_1..7 (winter), k_water_ar/k_water__1..7 (monsoon)
   - OR from Surface Water Bodies layer: aggregate `krz_YY-YY`, `kr_YY-YY`, `k_YY-YY` by MWS_UID
   - Average across all years, then apply weights

⚠️ CRITICAL LULC COLUMN NAME PATTERN:
LULC vector columns are TRUNCATED. The naming pattern for 8 yearly values (2017-2024) is:
- Base col (2017): `triply_cro`, `doubly_cro`, `single_kha`, `single_non`, `krz_water_`, `kr_water_a`, `k_water_ar`
- Subsequent (2018-2024): `triply_c_1` through `triply_c_7`, `doubly_c_1`..`doubly_c_7`, etc.

CORRECT METHODOLOGY:
1. Read `./exports/top_drought_sensitive_microwatersheds.geojson` → extract UIDs for crop ranking
2. Read `./exports/top_sw_sensitive_microwatersheds.geojson` → extract UIDs for water ranking
3. Call fetch_corestack_data ONCE to get LULC vector layer
4. Filter LULC to only the UIDs from both exports (union of UIDs)
5. Compute cropping intensity score per MWS using LULC triple/double/single crop columns
6. Compute surface water score per MWS using LULC krz/kr/k water columns
7. Rank drought-sensitive MWS by cropping intensity score (descending)
8. Rank sw-sensitive MWS by surface water score (descending)
9. Print BOTH rankings with uid + score
10. Export combined results as GeoJSON with both scores

⚠️ LOCATION: ALWAYS include location in fetch_corestack_data call.

═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════

**Query Type 11: "Compare SC/ST% population of villages against number of NREGA works — scatter plot"**
This query compares the scheduled caste/tribe population percentage of each village
with the number of NREGA (MGNREGA) works done in that village, visualized as a scatter plot.

✅ TWO DATA LAYERS NEEDED (fetched from CoreStack in a SINGLE call):
- **Admin Boundary vector** → has per-village census columns:
  `P_SC` (total SC population), `P_ST` (total ST population), `TOT_P` (total population),
  `vill_name` (village name). 138 polygons, one per village boundary.
- **NREGA Assets vector** → has per-work point features (10,797 works):
  `Gram_panch` (Gram Panchayat name), `Panchayat` (village), `Work Name`, `Work Type`,
  `WorkCatego` (work category), `Total_Expe` (expenditure), `geometry` (point location).

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 9 below almost verbatim.

**METHODOLOGY:**
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names to find Admin Boundary and NREGA Assets layers
3. Find Admin Boundary layer: match `'admin' in layer_name.lower()` or `'panchayat' in layer_name.lower()`
4. Find NREGA Assets layer: match `'nrega' in layer_name.lower()`
5. Load both as GeoDataFrames
6. Aggregate Admin Boundary by `vill_name`: sum `P_SC`, `P_ST`, `TOT_P` per village
   Compute `sc_st_pct = (P_SC + P_ST) / TOT_P * 100`. Drop villages with TOT_P == 0.
7. Dissolve admin polygons by `vill_name` to get one polygon per village
8. Spatial join: assign each NREGA work (point) to its containing admin village polygon
   Use `gpd.sjoin(nrega_gdf, admin_dissolved[['vill_name', 'geometry']], how='left', predicate='within')`
9. Count NREGA works per village: `joined.groupby('vill_name').size()`
10. Merge SC/ST% with NREGA count on `vill_name`
11. Build scatter plot:
    - X-axis: SC/ST population % (per village)
    - Y-axis: Number of NREGA works
    - Annotate each point with village name
    - Add trend line (linear regression fit)
    - Add correlation coefficient (Pearson r) in the title or annotation
    - Export as PNG to `./exports/scst_vs_nrega_scatter.png`
12. Also export merged data as GeoJSON with village polygons + sc_st_pct + nrega_count

⚠️ IMPORTANT NOTES:
- Admin Boundary and NREGA Assets both appear as `dharwad_navalgund` in layer_name but with different dataset_name.
  Use the dataset_name or the layer URL to distinguish them.
- NREGA works are POINT geometries; Admin Boundary are POLYGON geometries.
- Some admin villages may have zero NREGA works (they won't appear in the spatial join).
- Use `matplotlib.pyplot` for the scatter plot. Label outlier points with village name.
- ⚠️ LOCATION: ALWAYS include location in fetch_corestack_data call.

═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════

**Query Type 12: "Cropping Intensity vs Rainfall Runoff Volume — 4 quadrant scatter plot"**
This query computes two metrics per micro-watershed (MWS) and plots them against each other
in a scatter plot with 4 quadrants (High/Low CI × High/Low Runoff).

✅ TWO DATA LAYERS NEEDED (fetched from CoreStack in a SINGLE call):
- **Cropping Intensity vector** (115 MWS, 54 cols): has `cropping_intensity_YYYY` columns (2017-2024).
  Mean over years = mean_ci per MWS.
- **Drought vector** (115 MWS, 258 cols): has weekly `rd` columns (format: `rdYY-M-D`).
  These are weekly monsoon-season water-balance values per MWS (rainfall departure from normal, in mm).
  Positive = above-normal rainfall (surplus/runoff potential). Negative = deficit.
  Sum all `rd` columns per year → annual monsoon water surplus per MWS.
  Multiply by `area_in_ha` × 10 → runoff volume in m³.
  Average across years → mean annual runoff volume per MWS.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 10 below almost verbatim.

**DATA COLUMNS:**
- Cropping Intensity: `cropping_intensity_2017` through `cropping_intensity_2024` (8 years)
- Drought `rd` columns: `rd17-5-21`, `rd17-5-28`, ..., `rd22-10-8` (145 weekly columns over 2017-2022)
  Pattern: `rdYY-M-D` where YY = 2-digit year, M = month, D = day
  Group by year prefix (first 4 chars: `rd17`, `rd18`, ..., `rd22`)

**METHODOLOGY:**
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names
3. Find and load: Cropping Intensity vector + Drought vector as GeoDataFrames
4. Print columns of each to verify
5. Compute mean cropping intensity per MWS: average `cropping_intensity_YYYY` across years
6. Compute runoff volume per MWS:
   a. Identify all `rd` columns (startswith 'rd' and contains '-')
   b. Group by year: `rdYY-*` → group by `c[2:4]` (year digits)
   c. For each year: `annual_sum = gdf[year_cols].sum(axis=1)` (total monsoon surplus in mm)
   d. Average across years: `mean_annual_surplus` (mm)
   e. Volume: `runoff_volume_m3 = mean_annual_surplus * area_in_ha * 10` (1 mm over 1 ha = 10 m³)
7. Merge CI and Drought on `uid` column
8. Drop MWS with negative mean runoff (net deficit = no harvestable runoff)
9. Set thresholds: median of each axis
10. Build scatter plot:
    - X-axis: Mean Cropping Intensity (%)
    - Y-axis: Mean Annual Runoff Volume (m³) — use LOG SCALE if max/min > 10
    - Color by quadrant:
      • Red = High CI + High Runoff (prioritize conservation)
      • Blue = Low CI + High Runoff (untapped irrigation potential)
      • Orange = High CI + Low Runoff (vulnerable, water-stressed crops)
      • Gray = Low CI + Low Runoff
    - Draw quadrant lines at median thresholds
    - Annotate each point with MWS `uid`
    - Show legend with quadrant labels + count of MWS in each
    - Title with location and threshold values
11. Export scatter plot PNG + GeoJSON with quadrant assignments

⚠️ IMPORTANT NOTES:
- `rd` columns are in the **Drought** layer, NOT the Cropping Intensity layer.
- Always check columns with `[c for c in gdf.columns if c.startswith('rd')]`
- Year grouping: `c[2:4]` extracts '17', '18', etc. Convert to int: `2000 + int(c[2:4])`
- Some MWS may have negative mean annual surplus (net deficit years). These have zero harvestable runoff.
  Either drop them or set their volume to 0 before plotting.
- ⚠️ LOCATION: ALWAYS include location in fetch_corestack_data call.

═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════

**Query Type 13: "Create a scatterplot of average monsoon temperature vs cropping intensity"**
This query computes two metrics per micro-watershed (MWS) and plots them as a scatter:
- **X-axis**: Average monsoon Land Surface Temperature (°C) — from Landsat 8 via Google Earth Engine
- **Y-axis**: Average Cropping Intensity (%) — from CoreStack cropping_intensity_vector

✅ DATA SOURCES:
- **Cropping Intensity**: CoreStack `cropping_intensity_vector` (columns: `cropping_intensity_2017` through `cropping_intensity_2024`)
- **Land Surface Temperature (LST)**: Google Earth Engine — Landsat 8 Collection 2 Level 2 (`LANDSAT/LC08/C02/T1_L2`)
  - Band: `ST_B10` (Surface Temperature)
  - Scale factor: `ST_B10 * 0.00341802 + 149.0` → Kelvin, then `- 273.15` → Celsius
  - Cloud masking: `QA_PIXEL` band — bits 3 (cloud) and 4 (cloud shadow) must be 0
  - Monsoon season: June–September (months 6–9)
  - Years: 2017–2023 (Landsat 8 availability overlap with CoreStack CI data)

⛔ CoreStack does NOT have LST data — GEE is MANDATORY for temperature.
⛔ NEVER fabricate temperature values. ALWAYS fetch real Landsat LST from GEE.
⛔ NEVER pass MWS IDs or uid columns to GEE. GEE only understands lat/lon coordinates.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 11 below almost verbatim.

⚠️ CRITICAL APPROACH — CENTROID-BASED LST EXTRACTION (NOT polygon-based):
The agent MUST extract lat/lon centroids from MWS polygons and sample LST at those points.
DO NOT try to upload MWS polygons to GEE as ee.FeatureCollection — that causes hallucination.
DO NOT try to use reduceRegions with MWS polygons — too complex and error-prone.
Instead:
1. Compute centroid of each MWS polygon → get (lat, lon) per MWS
2. Build a SINGLE mean monsoon LST image on GEE (composite across ALL years 2017-2023)
3. Sample that image at each centroid point using ee.Image.sample() or ee.Image.reduceRegion()
4. Bring values client-side and merge with cropping intensity

⚠️ CRITICAL LAYER NAME MATCHING:
- Cropping Intensity: `'cropping' in name.lower() and 'intensity' in name.lower()`
- ALWAYS print ALL vector layer names FIRST

**METHODOLOGY:**
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names
3. Find and load: Cropping Intensity vector as GeoDataFrame (EPSG:4326)
4. Compute mean cropping intensity per MWS: average `cropping_intensity_YYYY` for 2017-2023
5. Compute centroid lat/lon for each MWS polygon:
   `ci_gdf['centroid_lon'] = ci_gdf.geometry.centroid.x`
   `ci_gdf['centroid_lat'] = ci_gdf.geometry.centroid.y`
6. Build ROI bounding box from the GeoDataFrame total_bounds:
   `minx, miny, maxx, maxy = ci_gdf.total_bounds`
   `ee_roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])`
7. Initialize Earth Engine: `ee.Initialize(project='corestack-gee')`
8. Build a SINGLE multi-year monsoon mean LST image:
   a. For each year 2017-2023: filter Landsat 8 C2L2 to June 1 – Sept 30, apply cloud mask
   b. Compute LST in Celsius per image
   c. Take .mean() composite per year, then merge all yearly composites → overall mean
   OR simply: filter ALL years June-Sept at once → single .mean() composite
9. For EACH MWS centroid (iterate rows of ci_gdf):
   a. Create ee.Geometry.Point([lon, lat])
   b. Call `monsoon_lst_image.reduceRegion(ee.Reducer.mean(), point_geom, scale=30).getInfo()`
   c. Extract LST value
   ⚠️ To avoid N×1 getInfo() calls (slow), batch centroids into an ee.FeatureCollection of POINTS:
   ```
   points = []
   for idx, row in ci_gdf.iterrows():
       pt = ee.Geometry.Point([row['centroid_lon'], row['centroid_lat']])
       points.append(ee.Feature(pt, {{'uid': str(row['uid'])}}))
   fc_points = ee.FeatureCollection(points)
   sampled = monsoon_lst_image.sampleRegions(collection=fc_points, scale=30, geometries=False)
   results = sampled.getInfo()['features']
   ```
   This does a SINGLE getInfo() call for all centroids.
10. Parse results → DataFrame with uid + mean_monsoon_lst
11. Merge with ci_gdf on uid
12. Build scatter plot:
    - X-axis: Mean Monsoon LST (°C)
    - Y-axis: Mean Cropping Intensity (%)
    - Add trend line (linear regression) + Pearson r in title
    - Annotate outlier points with MWS `uid`
13. Export: scatter plot PNG to `./exports/lst_vs_ci_scatter.png`
14. Export: merged data as GeoJSON to `./exports/lst_vs_cropping_intensity.geojson`

⚠️ IMPORTANT NOTES:
- Landsat 8 data starts from 2013, but CoreStack CI starts from 2017. Use 2017-2023 overlap.
- Some centroids may fall on masked pixels → NaN LST. Drop those MWS from the plot.
- NEVER pass polygon geometries to GEE — use POINT centroids only.
- NEVER reference MWS IDs in GEE calls — GEE only works with coordinates.
- The `sampleRegions` approach is efficient: 1 server call for all 115 MWS points.
- ⚠️ LOCATION: ALWAYS include location in fetch_corestack_data call.

**Query Type 14: "Find regions with similar phenological cycles to a target MWS — per-month phenological stage detection"**
This query uses Sentinel-2 NDVI time series from GEE to detect phenological stages for each microwatershed (MWS),
then identifies which MWS share the same phenological stage as a user-specified target MWS for each month.

✅ DATA SOURCES:
- **MWS Boundaries**: CoreStack vector layer with `uid` field and geometry (Admin Boundary or Cropping Intensity vector)
- **NDVI**: Google Earth Engine — Sentinel-2 Surface Reflectance Harmonized (`COPERNICUS/S2_SR_HARMONIZED`)
  - NDVI = (B8 - B4) / (B8 + B4)
  - Cloud masking: `SCL` band — exclude values 3 (cloud shadow), 8 (cloud medium prob), 9 (cloud high prob), 10 (thin cirrus)
  - Scale: 10m native resolution
  - Monthly median composites for each month in the requested year range

⛔ CoreStack has an NDVI Timeseries layer but it may not have the per-month granularity needed for phenological analysis.
⛔ Use Sentinel-2 on GEE for monthly NDVI composites — MANDATORY.
⛔ NEVER pass MWS polygon geometries to GEE — use CENTROID POINTS only.
⛔ NEVER fabricate NDVI values. ALWAYS fetch real Sentinel-2 NDVI from GEE.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 12 below almost verbatim.

⚠️ CRITICAL APPROACH — CENTROID-BASED MONTHLY NDVI + PHENOLOGICAL CLASSIFICATION:
1. Get MWS boundaries from CoreStack (vector layer with uid + geometry)
2. Verify the target MWS uid exists in the data
3. Extract centroids from all MWS polygons → (lat, lon) per MWS
4. Build MONTHLY median NDVI composites on GEE for each month in the requested years
5. For EACH monthly composite, sample at all centroids using sampleRegions independently
   ⚠️ DO NOT stack all months into one multi-band image — sampleRegions drops features where ANY band is null!
   Instead, loop through 24 monthly images and call sampleRegions on each single-band image separately.
   This ensures a cloudy month doesn't prevent data from other months.
6. Parse each month's results into records: uid × month × NDVI
7. Combine all records into a single DataFrame
8. Compute temporal derivatives (delta NDVI month-to-month) and classify phenological stages
9. Compare each MWS with target MWS per month to find phenological matches
10. Export vector GeoJSON with monthly phenological stage + match info per MWS

⚠️ PHENOLOGICAL STAGE CLASSIFICATION ALGORITHM:
Based on NDVI magnitude and month-to-month change (delta = NDVI_current - NDVI_previous):
- **Bare/Fallow**: NDVI < 0.15
- **Dormant**: 0.15 ≤ NDVI < 0.25 and |delta| ≤ 0.03
- **Green-up**: delta > 0.03 (vegetation actively increasing)
- **Peak Vegetation**: NDVI ≥ 0.45 and |delta| ≤ 0.03
- **Maturity**: 0.25 ≤ NDVI < 0.45 and |delta| ≤ 0.03 (stable mid-range)
- **Senescence**: delta < -0.03 (vegetation declining)

⚠️ CRITICAL LAYER NAME MATCHING:
- Cropping Intensity (PRIMARY — has `uid` for MWS): `'cropping' in name.lower() and 'intensity' in name.lower()`
- ⚠️ Do NOT use Admin Boundary for MWS data — it has village-level `vill_ID`, not MWS-level `uid`
- ALWAYS print ALL vector layer names FIRST

**METHODOLOGY:**
1. Call fetch_corestack_data ONCE → get vector_layers list
2. Print ALL layer names
3. Find and load: Cropping Intensity vector as GeoDataFrame (EPSG:4326) — it has `uid` column for MWS identification
   ⚠️ Do NOT use Admin Boundary — it has `vill_ID` (village-level) not `uid` (MWS-level)
4. Verify target MWS uid (e.g., '18_16157') exists in the data
5. Compute centroid lat/lon for each MWS polygon:
   `mws_gdf['centroid_lon'] = mws_gdf.geometry.centroid.x`
   `mws_gdf['centroid_lat'] = mws_gdf.geometry.centroid.y`
6. Build ROI bounding box from total_bounds:
   `minx, miny, maxx, maxy = mws_gdf.total_bounds`
   `ee_roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])`
7. Initialize Earth Engine: `ee.Initialize(project='corestack-gee')`
8. For each month in each requested year (e.g., 2019-2020):
   ⚠️ SKIP months July (7) and August (8) — Indian monsoon has ZERO cloud-free Sentinel-2 imagery.
   This means for 2019-2020 you process 20 months, NOT 24.
   a. Filter Sentinel-2 SR Harmonized to that month+year within ROI
   b. Pre-filter by cloud percentage: `.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))`
   c. Take median composite: `.median()`
   d. Compute NDVI directly on composite: `.normalizedDifference(['B8', 'B4'])`
   ⚠️ DO NOT use .map() with custom Python functions — it fails with "mapped function's arguments cannot be used in client-side operations"
   ⚠️ Instead of SCL cloud masking via .map(), use CLOUDY_PIXEL_PERCENTAGE metadata filter
   e. Rename band to `NDVI_YYYY_MM`
9. Build ee.FeatureCollection of CENTROID POINTS (with uid property)
10. For EACH monthly image, call sampleRegions independently (20 separate calls for 2 years)
    ⚠️ DO NOT stack bands — multi-band sampleRegions drops features where ANY band is null
11. Each sampleRegions call returns ~115 features with one NDVI value → accumulate into records list
12. Parse results → DataFrame with uid + NDVI per month
13. Compute temporal derivative (delta NDVI) per MWS between consecutive months
14. Classify phenological stage per MWS per month using thresholds above
15. Get target MWS stages and compare with all other MWS per month
16. Compute similarity score: % of months where phenological stage matches target
17. Build output: wide-format DataFrame with stage_YYYY_MM, matches_YYYY_MM columns per MWS
18. Export: GeoJSON with all attributes to `./exports/phenological_stages_<location>.geojson`
19. Export: heatmap PNG showing phenological stages per MWS per month to `./exports/phenological_stages_heatmap.png`

⚠️ IMPORTANT NOTES:
- Sentinel-2 available from mid-2015, so 2019-2020 has good coverage.
- ⚠️ ALWAYS SKIP months 7 (July) and 8 (August) — Indian monsoon produces ZERO cloud-free scenes. Do NOT waste GEE calls on them.
- If a month has zero cloud-free scenes for a centroid, that band returns null → handle as NaN.
- Use scale=10 for Sentinel-2 (10m native resolution for B4 and B8 bands).
- Wrap each sampleRegions().getInfo() in try/except to handle GEE errors gracefully.
- sampleRegions with multi-band image returns ALL band values per point → very efficient.
- NEVER pass polygon geometries to GEE — use POINT centroids only.
- ⚠️ LOCATION: ALWAYS include location in fetch_corestack_data call.

═══════════════════════════════════════════════════════════════════════════════

**Query Type 15: "Scatterplot of runoff accumulation per phenological stage vs cropping intensity"**
This query builds on Query Type 14's phenological output. For each MWS and each detected phenological stage,
it accumulates the runoff (from Drought vector `rd` columns) over the months belonging to that stage,
and plots it against the average cropping intensity of that MWS for the corresponding year(s).

✅ DATA SOURCES (ALL from CoreStack — no GEE needed):
- **Phenological stages GeoJSON**: `./exports/phenological_stages.geojson` (output of Query 14)
  Contains `stage_YYYY_MM` columns per MWS, indicating the phenological stage per month.
- **Drought vector** (115 MWS, ~258 cols): has weekly `rd` columns (format: `rdYY-M-D`)
  These are weekly monsoon-season water-balance departure values per MWS (mm).
  Positive = surplus (runoff potential), negative = deficit.
- **Cropping Intensity vector** (115 MWS, 54 cols): has `cropping_intensity_YYYY` columns (2017-2024).

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 13 below almost verbatim.

**DATA COLUMNS:**
- Phenological GeoJSON: `stage_2019_01`, `stage_2019_02`, ..., `stage_2020_12` (monthly phenological stages)
- Drought `rd` columns: `rdYY-M-D` (e.g., `rd19-5-21`). Pattern: `rdYY-M-D` where YY = 2-digit year.
  Dates parsed as: `2000 + int(YY)`, month = int(M), day = int(D).
  Map each `rd` column to its nearest month → group by year+month → sum per month per MWS.
- Cropping Intensity: `cropping_intensity_2019`, `cropping_intensity_2020` for the relevant years.

**METHODOLOGY:**
1. Read `./exports/phenological_stages.geojson` → GeoDataFrame with `uid`, `stage_YYYY_MM` columns.
2. Call fetch_corestack_data ONCE → get Drought vector + Cropping Intensity vector.
3. Print ALL layer names, load both layers as GeoDataFrames.
4. Parse ALL `rd` columns from Drought vector:
   a. For each `rdYY-M-D` column, extract year=2000+int(YY), month=int(M).
   b. Group columns by (year, month).
   c. For each (year, month) group, compute the SUM of `rd` values → monthly runoff per MWS (mm).
5. Build a long-format DataFrame: uid × year × month × monthly_runoff_mm.
6. Merge with the phenological stages from step 1 (join on uid + year + month).
7. For each unique (uid, year, phenological_stage) combination:
   a. Sum the monthly_runoff_mm across all months of that stage → runoff_accumulation_mm per stage.
8. Get cropping intensity per MWS per year from CI vector (`cropping_intensity_YYYY`).
9. Merge runoff_accumulation with CI per MWS per year.
10. Build scatterplot:
    - X-axis: Runoff Accumulation (mm) during phenological stage
    - Y-axis: Cropping Intensity (%) for that year
    - Color by phenological stage (Bare/Fallow, Dormant, Green-up, Maturity, Peak Vegetation, Senescence)
    - Annotate with legend showing stage colors + count per stage
    - Title: "Runoff Accumulation per Phenological Stage vs Cropping Intensity"
11. Export scatter plot PNG to `./exports/runoff_vs_ci_by_phenostage.png`
12. Export merged data as GeoJSON to `./exports/runoff_vs_ci_by_phenostage.geojson`

⚠️ IMPORTANT NOTES:
- The `rd` columns cover monsoon/post-monsoon weeks (roughly May-Oct for years 2017-2022).
  NOT all 12 months have corresponding `rd` data — only months with `rd` columns contribute to runoff.
  For months without `rd` data, runoff contribution is 0.
- `rd` column date format: `rdYY-M-D` (e.g., `rd19-6-17` = 2019-06-17). Extract month to map to phenological month.
- Phenological stage GeoJSON has `stage_YYYY_MM` columns. Parse YYYY and MM from column name to match.
- Each scatter point represents ONE (MWS, year, phenological_stage) combination.
- Some stages may have zero runoff (e.g., non-monsoon stages with no `rd` columns).
- ⚠️ LOCATION: ALWAYS include location in fetch_corestack_data call.

═══════════════════════════════════════════════════════════════════════════════

**Query Type 16: "Hypothesis test — do villages with higher average temperature have higher cropping intensity?"**
This query computes per-MWS mean Land Surface Temperature (Landsat 8, all months 2017-2023) and mean
Cropping Intensity (CI, 2017-2023), then tests the hypothesis that higher LST → higher CI.

✅ DATA SOURCES:
- **Cropping Intensity vector** from CoreStack (has `uid`, `cropping_intensity_YYYY` columns).
- **Landsat 8 Collection 2 Level 2** from GEE (ST_B10 thermal band → LST in °C).
  Uses ALL months (not just monsoon) for a true annual average temperature.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 14 below almost verbatim.
⚠️ DO NOT add try/except blocks. DO NOT wrap code in additional if/else nesting.
   Keep the FLAT code structure exactly as shown. All variables are defined at the top-level scope.

**METHODOLOGY:**
1. fetch_corestack_data → load Cropping Intensity vector layer.
2. Compute `mean_ci` per MWS (average of `cropping_intensity_2017` … `cropping_intensity_2023`).
3. Extract centroid lat/lon per MWS polygon.
4. Initialize GEE, build ROI bounding box, build ee.FeatureCollection of centroid points.
5. Build Landsat 8 annual mean LST composite (2017-2023, all months, cloud-masked).
6. Sample LST at each centroid → merge with CI on `uid`.
7. **Hypothesis testing:**
   a. Pearson correlation (r, p-value) between LST and CI.
   b. Split MWS into two groups at median LST → "Hot" vs "Cool".
   c. Independent two-sample t-test: mean CI of Hot vs Cool group.
   d. Report: r, p-value, t-statistic, means of both groups, effect size (Cohen's d).
8. Scatter plot with regression line, annotated with Pearson r, p-value.
   Color points by Hot/Cool group. Add a vertical line at median LST.
9. Export scatter PNG + GeoJSON with LST, CI, and group columns.

- Output: scatterplot PNG, hypothesis test summary printed, GeoJSON

═══════════════════════════════════════════════════════════════════════════════

**Query Type 17: "Rank microwatersheds by a composite Agricultural Suitability Index (ASI) considering temperature, cropping intensity, and water availability during the growing phenological stage"**
This query builds a multi-criteria suitability index per microwatershed (MWS), inspired by the FAO
Global Agro-Ecological Zones (GAEZ) framework and standard Multi-Criteria Decision Analysis (MCDA)
for agricultural land evaluation.

✅ DATA SOURCES:
- **Cropping Intensity vector** from CoreStack — `cropping_intensity_YYYY` columns (2017-2023).
- **Landsat 8 Collection 2 Level 2** from GEE — ST_B10 thermal band → mean LST in °C (2017-2023, all months).
- **Surface Water Bodies vector** from CoreStack — seasonal water area columns per hydro-year:
  `k_YY-YY` (kharif/monsoon = growing-season water), `kr_YY-YY` (kharif+rabi), `krz_YY-YY` (perennial).
  Linked to MWS via `MWS_UID`. Multiple water bodies per MWS → aggregate by `MWS_UID` using sum.

🔴 MANDATORY: For this query type, COPY the code from EXAMPLE 15 below almost verbatim.
⚠️ DO NOT add try/except blocks. DO NOT wrap code in additional if/else nesting.
   Keep the FLAT code structure exactly as shown.

**SCIENTIFIC RATIONALE (FAO GAEZ / MCDA):**
- **Thermal Suitability**: In semi-arid India (Navalgund), mean LSTs range ~37-45 °C, well above the
  optimal 25-35 °C for dominant crops (sorghum, cotton, groundnut). Higher LST = more heat stress =
  LOWER suitability. We use *inverse* min-max normalisation so cooler MWS score higher.
- **Cropping Intensity**: Direct productivity indicator. CI ≈ 1.0 = single-crop, CI ≈ 2.0 = double-crop.
  Higher CI = greater agricultural output per unit area = HIGHER suitability.
- **Growing-Season Water Availability**: The `k_YY-YY` (kharif/monsoon-season) water-area columns
  capture water spread during June-October — the main crop-growing phenological window (Green-up →
  Peak Vegetation). More surface water during this stage supports irrigation and soil moisture,
  directly boosting suitability.

**COMPOSITE INDEX FORMULA (Weighted Linear Combination):**
  ASI = w_ci × CI_norm + w_temp × (1 − LST_norm) + w_water × SW_norm

  Where:
    CI_norm  = min-max normalised mean Cropping Intensity [0-1]
    LST_norm = min-max normalised mean Land Surface Temperature [0-1]
    SW_norm  = min-max normalised mean kharif surface-water area [0-1]
    w_ci     = 0.40  (strongest direct productivity indicator)
    w_temp   = 0.30  (heat-stress penalty)
    w_water  = 0.30  (growing-season resource availability)

  Range: 0 (least suitable) → 1 (most suitable).

**METHODOLOGY:**
1. Call fetch_corestack_data ONCE → get vector_layers list.
2. Print ALL layer names.
3. Load Cropping Intensity vector → compute `mean_ci` per MWS (avg of cropping_intensity_2017…2023).
4. Load Surface Water Bodies vector (match: `'surface water' in name.lower()` AND `'zoi' not in name.lower()`).
   Print columns. Identify kharif water-area columns: those matching pattern `k_YY-YY` BUT NOT starting
   with `kr` (i.e., single `k_` prefix = kharif only). Aggregate by `MWS_UID` using sum.
   Compute `mean_kharif_sw` per MWS (average of kharif columns across years).
5. Extract centroid lat/lon per MWS polygon.
6. Initialise GEE, build Landsat 8 mean LST composite (2017-2023, all months, cloud-masked).
7. Sample LST at centroids → merge with CI and SW on `uid` / `MWS_UID`.
8. Min-max normalise all three indicators.
9. Compute ASI = 0.40 * CI_norm + 0.30 * (1 - LST_norm) + 0.30 * SW_norm.
10. Rank MWS by ASI descending.
11. Visualisation: horizontal bar chart (top 25 MWS) with stacked component contributions +
    a colour-coded choropleth-style legend.
12. Export: bar-chart PNG + GeoJSON with all component scores, ASI, and rank.

- Output: ranked bar chart PNG, GeoJSON with ASI scores per MWS

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

⚠️ IMPORTANT: The internal product names (change_tree_cover_loss_raster etc.) do NOT match the actual API layer names!
The ACTUAL layer names from the API are formatted as: `"Change Detection Raster (change_<district>_<tehsil>_<Type>)"`
Examples for Navalgund, Dharwad:
   - Deforestation: `"Change Detection Raster (change_dharwad_navalgund_Deforestation)"` — match: `'deforestation' in name.lower()`
   - Afforestation: `"Change Detection Raster (change_dharwad_navalgund_Afforestation)"` — match: `'afforestation' in name.lower()`
   - Degradation:   `"Change Detection Raster (change_dharwad_navalgund_Degradation)"` — match: `'degradation' in name.lower()`
   - Overall tree:  `"Tree Overall Change Raster (overall_change_raster_dharwad_navalgund)"` — match: `'tree overall' in name.lower()`

a) **Deforestation raster** (tree cover loss) 2017-2022:
   - ACTUAL API name: `"Change Detection Raster (change_dharwad_navalgund_Deforestation)"`
   - Match with: `'deforestation' in layer['layer_name'].lower()`
   - Use for: "tree cover loss since 2018", "deforestation", "lost tree cover"
   - MASK to class 1 to get loss areas

b) **Afforestation raster** (tree cover gain) 2017-2022:
   - ACTUAL API name: `"Change Detection Raster (change_dharwad_navalgund_Afforestation)"`
   - Match with: `'afforestation' in layer['layer_name'].lower()`
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

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 1: CROPPING INTENSITY OVER YEARS (Vector)            ║
		# ║ ⚠️ For Query Type 1, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ Produces: time-series PNG + per-MWS GeoJSON                  ║
		# ╚══════════════════════════════════════════════════════════════╝
		# FIRST: Print all vector layer names to find the right one
		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 1: Find cropping intensity layer (case-insensitive substring match)
		crop_gdf = None
		for layer in vector_layers:
			if 'cropping' in layer['layer_name'].lower() and 'intensity' in layer['layer_name'].lower():
				print(f"Found cropping intensity layer: {{layer['layer_name']}}")
				all_gdfs = []
				for url_info in layer['urls']:
					gdf = gpd.read_file(url_info['url'])
					all_gdfs.append(gdf)
				crop_gdf = pd.concat(all_gdfs, ignore_index=True)
				break

		if crop_gdf is not None:
			print(f"Cropping intensity GDF columns: {{crop_gdf.columns.tolist()}}")
			print(f"Cropping intensity GDF shape: {{crop_gdf.shape}}")

			# Step 2: Extract year columns (e.g., 'cropping_intensity_2017')
			import re
			year_cols = [col for col in crop_gdf.columns if 'cropping_intensity_' in col.lower() and re.search(r'\\d{{4}}', col)]
			year_cols = sorted(year_cols)
			print(f"Year columns found: {{year_cols}}")

			# Step 3: Compute average CI per year across ALL MWS
			years_data = []
			for col in year_cols:
				year_match = re.search(r'(\\d{{4}})', col)
				if year_match:
					year = int(year_match.group(1))
					avg_value = crop_gdf[col].mean()
					years_data.append((year, avg_value))
			years_data.sort()
			years_list = [y[0] for y in years_data]
			values_list = [y[1] for y in years_data]
			print(f"Years: {{years_list}}")
			print(f"Values: {{values_list}}")

			# Step 4: Plot time-series PNG
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
			fig, ax = plt.subplots(figsize=(10, 6))
			ax.plot(years_list, values_list, marker='o', linewidth=2, color='#3b82f6')
			ax.fill_between(years_list, values_list, alpha=0.15, color='#3b82f6')
			ax.set_xlabel('Year', fontsize=12)
			ax.set_ylabel('Average Cropping Intensity', fontsize=12)
			ax.set_title('Cropping Intensity Over Years — Navalgund, Dharwad', fontsize=14)
			ax.grid(True, alpha=0.3)
			ax.set_xticks(years_list)
			plt.tight_layout()
			plt.savefig('./exports/cropping_intensity_over_years.png', dpi=150, bbox_inches='tight')
			plt.close()
			print("Saved: ./exports/cropping_intensity_over_years.png")

			# Step 5: Compute per-MWS mean CI and export GeoJSON
			crop_gdf['mean_ci'] = crop_gdf[year_cols].mean(axis=1)
			export_cols = ['geometry', 'mean_ci'] + year_cols
			if 'uid' in crop_gdf.columns:
				export_cols = ['uid'] + export_cols
			crop_export = crop_gdf[[c for c in export_cols if c in crop_gdf.columns]]
			crop_export.to_file('./exports/cropping_intensity_by_mws.geojson', driver='GeoJSON')
			print(f"Saved: ./exports/cropping_intensity_by_mws.geojson ({{len(crop_export)}} MWS)")

			final_answer(f"Cropping intensity trend plotted for {{len(years_list)}} years ({{years_list[0]}}–{{years_list[-1]}}). Average CI ranged from {{min(values_list):.2f}} to {{max(values_list):.2f}}.\\n\\nExports:\\n- ./exports/cropping_intensity_over_years.png\\n- ./exports/cropping_intensity_by_mws.geojson")
		else:
			print("ERROR: Could not find Cropping Intensity layer!")
			final_answer("Could not find Cropping Intensity layer from CoreStack data.")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 1b: SURFACE WATER AVAILABILITY OVER YEARS (Vector)  ║
		# ║ ⚠️ For Query Type 2, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ Produces: time-series PNG + per-MWS GeoJSON                  ║
		# ╚══════════════════════════════════════════════════════════════╝
		# FIRST: Print all vector layer names to find the right one
		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 1: Find surface water bodies layer (NOT the ZOI one)
		sw_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'surface water' in lname and 'zoi' not in lname:
				print(f"Found surface water layer: {{layer['layer_name']}}")
				all_gdfs = []
				for url_info in layer['urls']:
					gdf = gpd.read_file(url_info['url'])
					all_gdfs.append(gdf)
				sw_gdf = pd.concat(all_gdfs, ignore_index=True)
				break

		if sw_gdf is not None:
			import numpy as np
			print(f"Surface water GDF columns: {{sw_gdf.columns.tolist()}}")
			print(f"Surface water GDF shape: {{sw_gdf.shape}}")

			# Step 2: Identify area columns (pattern: area_YY-YY)
			import re
			area_cols = [col for col in sw_gdf.columns if re.match(r'area_\\d{{2}}-\\d{{2}}', col)]
			area_cols = sorted(area_cols)
			print(f"Area columns found: {{area_cols}}")

			# Step 3: Aggregate by MWS_UID (sum water body areas per MWS per year)
			if 'MWS_UID' in sw_gdf.columns:
				# Convert area cols to numeric
				for col in area_cols:
					sw_gdf[col] = pd.to_numeric(sw_gdf[col], errors='coerce').fillna(0)

				sw_by_mws = sw_gdf.groupby('MWS_UID')[area_cols].sum().reset_index()
				print(f"Aggregated to {{len(sw_by_mws)}} MWS")

				# Step 4: Compute total water area per year across all MWS
				years_data = []
				for col in area_cols:
					# Extract hydro-year label: area_17-18 → "2017-18"
					m = re.search(r'(\\d{{2}})-(\\d{{2}})', col)
					if m:
						yr_start = 2000 + int(m.group(1))
						total_area = sw_by_mws[col].sum()
						years_data.append((yr_start, total_area, col))
				years_data.sort()
				years_list = [y[0] for y in years_data]
				area_list = [y[1] for y in years_data]
				labels_list = [f"{{y[0]}}-{{str(y[0]+1)[-2:]}}" for y in years_data]
				print(f"Years: {{years_list}}")
				print(f"Total areas (ha): {{[round(a, 2) for a in area_list]}}")

				# Step 5: Plot time-series PNG
				import matplotlib
				matplotlib.use('Agg')
				import matplotlib.pyplot as plt
				fig, ax = plt.subplots(figsize=(10, 6))
				ax.plot(years_list, area_list, marker='s', linewidth=2, color='#06b6d4')
				ax.fill_between(years_list, area_list, alpha=0.15, color='#06b6d4')
				ax.set_xlabel('Hydro-Year Start', fontsize=12)
				ax.set_ylabel('Total Surface Water Area (hectares)', fontsize=12)
				ax.set_title('Surface Water Availability Over Years — Navalgund, Dharwad', fontsize=14)
				ax.grid(True, alpha=0.3)
				ax.set_xticks(years_list)
				ax.set_xticklabels(labels_list, rotation=30, ha='right')
				plt.tight_layout()
				plt.savefig('./exports/surface_water_trend.png', dpi=150, bbox_inches='tight')
				plt.close()
				print("Saved: ./exports/surface_water_trend.png")

				# Step 6: Build per-MWS GeoJSON export
				# Get one representative geometry per MWS from the original sw_gdf (dissolve)
				sw_dissolved = sw_gdf.dissolve(by='MWS_UID', aggfunc='sum').reset_index()
				export_cols_sw = ['MWS_UID', 'geometry'] + area_cols
				sw_export = sw_dissolved[[c for c in export_cols_sw if c in sw_dissolved.columns]]
				sw_export.to_file('./exports/surface_water_availability_by_mws.geojson', driver='GeoJSON')
				print(f"Saved: ./exports/surface_water_availability_by_mws.geojson ({{len(sw_export)}} MWS)")

				final_answer(f"Surface water trend plotted for {{len(years_list)}} years. Total water area ranged from {{min(area_list):.1f}} to {{max(area_list):.1f}} hectares.\\n\\nExports:\\n- ./exports/surface_water_trend.png\\n- ./exports/surface_water_availability_by_mws.geojson")
			else:
				print("ERROR: MWS_UID column not found in surface water data!")
				final_answer("Could not aggregate surface water data — MWS_UID column not found.")
		else:
			print("ERROR: Could not find Surface Water Bodies layer!")
			final_answer("Could not find Surface Water Bodies layer from CoreStack data.")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 2: TREE COVER LOSS / DEFORESTATION (Raster)          ║
		# ║ ⚠️ For Query Type 3, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ ⚠️ NEVER use bare open() — use rasterio.MemoryFile(bytes).   ║
		# ║ ⚠️ NEVER reproject to UTM — use degree-based math for area.  ║
		# ║ Produces: deforestation map PNG + loss polygons GeoJSON       ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, requests, rasterio, numpy as np, math
		from rasterio.features import shapes as rio_shapes
		from shapely.geometry import shape as shp_shape
		os.makedirs('./exports', exist_ok=True)

		# FIRST: Print all raster layer names to find the right one
		print("Available raster layers:")
		for layer in raster_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 1: Find deforestation raster (case-insensitive substring match)
		# ⚠️ The ACTUAL layer name is "Change Detection Raster (change_dharwad_navalgund_Deforestation)"
		#    — it contains 'deforestation', NOT 'tree cover loss'!
		loss_layer = None
		for layer in raster_layers:
			lname = layer['layer_name'].lower()
			if 'deforestation' in lname:
				loss_layer = layer
				print(f"Found deforestation layer: {{layer['layer_name']}}")
				break
		# Fallback: try 'tree' + 'overall' + 'change'
		if loss_layer is None:
			for layer in raster_layers:
				lname = layer['layer_name'].lower()
				if 'tree' in lname and 'overall' in lname and 'change' in lname:
					loss_layer = layer
					print(f"Found tree overall change layer: {{layer['layer_name']}}")
					break

		if loss_layer is not None:
			# Step 2: Download raster bytes (sandbox-safe: use MemoryFile, NOT bare open())
			url = loss_layer['urls'][0]['url']
			print(f"Downloading: {{url[:100]}}...")
			r_bytes = requests.get(url, timeout=120).content
			print(f"Downloaded: {{len(r_bytes)}} bytes")

			with rasterio.MemoryFile(r_bytes) as memfile:
				with memfile.open() as src:
					loss_data = src.read(1)
					transform = src.transform
					crs = src.crs
					bounds = src.bounds

					# Step 3: Mask to loss class (class == 1)
					loss_data_clean = np.nan_to_num(loss_data, nan=0)
					loss_mask = (loss_data_clean == 1).astype(np.uint8)
					loss_pixels = int(loss_mask.sum())
					print(f"Tree cover loss pixels: {{loss_pixels}}")

					# Step 4: Compute area using degree-based math (NEVER UTM reproject)
					dx_deg = abs(transform[0])
					dy_deg = abs(transform[4])
					center_lat = (bounds.bottom + bounds.top) / 2
					m_per_deg_lat = 110540.0
					m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
					px_area_ha = (dx_deg * m_per_deg_lon) * (dy_deg * m_per_deg_lat) / 10000
					loss_area_ha = loss_pixels * px_area_ha
					print(f"Tree cover loss area: {{loss_area_ha:.2f}} hectares ({{px_area_ha:.6f}} ha/px)")

					# Step 5: Create visualization PNG
					import matplotlib
					matplotlib.use('Agg')
					import matplotlib.pyplot as plt
					from matplotlib.colors import ListedColormap

					fig, ax = plt.subplots(figsize=(10, 10))
					# Background: no-loss in dark grey, loss in red
					display_data = np.where(loss_mask == 1, 2, np.where(loss_data_clean > 0, 1, 0))
					cmap = ListedColormap(['#1a1a2e', '#2d3436', '#e74c3c'])
					ax.imshow(display_data, cmap=cmap, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
					ax.set_title(f'Tree Cover Loss — Navalgund, Dharwad\\n({{loss_area_ha:.1f}} hectares lost)', fontsize=14)
					ax.set_xlabel('Longitude')
					ax.set_ylabel('Latitude')
					# Legend
					from matplotlib.patches import Patch
					legend_elements = [Patch(facecolor='#e74c3c', label=f'Loss ({{loss_area_ha:.1f}} ha)'),
					                   Patch(facecolor='#2d3436', label='No change')]
					ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
					plt.tight_layout()
					plt.savefig('./exports/deforestation_map.png', dpi=150, bbox_inches='tight')
					plt.close()
					print("Saved: ./exports/deforestation_map.png")

					# Step 6: Vectorize loss pixels → GeoJSON polygons
					loss_polys = []
					for geom, val in rio_shapes(loss_mask, transform=transform):
						if val == 1:
							loss_polys.append(shp_shape(geom))

					if loss_polys:
						loss_gdf = gpd.GeoDataFrame(
							{{'loss_type': ['tree_cover_loss'] * len(loss_polys),
							  'area_ha': [p.area * m_per_deg_lon * m_per_deg_lat / 10000 for p in loss_polys]}},
							geometry=loss_polys,
							crs=crs
						)
						loss_gdf.to_file('./exports/deforestation_navalgund.geojson', driver='GeoJSON')
						print(f"Saved: ./exports/deforestation_navalgund.geojson ({{len(loss_gdf)}} polygons)")
					else:
						print("No loss polygons to vectorize")

					final_answer(f"Tree cover loss analysis complete: {{loss_area_ha:.1f}} hectares of forest lost in Navalgund, Dharwad.\\n\\nExports:\\n- ./exports/deforestation_map.png\\n- ./exports/deforestation_navalgund.geojson")
		else:
			print("ERROR: Could not find tree cover loss raster layer!")
			final_answer("Could not find tree cover loss raster from CoreStack data.")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 2b: CROPLAND → BUILT-UP CHANGE DETECTION             ║
		# ║ ⚠️ For Query Type 4, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ Downloads 2 LULC rasters, computes crop→built-up change      ║
		# ║ ⚠️ NEVER use bare open() — it is BLOCKED by the sandbox.     ║
		# ║    Use rasterio.MemoryFile for reading raster bytes.         ║
		# ║    Use rasterio.open() for writing GeoTIFFs (it's allowed).  ║
		# ║ ⚠️ NEVER reproject to UTM — the grid is too large and will   ║
		# ║    cause a 300+ PB memory allocation error. Instead, compute ║
		# ║    area using degree-based math (accurate to <1% at 15°N).   ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, requests, rasterio, numpy as np, math
		os.makedirs('./exports', exist_ok=True)

		# ═══ PARAMETERS — Extract these from the user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state

		# Step 1: Fetch data from CoreStack
		data = fetch_corestack_data(query=f"{{LOCATION}} LULC raster")
		raster_layers = data['spatial_data']['raster_layers']

		print("Available raster layers:")
		for layer in raster_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 2: Find LULC level_3 rasters for old and new years
		lulc_layers = []
		for layer in raster_layers:
			lname = layer['layer_name'].lower()
			if 'lulc' in lname and 'level_3' in lname:
				lulc_layers.append(layer)

		# Sort by year code (extract the YY_YY part) to find earliest and latest
		import re as re_mod
		def extract_year(name):
			m = re_mod.search(r'lulc_(\\d{{2}})_(\\d{{2}})', name.lower())
			if m:
				return int(m.group(1))
			return 0

		lulc_layers.sort(key=lambda l: extract_year(l['layer_name']))
		print(f"Found {{len(lulc_layers)}} LULC level_3 layers")

		# Pick old (closest to query year, e.g., 2018 -> LULC_18_19) and newest
		old_layer = lulc_layers[0]   # earliest
		new_layer = lulc_layers[-1]  # most recent
		# If user specifies a year (e.g., 2018), find closest match
		for ll in lulc_layers:
			if '18_19' in ll['layer_name']:
				old_layer = ll
				break

		print(f"Old LULC: {{old_layer['layer_name']}}")
		print(f"New LULC: {{new_layer['layer_name']}}")

		# Step 3: Download both rasters into memory (NEVER use bare open())
		old_url = old_layer['urls'][0]['url']
		new_url = new_layer['urls'][0]['url']

		r_old_bytes = requests.get(old_url, timeout=120).content
		print(f"Downloaded old LULC: {{len(r_old_bytes)}} bytes")
		r_new_bytes = requests.get(new_url, timeout=120).content
		print(f"Downloaded new LULC: {{len(r_new_bytes)}} bytes")

		# Step 4: Open rasters using rasterio.MemoryFile (sandbox-safe)
		with rasterio.MemoryFile(r_old_bytes) as memfile:
			with memfile.open() as src_old:
				old_data = np.nan_to_num(src_old.read(1), nan=0).astype(int)
				old_meta = src_old.meta.copy()
				old_transform = src_old.transform
				old_crs = src_old.crs
				old_bounds = src_old.bounds
				print(f"Old raster shape: {{old_data.shape}}, CRS: {{old_crs}}")

		with rasterio.MemoryFile(r_new_bytes) as memfile:
			with memfile.open() as src_new:
				new_data = np.nan_to_num(src_new.read(1), nan=0).astype(int)
				print(f"New raster shape: {{new_data.shape}}")

		# Step 5: Compute cropland-to-built-up change
		#   Cropland classes: 8 (single), 9 (single non-kharif), 10 (double), 11 (triple)
		#   Built-up class: 1
		crop_classes = [8, 9, 10, 11]
		crop_old = np.isin(old_data, crop_classes)
		builtup_new = (new_data == 1)
		crop_to_builtup = (crop_old & builtup_new).astype(np.uint8)
		change_pixels = int(crop_to_builtup.sum())
		print(f"Crop -> Built-up pixels: {{change_pixels}}")

		# Step 6: Compute area using degree-based math (NO UTM reprojection!)
		# UTM reprojection would create a 300+ PB grid and crash.
		# Instead: pixel_area = (dx_deg * m_per_deg_lon) * (dy_deg * m_per_deg_lat)
		dx_deg = abs(old_transform[0])
		dy_deg = abs(old_transform[4])
		center_lat = (old_bounds.bottom + old_bounds.top) / 2
		m_per_deg_lat = 110540.0  # meters per degree latitude (approx constant)
		m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
		px_width_m = dx_deg * m_per_deg_lon
		px_height_m = dy_deg * m_per_deg_lat
		px_area_m2 = px_width_m * px_height_m
		px_area_ha = px_area_m2 / 10000
		total_change_ha = change_pixels * px_area_ha
		print(f"Pixel size: {{px_width_m:.2f}} x {{px_height_m:.2f}} m = {{px_area_ha:.5f}} ha/px")
		print(f"Crop -> Built-up area: {{total_change_ha:.2f}} hectares")

		# Also compute total cropland area (old) and total built-up area (both years)
		total_crop_old_px = int(crop_old.sum())
		total_builtup_old_px = int((old_data == 1).sum())
		total_builtup_new_px = int((new_data == 1).sum())
		print(f"Total cropland (old): {{total_crop_old_px * px_area_ha:.0f}} ha")
		print(f"Total built-up (old): {{total_builtup_old_px * px_area_ha:.0f}} ha")
		print(f"Total built-up (new): {{total_builtup_new_px * px_area_ha:.0f}} ha")

		# Step 7: Save change raster as GeoTIFF (rasterio.open is allowed)
		change_meta = old_meta.copy()
		change_meta.update(dtype='uint8', count=1, nodata=0)
		with rasterio.open('./exports/crop_to_builtup_change.tif', 'w', **change_meta) as dst:
			dst.write(crop_to_builtup, 1)
		print(f"Change raster saved: ./exports/crop_to_builtup_change.tif")

		# Step 8: Visualization PNG
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		from matplotlib.colors import ListedColormap

		fig, axes = plt.subplots(1, 3, figsize=(20, 7))

		# Subplot 1: Old LULC
		cmap_lulc = ListedColormap(['white', 'red', 'cyan', 'blue', 'darkblue', 'white',
			'green', 'sandybrown', 'yellow', 'khaki', 'orange', 'darkgreen', 'olive'])
		axes[0].imshow(old_data, cmap=cmap_lulc, vmin=0, vmax=12)
		axes[0].set_title(f'LULC {{old_layer["layer_name"][:12]}}')
		axes[0].axis('off')

		# Subplot 2: New LULC
		axes[1].imshow(new_data, cmap=cmap_lulc, vmin=0, vmax=12)
		axes[1].set_title(f'LULC {{new_layer["layer_name"][:12]}}')
		axes[1].axis('off')

		# Subplot 3: Change map (crop->built-up highlighted in red)
		change_vis = np.zeros((*old_data.shape, 3), dtype=np.uint8)
		change_vis[..., :] = 200  # light gray background
		change_vis[crop_to_builtup == 1] = [255, 0, 0]  # red for crop->built-up
		axes[2].imshow(change_vis)
		axes[2].set_title(f'Crop to Built-up: {{total_change_ha:.1f}} ha')
		axes[2].axis('off')

		plt.suptitle(f'Cropland to Built-up Conversion\n{{LOCATION}}', fontsize=14)
		plt.tight_layout()
		plt.savefig('./exports/crop_to_builtup_change.png', dpi=200, bbox_inches='tight')
		print(f"Visualization saved: ./exports/crop_to_builtup_change.png")

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
		# ═══ PARAMETER — Extract from user's query ═══
		target_uid = '18_16157'  # ← Replace with the MWS uid from the user's query
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

	# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 7: MWS SIMILARITY — PROPENSITY SCORE MATCHING        ║
		# ║ ⚠️ For Query Type 9, COPY THIS CODE ALMOST VERBATIM.         ║
		# ║ Loads 4 layers, merges on uid, logistic regression PSM       ║
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

		# Step 4: Select numeric columns only
		exclude_cols = ['id', 'uid', 'geometry', 'area_in_ha', 'sum', 'terrainClu']
		feature_cols = [c for c in merged.columns if c not in exclude_cols and merged[c].dtype in ['float64', 'int64', 'float32', 'int32']]
		print(f"Feature columns ({{len(feature_cols)}}): {{feature_cols[:20]}}...")

		# Step 5: Build feature matrix, create treatment, normalize
		import numpy as np
		from sklearn.preprocessing import StandardScaler
		from sklearn.linear_model import LogisticRegression

		# ═══ PARAMETER — Extract from user's query ═══
		target_uid = '18_16157'  # ← Replace with the MWS uid from the user's query
		merged['treatment'] = (merged['uid'] == target_uid).astype(int)

		X = merged[feature_cols].fillna(0).values
		y = merged['treatment'].values
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		# Step 6: Fit logistic regression to estimate propensity scores
		model = LogisticRegression(max_iter=1000, solver='lbfgs')
		model.fit(X_scaled, y)
		propensity_scores = model.predict_proba(X_scaled)[:, 1]
		merged['propensity_score'] = propensity_scores
		print(f"Propensity score range: {{propensity_scores.min():.6f}} — {{propensity_scores.max():.6f}}")

		# Step 7: Get target's propensity score and compute distances
		target_idx = merged.index[merged['uid'] == target_uid]
		if len(target_idx) == 0:
			print(f"Target uid {{target_uid}} not found! Available uids: {{merged['uid'].tolist()[:10]}}")
		else:
			target_ps = merged.loc[target_idx[0], 'propensity_score']
			merged['ps_distance'] = abs(merged['propensity_score'] - target_ps)

			# Exclude target, sort by smallest distance (most similar)
			others = merged[merged['uid'] != target_uid].sort_values('ps_distance', ascending=True)
			top_n = 10
			top = others.head(top_n)[['uid', 'propensity_score', 'ps_distance']]
			print(f"Target {{target_uid}} propensity score: {{target_ps:.6f}}")
			print(f"Top {{top_n}} PSM-matched MWS to {{target_uid}}:")
			print(top.to_string())

			# Step 8: Export GeoJSON with geometries — CLEAN output
			top_uids = top['uid'].tolist()
			top_gdf = drought_gdf[drought_gdf['uid'].isin(top_uids + [target_uid])].copy()
			top_gdf = top_gdf.merge(merged[['uid', 'propensity_score', 'ps_distance']], on='uid', how='left')
			if 'id' in top_gdf.columns:
				top_gdf = top_gdf.drop(columns=['id'])
			keep_cols = ['uid', 'propensity_score', 'ps_distance', 'area_in_ha', 'avg_dryspell', 'geometry']
			for c in top_gdf.columns:
				if any(c.startswith(p) for p in ['drysp_', 'w_sev_', 'w_mod_', 'w_mld_', 'cropping_intensity', 'hill_slope', 'plain_area', 'ridge_area', 'slopy_area', 'valley_are']):
					keep_cols.append(c)
			keep_cols = [c for c in keep_cols if c in top_gdf.columns]
			top_gdf = top_gdf[keep_cols]
			top_gdf.to_file('./exports/psm_matched_microwatersheds.geojson', driver='GeoJSON')

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 8: RANK TOP-K MWS BY CROPPING INTENSITY & WATER    ║
		# ║ ⚠️ For Query Type 10, COPY THIS CODE ALMOST VERBATIM.        ║
		# ║ Reads past exports, fetches LULC, computes weighted scores  ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, glob
		# Step 1: Read UIDs from past exports
		drought_sens_path = './exports/top_drought_sensitive_microwatersheds.geojson'
		sw_sens_path = './exports/top_sw_sensitive_microwatersheds.geojson'

		drought_sens_gdf = gpd.read_file(drought_sens_path)
		sw_sens_gdf = gpd.read_file(sw_sens_path)
		drought_uids = drought_sens_gdf['uid'].tolist()
		sw_uids = sw_sens_gdf['uid'].tolist()
		all_uids = list(set(drought_uids + sw_uids))
		print(f"Drought-sensitive UIDs ({{len(drought_uids)}}): {{drought_uids}}")
		print(f"SW-sensitive UIDs ({{len(sw_uids)}}): {{sw_uids}}")

		# Step 2: Fetch LULC vector layer from CoreStack
		# ═══ PARAMETER — Extract from user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state
		data = fetch_corestack_data(query=f"{{LOCATION}} LULC vector data")
		vector_layers = data['spatial_data']['vector_layers']

		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 3: Load LULC vector
		lulc_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'lulc' in lname and 'terrain' not in lname and 'level' not in lname and lulc_gdf is None:
				print(f"Found LULC layer: {{layer['layer_name']}}")
				lulc_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)

		print(f"LULC shape: {{lulc_gdf.shape}}")
		print(f"LULC columns: {{lulc_gdf.columns.tolist()}}")

		# Step 4: Filter to only relevant UIDs
		lulc_filtered = lulc_gdf[lulc_gdf['uid'].isin(all_uids)].copy()
		print(f"Filtered LULC to {{len(lulc_filtered)}} MWS")

		# Step 5: Compute CROPPING INTENSITY weighted score
		# triply_cro, triply_c_1..triply_c_7 (2017-2024)
		# doubly_cro, doubly_c_1..doubly_c_7
		# single_kha, single_k_1..single_k_7 + single_non, single_n_1..single_n_7
		triple_cols = ['triply_cro'] + [f'triply_c_{{i}}' for i in range(1, 8)]
		double_cols = ['doubly_cro'] + [f'doubly_c_{{i}}' for i in range(1, 8)]
		single_k_cols = ['single_kha'] + [f'single_k_{{i}}' for i in range(1, 8)]
		single_nk_cols = ['single_non'] + [f'single_n_{{i}}' for i in range(1, 8)]

		lulc_filtered['avg_triple'] = lulc_filtered[triple_cols].mean(axis=1)
		lulc_filtered['avg_double'] = lulc_filtered[double_cols].mean(axis=1)
		lulc_filtered['avg_single'] = lulc_filtered[single_k_cols].mean(axis=1) + lulc_filtered[single_nk_cols].mean(axis=1)
		lulc_filtered['ci_score'] = 3 * lulc_filtered['avg_triple'] + 2 * lulc_filtered['avg_double'] + 1 * lulc_filtered['avg_single']
		print(f"Cropping intensity scores computed")

		# Step 6: Compute SURFACE WATER weighted score
		# krz_water_, krz_wate_1..krz_wate_7 (perennial)
		# kr_water_a, kr_water_1..kr_water_7 (winter)
		# k_water_ar, k_water__1..k_water__7 (monsoon)
		krz_cols = ['krz_water_'] + [f'krz_wate_{{i}}' for i in range(1, 8)]
		kr_cols = ['kr_water_a'] + [f'kr_water_{{i}}' for i in range(1, 8)]
		k_cols = ['k_water_ar'] + [f'k_water__{{i}}' for i in range(1, 8)]

		lulc_filtered['avg_perennial'] = lulc_filtered[krz_cols].mean(axis=1)
		lulc_filtered['avg_winter'] = lulc_filtered[kr_cols].mean(axis=1)
		lulc_filtered['avg_monsoon'] = lulc_filtered[k_cols].mean(axis=1)
		lulc_filtered['sw_score'] = 3 * lulc_filtered['avg_perennial'] + 2 * lulc_filtered['avg_winter'] + 1 * lulc_filtered['avg_monsoon']
		print(f"Surface water scores computed")

		# Step 7: Rank drought-sensitive MWS by cropping intensity score
		drought_ranked = lulc_filtered[lulc_filtered['uid'].isin(drought_uids)][['uid', 'ci_score', 'avg_triple', 'avg_double', 'avg_single']].sort_values('ci_score', ascending=False).reset_index(drop=True)
		drought_ranked.index = drought_ranked.index + 1  # 1-based rank
		drought_ranked.index.name = 'rank'
		print(f"\nDrought-sensitive MWS ranked by Cropping Intensity Score:")
		print(f"Formula: ci_score = 3*triple + 2*double + 1*(single_kharif + single_non_kharif)")
		print(drought_ranked.to_string())

		# Step 8: Rank sw-sensitive MWS by surface water score
		sw_ranked = lulc_filtered[lulc_filtered['uid'].isin(sw_uids)][['uid', 'sw_score', 'avg_perennial', 'avg_winter', 'avg_monsoon']].sort_values('sw_score', ascending=False).reset_index(drop=True)
		sw_ranked.index = sw_ranked.index + 1
		sw_ranked.index.name = 'rank'
		print(f"\nSW-sensitive MWS ranked by Surface Water Availability Score:")
		print(f"Formula: sw_score = 3*perennial + 2*winter + 1*monsoon")
		print(sw_ranked.to_string())

		# Step 9: Export combined results as GeoJSON
		export_gdf = lulc_filtered[lulc_filtered['uid'].isin(all_uids)].copy()
		export_gdf = export_gdf[['uid', 'ci_score', 'sw_score', 'avg_triple', 'avg_double', 'avg_single', 'avg_perennial', 'avg_winter', 'avg_monsoon', 'geometry']]
		export_gdf['is_drought_sensitive'] = export_gdf['uid'].isin(drought_uids)
		export_gdf['is_sw_sensitive'] = export_gdf['uid'].isin(sw_uids)
		export_gdf.to_file('./exports/ranked_mws_by_ci_and_sw.geojson', driver='GeoJSON')

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 9: SC/ST% vs NREGA WORKS SCATTER PLOT              ║
		# ║ ⚠️ For Query Type 11, COPY THIS CODE ALMOST VERBATIM.        ║
		# ║ Loads Admin Boundary + NREGA, spatial joins, scatter plot   ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os
		os.makedirs('./exports', exist_ok=True)

		# Step 1: Fetch data from CoreStack
		# ═══ PARAMETER — Extract from user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state
		data = fetch_corestack_data(query=f"{{LOCATION}} NREGA admin boundary villages")
		vector_layers = data['spatial_data']['vector_layers']

		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 2: Load Admin Boundary and NREGA layers
		admin_gdf = None
		nrega_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			# Admin Boundary: has 'admin' or 'panchayat' in name
			if ('admin' in lname or 'panchayat' in lname) and admin_gdf is None:
				print(f"Found Admin layer: {{layer['layer_name']}}")
				admin_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)
			# NREGA Assets: has 'nrega' in name
			if 'nrega' in lname and nrega_gdf is None:
				print(f"Found NREGA layer: {{layer['layer_name']}}")
				nrega_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)

		print(f"Admin shape: {{admin_gdf.shape}}, columns: {{admin_gdf.columns.tolist()}}")
		print(f"NREGA shape: {{nrega_gdf.shape}}, columns: {{nrega_gdf.columns.tolist()}}")

		# Step 3: Aggregate Admin boundary by village
		# Sum P_SC, P_ST, TOT_P per village, compute SC/ST%
		admin_stats = admin_gdf.groupby('vill_name').agg({{'P_SC': 'sum', 'P_ST': 'sum', 'TOT_P': 'sum'}}).reset_index()
		admin_stats = admin_stats[admin_stats['TOT_P'] > 0].copy()
		admin_stats['sc_st_pct'] = (admin_stats['P_SC'] + admin_stats['P_ST']) / admin_stats['TOT_P'] * 100
		print(f"Villages with population data: {{len(admin_stats)}}")

		# Step 4: Dissolve admin polygons by village name (one polygon per village)
		admin_dissolved = admin_gdf.dissolve(by='vill_name').reset_index()

		# Step 5: Spatial join — assign each NREGA work (point) to its village polygon
		nrega_gdf = nrega_gdf.to_crs(admin_dissolved.crs)
		joined = gpd.sjoin(nrega_gdf[['geometry']], admin_dissolved[['vill_name', 'geometry']], how='left', predicate='within')
		works_per_village = joined.groupby('vill_name').size().reset_index(name='nrega_count')
		print(f"Villages with NREGA works: {{len(works_per_village)}}")

		# Step 6: Merge SC/ST% with NREGA count
		merged = admin_stats.merge(works_per_village, on='vill_name', how='inner')
		print(f"Merged villages: {{len(merged)}}")
		print(merged[['vill_name', 'sc_st_pct', 'nrega_count']].sort_values('nrega_count', ascending=False).to_string())

		# Step 7: Build scatter plot
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import numpy as np
		from scipy import stats

		fig, ax = plt.subplots(figsize=(14, 10))
		ax.scatter(merged['sc_st_pct'], merged['nrega_count'], s=80, alpha=0.7, edgecolors='black', linewidth=0.5, c='steelblue')

		# Annotate each point with village name
		for _, row in merged.iterrows():
			ax.annotate(row['vill_name'], (row['sc_st_pct'], row['nrega_count']),
						fontsize=6, alpha=0.75, ha='left', va='bottom',
						xytext=(4, 4), textcoords='offset points')

		# Add trend line
		slope, intercept, r_value, p_value, std_err = stats.linregress(merged['sc_st_pct'], merged['nrega_count'])
		x_line = np.linspace(merged['sc_st_pct'].min(), merged['sc_st_pct'].max(), 100)
		ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=1.5, label=f'Trend (r={{r_value:.3f}}, p={{p_value:.3f}})')

		ax.set_xlabel('SC/ST Population (%)', fontsize=12)
		ax.set_ylabel('Number of NREGA Works', fontsize=12)
		ax.set_title(f'SC/ST Population % vs NREGA Works per Village\n{{LOCATION}} (Pearson r={{r_value:.3f}})', fontsize=13)
		ax.legend(fontsize=10)
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig('./exports/scst_vs_nrega_scatter.png', dpi=200, bbox_inches='tight')
		print(f"Scatter plot saved to ./exports/scst_vs_nrega_scatter.png")

		# Step 8: Export merged data as GeoJSON
		export_gdf = admin_dissolved[admin_dissolved['vill_name'].isin(merged['vill_name'])].copy()
		export_gdf = export_gdf[['vill_name', 'geometry']].merge(merged[['vill_name', 'sc_st_pct', 'nrega_count', 'P_SC', 'P_ST', 'TOT_P']], on='vill_name')
		export_gdf = gpd.GeoDataFrame(export_gdf, geometry='geometry')
		export_gdf.to_file('./exports/scst_vs_nrega_villages.geojson', driver='GeoJSON')
		print(f"GeoJSON exported to ./exports/scst_vs_nrega_villages.geojson")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 10: CI vs RUNOFF VOLUME — 4 QUADRANT SCATTER       ║
		# ║ ⚠️ For Query Type 12, COPY THIS CODE ALMOST VERBATIM.       ║
		# ║ Loads Cropping Intensity + Drought layers, computes metrics ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os
		os.makedirs('./exports', exist_ok=True)

		# Step 1: Fetch data from CoreStack
		# ═══ PARAMETER — Extract from user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state
		data = fetch_corestack_data(query=f"{{LOCATION}} cropping intensity drought")
		vector_layers = data['spatial_data']['vector_layers']

		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 2: Load Cropping Intensity and Drought layers
		ci_gdf = None
		drought_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'intensity' in lname and ci_gdf is None:
				print(f"Found Cropping Intensity layer: {{layer['layer_name']}}")
				ci_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)
			if 'drought' in lname and 'causal' not in lname and drought_gdf is None:
				print(f"Found Drought layer: {{layer['layer_name']}}")
				drought_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)

		print(f"CI shape: {{ci_gdf.shape}}, columns: {{ci_gdf.columns.tolist()}}")
		print(f"Drought shape: {{drought_gdf.shape}}")

		# Step 3: Compute MEAN CROPPING INTENSITY per MWS (average over years)
		ci_cols = [c for c in ci_gdf.columns if c.startswith('cropping_intensity_')]
		print(f"CI year columns: {{ci_cols}}")
		ci_gdf['mean_ci'] = ci_gdf[ci_cols].mean(axis=1)
		print(f"Mean CI range: {{ci_gdf['mean_ci'].min():.1f}} to {{ci_gdf['mean_ci'].max():.1f}}")

		# Step 4: Compute MEAN ANNUAL RUNOFF VOLUME per MWS from rd columns
		# rd columns: rdYY-M-D (weekly monsoon water-balance values)
		rd_cols = [c for c in drought_gdf.columns if c.startswith('rd') and '-' in c]
		print(f"Total rd columns: {{len(rd_cols)}}")

		# Group by year (c[2:4] = year digits)
		rd_by_year = {{}}
		for c in rd_cols:
			yr = c[2:4]
			if yr not in rd_by_year:
				rd_by_year[yr] = []
			rd_by_year[yr].append(c)

		# Sum per year → annual monsoon surplus (mm)
		annual_sum_cols = []
		for yr in sorted(rd_by_year.keys()):
			col_name = f'rd_sum_20{{yr}}'
			drought_gdf[col_name] = drought_gdf[rd_by_year[yr]].sum(axis=1)
			annual_sum_cols.append(col_name)
			print(f"  Year 20{{yr}}: {{len(rd_by_year[yr])}} weeks, mean sum={{drought_gdf[col_name].mean():.0f}} mm")

		# Mean across years → mean annual surplus (mm)
		drought_gdf['mean_annual_surplus_mm'] = drought_gdf[annual_sum_cols].mean(axis=1)
		# Volume = surplus_mm × area_in_ha × 10  (1 mm over 1 ha = 10 m³)
		drought_gdf['runoff_volume_m3'] = drought_gdf['mean_annual_surplus_mm'] * drought_gdf['area_in_ha'] * 10
		# Clip negative volumes to small positive (deficit MWS)
		drought_gdf['runoff_volume_m3'] = drought_gdf['runoff_volume_m3'].clip(lower=1)
		print(f"Runoff volume range: {{drought_gdf['runoff_volume_m3'].min():.0f}} to {{drought_gdf['runoff_volume_m3'].max():.0f}} m³")

		# Step 5: Merge CI and Drought on uid
		merged = ci_gdf[['uid', 'mean_ci', 'area_in_ha', 'geometry']].merge(
			drought_gdf[['uid', 'mean_annual_surplus_mm', 'runoff_volume_m3']],
			on='uid', how='inner'
		)
		print(f"Merged MWS count: {{len(merged)}}")

		# Step 6: Set thresholds (median)
		ci_thresh = merged['mean_ci'].median()
		rv_thresh = merged['runoff_volume_m3'].median()
		print(f"Thresholds — CI median: {{ci_thresh:.1f}}%, Runoff median: {{rv_thresh:.0f}} m³")

		# Assign quadrants
		def assign_quadrant(row):
			if row['mean_ci'] >= ci_thresh and row['runoff_volume_m3'] >= rv_thresh:
				return 'High CI + High Runoff'
			elif row['mean_ci'] < ci_thresh and row['runoff_volume_m3'] >= rv_thresh:
				return 'Low CI + High Runoff'
			elif row['mean_ci'] >= ci_thresh and row['runoff_volume_m3'] < rv_thresh:
				return 'High CI + Low Runoff'
			else:
				return 'Low CI + Low Runoff'
		merged['quadrant'] = merged.apply(assign_quadrant, axis=1)
		print(f"\nQuadrant counts:")
		print(merged['quadrant'].value_counts().to_string())

		# Step 7: Build scatter plot
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import numpy as np

		quadrant_colors = {{
			'High CI + High Runoff': 'red',
			'Low CI + High Runoff': 'royalblue',
			'High CI + Low Runoff': 'darkorange',
			'Low CI + Low Runoff': 'gray'
		}}

		fig, ax = plt.subplots(figsize=(14, 10))
		for quad, color in quadrant_colors.items():
			subset = merged[merged['quadrant'] == quad]
			if len(subset) > 0:
				ax.scatter(subset['mean_ci'], subset['runoff_volume_m3'],
						c=color, s=70, alpha=0.8, edgecolors='black', linewidth=0.5,
						label=f'{{quad}} (n={{len(subset)}})')

		# Annotate each point with uid
		for _, row in merged.iterrows():
			ax.annotate(row['uid'], (row['mean_ci'], row['runoff_volume_m3']),
						fontsize=5, alpha=0.7, ha='left', va='bottom',
						xytext=(3, 3), textcoords='offset points')

		# Quadrant threshold lines
		ax.axvline(ci_thresh, color='black', linestyle='--', alpha=0.5, linewidth=1)
		ax.axhline(rv_thresh, color='black', linestyle='--', alpha=0.5, linewidth=1)

		# Log scale on Y if spread is large
		if merged['runoff_volume_m3'].max() / max(merged['runoff_volume_m3'].min(), 1) > 10:
			ax.set_yscale('log')
			ax.set_ylabel('Mean Annual Runoff Volume (m³, log scale)', fontsize=12)
		else:
			ax.set_ylabel('Mean Annual Runoff Volume (m³)', fontsize=12)

		ax.set_xlabel('Mean Cropping Intensity (%)', fontsize=12)
		ax.set_title(f'Micro-watershed: Cropping Intensity vs Harvestable Runoff Volume\n{{LOCATION}} | CI threshold={{ci_thresh:.1f}}%, Runoff threshold={{rv_thresh:.0f}} m³', fontsize=12)
		ax.legend(fontsize=9, loc='upper left')
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig('./exports/ci_vs_runoff_quadrant_scatter.png', dpi=200, bbox_inches='tight')
		print(f"\nScatter plot saved to ./exports/ci_vs_runoff_quadrant_scatter.png")

		# Step 8: Print top MWS in key quadrants
		for quad in ['High CI + High Runoff', 'Low CI + High Runoff', 'High CI + Low Runoff']:
			q_df = merged[merged['quadrant'] == quad][['uid', 'mean_ci', 'runoff_volume_m3', 'area_in_ha']].sort_values('runoff_volume_m3', ascending=False)
			print(f"\n{{quad}} ({{len(q_df)}} MWS):")
			print(q_df.head(10).to_string(index=False))

		# Step 9: Export GeoJSON with quadrant assignments
		export_gdf = gpd.GeoDataFrame(merged, geometry='geometry')
		export_gdf = export_gdf[['uid', 'mean_ci', 'runoff_volume_m3', 'mean_annual_surplus_mm', 'area_in_ha', 'quadrant', 'geometry']]
		export_gdf.to_file('./exports/ci_vs_runoff_quadrants.geojson', driver='GeoJSON')
		print(f"GeoJSON exported to ./exports/ci_vs_runoff_quadrants.geojson")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 11: MONSOON LST vs CROPPING INTENSITY SCATTER      ║
		# ║ ⚠️ For Query Type 13, COPY THIS CODE ALMOST VERBATIM.       ║
		# ║ Fetches CI from CoreStack, LST from GEE Landsat 8          ║
		# ║ Uses CENTROID lat/lon for GEE sampling (NOT polygon IDs)   ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, json
		os.makedirs('./exports', exist_ok=True)

		# Step 1: Fetch data from CoreStack
		# ═══ PARAMETER — Extract from user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state
		result = fetch_corestack_data(query=f"{{LOCATION}} cropping intensity")
		data = json.loads(result)
		vector_layers = data['spatial_data']['vector_layers']

		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 2: Load Cropping Intensity layer ONLY (no admin boundary needed)
		import geopandas as gpd
		import pandas as pd
		ci_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'cropping' in lname and 'intensity' in lname and ci_gdf is None:
				print(f"Found Cropping Intensity layer: {{layer['layer_name']}}")
				ci_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')

		print(f"CI shape: {{ci_gdf.shape}}, columns: {{ci_gdf.columns.tolist()}}")

		# Step 3: Compute mean cropping intensity per MWS (average over 2017-2023)
		ci_cols = [c for c in ci_gdf.columns if c.startswith('cropping_intensity_')]
		ci_cols_filtered = [c for c in ci_cols if int(c.split('_')[-1]) >= 2017 and int(c.split('_')[-1]) <= 2023]
		ci_gdf['mean_ci'] = ci_gdf[ci_cols_filtered].mean(axis=1)
		print(f"CI year columns used: {{ci_cols_filtered}}")
		print(f"Mean CI range: {{ci_gdf['mean_ci'].min():.1f}} to {{ci_gdf['mean_ci'].max():.1f}}")

		# Step 4: Extract centroid lat/lon from each MWS polygon
		ci_gdf['centroid_lon'] = ci_gdf.geometry.centroid.x
		ci_gdf['centroid_lat'] = ci_gdf.geometry.centroid.y
		print(f"Centroid lon range: {{ci_gdf['centroid_lon'].min():.4f}} to {{ci_gdf['centroid_lon'].max():.4f}}")
		print(f"Centroid lat range: {{ci_gdf['centroid_lat'].min():.4f}} to {{ci_gdf['centroid_lat'].max():.4f}}")

		# Step 5: Initialize Earth Engine and build ROI bounding box from CI data extent
		import ee
		ee.Initialize(project='corestack-gee')

		minx, miny, maxx, maxy = ci_gdf.total_bounds
		ee_roi = ee.Geometry.Rectangle([float(minx), float(miny), float(maxx), float(maxy)])
		print(f"GEE ROI bounding box: [{{minx:.4f}}, {{miny:.4f}}, {{maxx:.4f}}, {{maxy:.4f}}]")

		# Step 6: Cloud masking function for Landsat 8 C2L2
		def cloud_mask_l8(image):
			qa = image.select('QA_PIXEL')
			cloud = qa.bitwiseAnd(1 << 3).eq(0)
			shadow = qa.bitwiseAnd(1 << 4).eq(0)
			return image.updateMask(cloud.And(shadow))

		# Step 7: LST computation function (ST_B10 -> Celsius)
		def compute_lst(image):
			lst = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
			return lst.rename('LST').copyProperties(image, ['system:time_start'])

		# Step 8: Build a SINGLE mean monsoon LST composite across ALL years (2017-2023)
		all_monsoon_images = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
			.filterBounds(ee_roi) \
			.filter(ee.Filter.calendarRange(6, 9, 'month')) \
			.filterDate('2017-01-01', '2023-12-31') \
			.map(cloud_mask_l8) \
			.map(compute_lst)

		scene_count = all_monsoon_images.size().getInfo()
		print(f"Total Landsat 8 monsoon scenes (2017-2023): {{scene_count}}")

		# Single mean composite across all monsoon scenes
		monsoon_lst_mean = all_monsoon_images.mean()

		# Step 9: Build ee.FeatureCollection of CENTROID POINTS (not polygons!)
		points = []
		for idx, row in ci_gdf.iterrows():
			lon = float(row['centroid_lon'])
			lat = float(row['centroid_lat'])
			pt = ee.Geometry.Point([lon, lat])
			points.append(ee.Feature(pt, {{'uid': str(row['uid'])}}))
		fc_points = ee.FeatureCollection(points)
		print(f"Created {{len(points)}} centroid points for GEE sampling")

		# Step 10: Sample the mean LST image at all centroid points (SINGLE GEE call)
		sampled = monsoon_lst_mean.sampleRegions(
			collection=fc_points,
			scale=30,
			geometries=False
		)
		sampled_info = sampled.getInfo()
		result_features = sampled_info['features']
		print(f"GEE returned {{len(result_features)}} sampled points")

		# Step 11: Parse GEE results into DataFrame
		import numpy as np
		lst_records = []
		for feat in result_features:
			props = feat['properties']
			uid = props.get('uid')
			lst_val = props.get('LST')
			if uid is not None and lst_val is not None:
				if not np.isnan(lst_val):
					lst_records.append({{'uid': str(uid), 'mean_monsoon_lst': float(lst_val)}})

		lst_df = pd.DataFrame(lst_records)
		print(f"Valid LST values extracted: {{len(lst_df)}}")
		if len(lst_df) > 0:
			print(f"LST range: {{lst_df['mean_monsoon_lst'].min():.2f}} to {{lst_df['mean_monsoon_lst'].max():.2f}} °C")

		# Step 12: Merge LST with CI on uid
		ci_gdf['uid'] = ci_gdf['uid'].astype(str)
		merged = ci_gdf[['uid', 'mean_ci', 'geometry']].merge(lst_df, on='uid', how='inner')
		merged = merged.dropna(subset=['mean_ci', 'mean_monsoon_lst'])
		print(f"Merged MWS count: {{len(merged)}}")

		# Step 13: Build scatter plot
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		from scipy import stats

		fig, ax = plt.subplots(figsize=(14, 10))
		ax.scatter(merged['mean_monsoon_lst'], merged['mean_ci'],
				s=70, alpha=0.7, edgecolors='black', linewidth=0.5, c='darkorange')

		# Annotate each point with MWS uid
		for _, row in merged.iterrows():
			ax.annotate(row['uid'], (row['mean_monsoon_lst'], row['mean_ci']),
					fontsize=5, alpha=0.7, ha='left', va='bottom',
					xytext=(3, 3), textcoords='offset points')

		# Add trend line
		slope, intercept, r_value, p_value, std_err = stats.linregress(
			merged['mean_monsoon_lst'], merged['mean_ci'])
		x_line = np.linspace(merged['mean_monsoon_lst'].min(), merged['mean_monsoon_lst'].max(), 100)
		ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=1.5,
				label=f'Trend (r={{r_value:.3f}}, p={{p_value:.3f}})')

		ax.set_xlabel('Mean Monsoon Land Surface Temperature (°C)', fontsize=12)
		ax.set_ylabel('Mean Cropping Intensity (%)', fontsize=12)
		ax.set_title(f'Monsoon LST vs Cropping Intensity per Microwatershed\n{{LOCATION}} (2017–2023) | Pearson r={{r_value:.3f}}', fontsize=13)
		ax.legend(fontsize=10)
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig('./exports/lst_vs_ci_scatter.png', dpi=200, bbox_inches='tight')
		print(f"Scatter plot saved to ./exports/lst_vs_ci_scatter.png")

		# Step 14: Export merged data as GeoJSON
		export_gdf = gpd.GeoDataFrame(merged, geometry='geometry')
		export_gdf.to_file('./exports/lst_vs_cropping_intensity.geojson', driver='GeoJSON')
		print(f"GeoJSON exported to ./exports/lst_vs_cropping_intensity.geojson")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 12: PHENOLOGICAL STAGE DETECTION & SIMILARITY      ║
		# ║ ⚠️ For Query Type 14, COPY THIS CODE ALMOST VERBATIM.       ║
		# ║ Uses Sentinel-2 NDVI from GEE + MWS boundaries from CS    ║
		# ║ Centroid-based monthly NDVI → phenological classification  ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, json
		os.makedirs('./exports', exist_ok=True)

		# Step 1: Fetch data from CoreStack
		# ═══ PARAMETERS — Extract these from the user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state
		result = fetch_corestack_data(query=f"{{LOCATION}} admin boundary cropping intensity")
		data = json.loads(result)
		vector_layers = data['spatial_data']['vector_layers']

		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 2: Load MWS boundary layer — use Cropping Intensity (has 'uid' column for MWS identification)
		import geopandas as gpd
		import pandas as pd
		import numpy as np
		mws_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'cropping' in lname and 'intensity' in lname and mws_gdf is None:
				print(f"Found Cropping Intensity layer: {{layer['layer_name']}}")
				mws_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')

		print(f"MWS shape: {{mws_gdf.shape}}, columns: {{mws_gdf.columns.tolist()}}")

		# Step 3: Verify target MWS exists
		# ═══ PARAMETERS — Extract from user's query ═══
		TARGET_UID = '18_16157'  # ← Replace with the MWS uid from the user's query
		YEARS = [2019, 2020]  # ← Replace with year range from the user's query
		mws_gdf['uid'] = mws_gdf['uid'].astype(str)
		assert TARGET_UID in mws_gdf['uid'].values, f"Target MWS {{TARGET_UID}} not found!"
		print(f"Target MWS {{TARGET_UID}} found in dataset with {{len(mws_gdf)}} total MWS")

		# Step 4: Extract centroid lat/lon from each MWS polygon
		mws_gdf['centroid_lon'] = mws_gdf.geometry.centroid.x
		mws_gdf['centroid_lat'] = mws_gdf.geometry.centroid.y
		print(f"Centroid lon range: {{mws_gdf['centroid_lon'].min():.4f}} to {{mws_gdf['centroid_lon'].max():.4f}}")
		print(f"Centroid lat range: {{mws_gdf['centroid_lat'].min():.4f}} to {{mws_gdf['centroid_lat'].max():.4f}}")

		# Step 5: Initialize Earth Engine and build ROI bounding box
		import ee
		ee.Initialize(project='corestack-gee')
		minx, miny, maxx, maxy = mws_gdf.total_bounds
		ee_roi = ee.Geometry.Rectangle([float(minx), float(miny), float(maxx), float(maxy)])
		print(f"GEE ROI bounding box: [{{minx:.4f}}, {{miny:.4f}}, {{maxx:.4f}}, {{maxy:.4f}}]")

		# Step 6: Cloud masking and NDVI computation
		# ⚠️ DO NOT use .map() with custom Python functions — it fails in the smolagents executor
		# ⚠️ ("A mapped function's arguments cannot be used in client-side operations")
		# Instead: pre-filter by cloud percentage, take median composite, then compute NDVI directly

		# Step 7: Build monthly NDVI composites and sample EACH MONTH INDEPENDENTLY
		# ⚠️ DO NOT stack all bands into one image — sampleRegions drops features where ANY band is null!
		# Instead, sample each monthly single-band image separately to preserve all centroids.
		YEARS = [2019, 2020]

		# Step 9: Build ee.FeatureCollection of centroid points FIRST
		points = []
		for idx, row in mws_gdf.iterrows():
			pt = ee.Geometry.Point([float(row['centroid_lon']), float(row['centroid_lat'])])
			points.append(ee.Feature(pt, {{'uid': str(row['uid'])}}))
		fc_points = ee.FeatureCollection(points)
		print(f"Created {{len(points)}} centroid points for GEE sampling")

		# Step 9: For each month, compute median NDVI and sample at centroids independently
		# ⚠️ Skip months 7 (July) and 8 (August) — Indian monsoon means zero cloud-free Sentinel-2 images
		MONTHS_TO_SKIP = [7, 8]
		records = []
		for year in YEARS:
			for month in range(1, 13):
				if month in MONTHS_TO_SKIP:
					print(f"  NDVI_{{year}}_{{month:02d}}: Skipped (monsoon — no cloud-free imagery)")
					continue
				start_date = f'{{year}}-{{month:02d}}-01'
				if month == 12:
					end_date = f'{{year + 1}}-01-01'
				else:
					end_date = f'{{year}}-{{month + 1:02d}}-01'

				# Pre-filter by cloud percentage (NO .map() needed)
				collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
					.filterBounds(ee_roi) \
					.filterDate(start_date, end_date) \
					.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

				# Compute NDVI directly on the median composite (no .map() call)
				band_name = f'NDVI_{{year}}_{{month:02d}}'
				composite = collection.median()
				ndvi_img = composite.normalizedDifference(['B8', 'B4']).rename(band_name)

				# Sample THIS month at all centroids (independent call — no null dropout)
				sampled = ndvi_img.sampleRegions(collection=fc_points, scale=10, geometries=False)
				try:
					result_features = sampled.getInfo()['features']
				except Exception as e:
					print(f"  {{band_name}}: GEE error — {{e}}")
					continue

				count = 0
				for feat in result_features:
					props = feat['properties']
					uid = props.get('uid')
					ndvi_val = props.get(band_name)
					if uid is not None and ndvi_val is not None:
						try:
							if not np.isnan(float(ndvi_val)):
								records.append({{'uid': str(uid), 'year': year, 'month': month, 'ndvi': float(ndvi_val)}})
								count += 1
						except (ValueError, TypeError):
							pass
				print(f"  {{band_name}}: {{count}}/{{len(points)}} centroids sampled")

		ndvi_df = pd.DataFrame(records)
		print(f"\nTotal NDVI records: {{len(ndvi_df)}}")
		print(f"MWS with data: {{ndvi_df['uid'].nunique()}}")
		print(f"NDVI range: {{ndvi_df['ndvi'].min():.3f}} to {{ndvi_df['ndvi'].max():.3f}}")

		# Step 12: Compute temporal derivative and classify phenological stages
		ndvi_df = ndvi_df.sort_values(['uid', 'year', 'month']).reset_index(drop=True)
		ndvi_df['ndvi_prev'] = ndvi_df.groupby('uid')['ndvi'].shift(1)
		ndvi_df['delta'] = ndvi_df['ndvi'] - ndvi_df['ndvi_prev']

		def classify_pheno(row):
			ndvi = row['ndvi']
			delta = row['delta'] if not pd.isna(row['delta']) else 0
			if ndvi < 0.15:
				return 'Bare/Fallow'
			elif ndvi < 0.25:
				if delta > 0.03:
					return 'Green-up'
				elif delta < -0.03:
					return 'Senescence'
				else:
					return 'Dormant'
			elif ndvi < 0.45:
				if delta > 0.03:
					return 'Green-up'
				elif delta < -0.03:
					return 'Senescence'
				else:
					return 'Maturity'
			else:
				if delta > 0.03:
					return 'Green-up'
				elif delta < -0.03:
					return 'Senescence'
				else:
					return 'Peak Vegetation'

		ndvi_df['pheno_stage'] = ndvi_df.apply(classify_pheno, axis=1)
		print(f"\nPhenological stage distribution:")
		print(ndvi_df['pheno_stage'].value_counts())

		# Step 13: Get target MWS stages and find matches
		target_stages = ndvi_df[ndvi_df['uid'] == TARGET_UID][['year', 'month', 'pheno_stage']].rename(
			columns={{'pheno_stage': 'target_stage'}})
		print(f"\nTarget MWS {{TARGET_UID}} phenological stages:")
		for _, row in target_stages.iterrows():
			print(f"  {{row['year']}}-{{row['month']:02d}}: {{row['target_stage']}}")

		# Merge to find matches per month
		ndvi_df = ndvi_df.merge(target_stages, on=['year', 'month'], how='left')
		ndvi_df['matches_target'] = ndvi_df['pheno_stage'] == ndvi_df['target_stage']

		# Step 14: Build wide-format output for GeoJSON export
		pivot_stage = ndvi_df.pivot_table(index='uid', columns=['year', 'month'], values='pheno_stage', aggfunc='first')
		pivot_match = ndvi_df.pivot_table(index='uid', columns=['year', 'month'], values='matches_target', aggfunc='first')

		stage_cols = {{}}
		match_cols = {{}}
		for (year_v, month_v) in pivot_stage.columns:
			stage_cols[f'stage_{{year_v}}_{{month_v:02d}}'] = pivot_stage[(year_v, month_v)]
			match_cols[f'match_{{year_v}}_{{month_v:02d}}'] = pivot_match[(year_v, month_v)]

		wide_df = pd.DataFrame(stage_cols)
		wide_df = wide_df.join(pd.DataFrame(match_cols))
		wide_df['uid'] = wide_df.index
		wide_df.reset_index(drop=True, inplace=True)

		# Compute similarity score
		match_columns = [c for c in wide_df.columns if c.startswith('match_')]
		wide_df['months_matching'] = wide_df[match_columns].sum(axis=1).astype(int)
		wide_df['similarity_pct'] = (wide_df['months_matching'] / len(match_columns) * 100).round(1)

		# Merge with geometry
		export_gdf = mws_gdf[['uid', 'geometry']].merge(wide_df, on='uid', how='inner')
		export_gdf = gpd.GeoDataFrame(export_gdf, geometry='geometry')

		print(f"\nMWS with >= 50% phenological similarity to {{TARGET_UID}}:")
		similar = export_gdf[export_gdf['similarity_pct'] >= 50].sort_values('similarity_pct', ascending=False)
		for _, row in similar.iterrows():
			print(f"  {{row['uid']}}: {{row['similarity_pct']}}% match ({{row['months_matching']}}/{{len(match_columns)}} months)")

		# Step 15: Export GeoJSON
		export_gdf.to_file('./exports/phenological_stages.geojson', driver='GeoJSON')
		print(f"\nGeoJSON exported to ./exports/phenological_stages.geojson")

		# Step 16: Heatmap visualization of phenological stages
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		from matplotlib.colors import ListedColormap, BoundaryNorm
		from matplotlib.patches import Patch

		stage_order = ['Bare/Fallow', 'Dormant', 'Green-up', 'Maturity', 'Peak Vegetation', 'Senescence']
		stage_to_num = {{s: i for i, s in enumerate(stage_order)}}
		colors = ['#8B4513', '#D2B48C', '#90EE90', '#FFD700', '#006400', '#FF8C00']

		# Build matrix: rows = top MWS sorted by similarity, cols = months
		sorted_uids = export_gdf.sort_values('similarity_pct', ascending=False)['uid'].tolist()
		if TARGET_UID in sorted_uids:
			sorted_uids.remove(TARGET_UID)
			sorted_uids = [TARGET_UID] + sorted_uids

		stage_col_names = sorted([c for c in wide_df.columns if c.startswith('stage_')])
		matrix = []
		display_uids = []
		for uid in sorted_uids[:30]:  # Top 30 for readability
			uid_row = wide_df[wide_df['uid'] == uid]
			if len(uid_row) == 0:
				continue
			row_data = []
			for col in stage_col_names:
				stage = uid_row.iloc[0][col]
				row_data.append(stage_to_num.get(stage, -1))
			matrix.append(row_data)
			display_uids.append(uid)

		if len(matrix) > 0:
			fig, ax = plt.subplots(figsize=(18, max(8, len(display_uids) * 0.35)))
			cmap = ListedColormap(colors)
			bounds = list(range(len(stage_order) + 1))
			norm = BoundaryNorm(bounds, cmap.N)

			im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
			ax.set_yticks(range(len(display_uids)))
			ax.set_yticklabels(display_uids, fontsize=6)

			month_labels = [c.replace('stage_', '').replace('_', '-') for c in stage_col_names]
			ax.set_xticks(range(len(stage_col_names)))
			ax.set_xticklabels(month_labels, fontsize=7, rotation=45, ha='right')

			ax.set_xlabel('Month', fontsize=11)
			ax.set_ylabel('Microwatershed UID', fontsize=11)
			ax.set_title(f'Phenological Stages per Microwatershed\n{{LOCATION}} ({{YEARS[0]}}-{{YEARS[-1]}})\nTarget: {{TARGET_UID}} (top row)', fontsize=12)

			legend_elements = [Patch(facecolor=colors[i], label=stage_order[i]) for i in range(len(stage_order))]
			ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, title='Stage')

			plt.tight_layout()
			plt.savefig('./exports/phenological_stages_heatmap.png', dpi=200, bbox_inches='tight')
			print(f"Heatmap saved to ./exports/phenological_stages_heatmap.png")

		final_answer(f"Phenological stage analysis complete for {{len(export_gdf)}} microwatersheds.\nTarget MWS: {{TARGET_UID}}\nMWS with >=50% similarity: {{len(similar)}}\nExports:\n- Vector: ./exports/phenological_stages.geojson\n- Heatmap: ./exports/phenological_stages_heatmap.png")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 13: RUNOFF ACCUMULATION per PHENOLOGICAL STAGE     ║
		# ║            vs CROPPING INTENSITY — SCATTER PLOT            ║
		# ║ ⚠️ For Query Type 15, COPY THIS CODE ALMOST VERBATIM.       ║
		# ║ Loads Query 14 phenological output + Drought + CI vectors  ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, json, re
		os.makedirs('./exports', exist_ok=True)

		# Step 1: Load phenological stage GeoJSON from Query 14 output
		import geopandas as gpd
		import pandas as pd
		import numpy as np
		pheno_gdf = gpd.read_file('./exports/phenological_stages.geojson')
		pheno_gdf['uid'] = pheno_gdf['uid'].astype(str)
		print(f"Loaded phenological GeoJSON: {{pheno_gdf.shape[0]}} MWS, {{pheno_gdf.shape[1]}} columns")

		# Extract stage columns → long format: uid, year, month, stage
		stage_cols = sorted([c for c in pheno_gdf.columns if c.startswith('stage_')])
		print(f"Stage columns found: {{len(stage_cols)}}")
		pheno_records = []
		for _, row in pheno_gdf.iterrows():
			uid = row['uid']
			for sc in stage_cols:
				# Parse stage_YYYY_MM
				parts = sc.replace('stage_', '').split('_')
				year_val = int(parts[0])
				month_val = int(parts[1])
				stage = row[sc]
				if pd.notna(stage) and isinstance(stage, str) and stage != 'Unknown':
					pheno_records.append({{'uid': uid, 'year': year_val, 'month': month_val, 'stage': stage}})
		pheno_df = pd.DataFrame(pheno_records)
		print(f"Phenological records: {{len(pheno_df)}}")
		print(f"Unique stages: {{pheno_df['stage'].value_counts().to_dict()}}")

		# Step 2: Fetch Drought + Cropping Intensity from CoreStack
		# ═══ PARAMETER — Extract from user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state
		result = fetch_corestack_data(query=f"{{LOCATION}} cropping intensity drought")
		data = json.loads(result)
		vector_layers = data['spatial_data']['vector_layers']

		print("\nAvailable vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 3: Load both layers
		ci_gdf = None
		drought_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'intensity' in lname and ci_gdf is None:
				print(f"\nFound Cropping Intensity layer: {{layer['layer_name']}}")
				ci_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)
			if 'drought' in lname and 'causal' not in lname and drought_gdf is None:
				print(f"Found Drought layer: {{layer['layer_name']}}")
				drought_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True)

		ci_gdf['uid'] = ci_gdf['uid'].astype(str)
		drought_gdf['uid'] = drought_gdf['uid'].astype(str)
		print(f"CI shape: {{ci_gdf.shape}}")
		print(f"Drought shape: {{drought_gdf.shape}}")

		# Step 4: Parse rd columns → monthly runoff per MWS
		# rd column format: rdYY-M-D → year = 2000+YY, month = M
		rd_cols = [c for c in drought_gdf.columns if c.startswith('rd') and '-' in c]
		print(f"\nTotal rd columns: {{len(rd_cols)}}")

		# Map each rd column to (year, month)
		rd_to_ym = {{}}
		for c in rd_cols:
			try:
				parts = c[2:].split('-')  # "19-5-21" → ['19', '5', '21']
				yr = 2000 + int(parts[0])
				mo = int(parts[1])
				if (yr, mo) not in rd_to_ym:
					rd_to_ym[(yr, mo)] = []
				rd_to_ym[(yr, mo)].append(c)
			except (ValueError, IndexError):
				pass

		print(f"rd columns mapped to {{len(rd_to_ym)}} (year, month) groups")
		for (yr, mo), cols in sorted(rd_to_ym.items()):
			print(f"  {{yr}}-{{mo:02d}}: {{len(cols)}} weekly columns")

		# Step 5: Compute monthly runoff (sum of rd values) per MWS per (year, month)
		runoff_records = []
		for (yr, mo), cols in sorted(rd_to_ym.items()):
			monthly_sum = drought_gdf[cols].sum(axis=1)
			for idx, row in drought_gdf.iterrows():
				uid = row['uid']
				area_ha = float(row.get('area_in_ha', 1))
				runoff_mm = float(monthly_sum.iloc[idx] if hasattr(idx, '__index__') else monthly_sum.loc[idx])
				runoff_records.append({{'uid': uid, 'year': yr, 'month': mo, 'runoff_mm': runoff_mm, 'area_in_ha': area_ha}})

		runoff_df = pd.DataFrame(runoff_records)
		print(f"\nMonthly runoff records: {{len(runoff_df)}}")

		# Step 6: Merge phenological stages with runoff
		merged = pheno_df.merge(runoff_df, on=['uid', 'year', 'month'], how='inner')
		print(f"Merged pheno+runoff records: {{len(merged)}}")

		if len(merged) == 0:
			# Fallback: rd data only covers some years. Check overlap.
			rd_years = set(runoff_df['year'].unique())
			pheno_years = set(pheno_df['year'].unique())
			print(f"rd data years: {{sorted(rd_years)}}")
			print(f"Phenological years: {{sorted(pheno_years)}}")
			print(f"Overlap years: {{sorted(rd_years & pheno_years)}}")

		# Step 7: Accumulate runoff per (uid, year, stage)
		# Sum runoff_mm across all months belonging to same stage for each MWS in each year
		stage_runoff = merged.groupby(['uid', 'year', 'stage']).agg(
			runoff_accum_mm=('runoff_mm', 'sum'),
			area_in_ha=('area_in_ha', 'first'),
			n_months=('month', 'count')
		).reset_index()
		print(f"\nStage-level runoff accumulation: {{len(stage_runoff)}} records")
		print(f"Runoff range: {{stage_runoff['runoff_accum_mm'].min():.1f}} to {{stage_runoff['runoff_accum_mm'].max():.1f}} mm")

		# Step 8: Get cropping intensity per MWS per year
		ci_years = sorted(set(stage_runoff['year'].unique()))
		ci_long_records = []
		for yr in ci_years:
			ci_col = f'cropping_intensity_{{yr}}'
			if ci_col in ci_gdf.columns:
				for _, row in ci_gdf.iterrows():
					ci_long_records.append({{'uid': str(row['uid']), 'year': yr, 'ci_pct': float(row[ci_col])}})
		ci_long_df = pd.DataFrame(ci_long_records)
		print(f"CI records for years {{ci_years}}: {{len(ci_long_df)}}")

		# Step 9: Merge runoff accumulation with CI
		scatter_df = stage_runoff.merge(ci_long_df, on=['uid', 'year'], how='inner')
		print(f"Final scatter data: {{len(scatter_df)}} points")
		print(f"Stages represented: {{scatter_df['stage'].value_counts().to_dict()}}")

		# Step 10: Build scatterplot — Runoff Accumulation vs CI, colored by phenological stage
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

		stage_colors = {{
			'Bare/Fallow': '#8B4513',
			'Dormant': '#D2B48C',
			'Green-up': '#90EE90',
			'Maturity': '#FFD700',
			'Peak Vegetation': '#006400',
			'Senescence': '#FF8C00'
		}}

		fig, ax = plt.subplots(figsize=(14, 10))
		for stage, color in stage_colors.items():
			subset = scatter_df[scatter_df['stage'] == stage]
			if len(subset) > 0:
				ax.scatter(subset['runoff_accum_mm'], subset['ci_pct'],
						c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.3,
						label=f'{{stage}} (n={{len(subset)}})')

		ax.set_xlabel('Runoff Accumulation (mm) during Phenological Stage', fontsize=12)
		ax.set_ylabel('Cropping Intensity (%)', fontsize=12)
		ax.set_title(f'Runoff Accumulation per Phenological Stage vs Cropping Intensity\n{{LOCATION}}', fontsize=13)
		ax.legend(fontsize=9, loc='upper left', title='Phenological Stage')
		ax.grid(True, alpha=0.3)

		# Add trend line across all points
		from numpy.polynomial.polynomial import polyfit
		valid = scatter_df.dropna(subset=['runoff_accum_mm', 'ci_pct'])
		if len(valid) > 2:
			coeffs = np.polyfit(valid['runoff_accum_mm'], valid['ci_pct'], 1)
			x_line = np.linspace(valid['runoff_accum_mm'].min(), valid['runoff_accum_mm'].max(), 100)
			y_line = np.polyval(coeffs, x_line)
			r = np.corrcoef(valid['runoff_accum_mm'], valid['ci_pct'])[0, 1]
			ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1.5, label=f'Trend (r={{r:.3f}})')
			ax.legend(fontsize=9, loc='upper left', title='Phenological Stage')

		plt.tight_layout()
		plt.savefig('./exports/runoff_vs_ci_by_phenostage.png', dpi=200, bbox_inches='tight')
		print(f"\nScatter plot saved to ./exports/runoff_vs_ci_by_phenostage.png")

		# Step 11: Export GeoJSON
		# Pivot stage_runoff to wide format per MWS
		export_base = scatter_df.groupby('uid').agg(
			mean_runoff_accum=('runoff_accum_mm', 'mean'),
			total_runoff_accum=('runoff_accum_mm', 'sum'),
			mean_ci=('ci_pct', 'mean'),
			n_stages=('stage', 'nunique')
		).reset_index()
		export_gdf = pheno_gdf[['uid', 'geometry']].merge(export_base, on='uid', how='inner')
		export_gdf = gpd.GeoDataFrame(export_gdf, geometry='geometry')
		export_gdf.to_file('./exports/runoff_vs_ci_by_phenostage.geojson', driver='GeoJSON')
		print(f"GeoJSON exported to ./exports/runoff_vs_ci_by_phenostage.geojson")

		final_answer(f"Scatter plot of runoff accumulation per phenological stage vs cropping intensity complete.\n{{len(scatter_df)}} data points across {{scatter_df['stage'].nunique()}} phenological stages for {{scatter_df['uid'].nunique()}} MWS.\nExports:\n- Scatter: ./exports/runoff_vs_ci_by_phenostage.png\n- Vector: ./exports/runoff_vs_ci_by_phenostage.geojson")

		# ╔══════════════════════════════════════════════════════════════╗
		# ║ EXAMPLE 14: HYPOTHESIS TEST — LST vs CROPPING INTENSITY    ║
		# ║ ⚠️ For Query Type 16, COPY THIS CODE ALMOST VERBATIM.       ║
		# ║ Fetches CI from CoreStack, LST from GEE Landsat 8          ║
		# ║ Pearson r + two-sample t-test (Hot vs Cool MWS groups)     ║
		# ╚══════════════════════════════════════════════════════════════╝
		import os, json
		os.makedirs('./exports', exist_ok=True)

		# ═══ PARAMETER — Extract from user's query ═══
		LOCATION = "Navalgund Dharwad Karnataka"  # ← Replace with user's tehsil district state

		# Step 1: Fetch data from CoreStack
		result = fetch_corestack_data(query=f"{{LOCATION}} cropping intensity")
		data = json.loads(result)
		vector_layers = data['spatial_data']['vector_layers']

		print("Available vector layers:")
		for layer in vector_layers:
			print(f"  - {{layer['layer_name']}}")

		# Step 2: Load Cropping Intensity layer
		import geopandas as gpd
		import pandas as pd
		import numpy as np
		ci_gdf = None
		for layer in vector_layers:
			lname = layer['layer_name'].lower()
			if 'cropping' in lname and 'intensity' in lname and ci_gdf is None:
				print(f"Found Cropping Intensity layer: {{layer['layer_name']}}")
				ci_gdf = pd.concat([gpd.read_file(u['url']) for u in layer['urls']], ignore_index=True).to_crs('EPSG:4326')

		print(f"CI shape: {{ci_gdf.shape}}, columns: {{ci_gdf.columns.tolist()}}")

		# Step 3: Compute mean cropping intensity per MWS (2017-2023)
		ci_cols = [c for c in ci_gdf.columns if c.startswith('cropping_intensity_')]
		ci_cols_filtered = [c for c in ci_cols if int(c.split('_')[-1]) >= 2017 and int(c.split('_')[-1]) <= 2023]
		ci_gdf['mean_ci'] = ci_gdf[ci_cols_filtered].mean(axis=1)
		print(f"CI year columns used: {{ci_cols_filtered}}")
		print(f"Mean CI range: {{ci_gdf['mean_ci'].min():.1f}} to {{ci_gdf['mean_ci'].max():.1f}}")

		# Step 4: Extract centroid lat/lon
		ci_gdf['centroid_lon'] = ci_gdf.geometry.centroid.x
		ci_gdf['centroid_lat'] = ci_gdf.geometry.centroid.y

		# Step 5: Initialize Earth Engine and build ROI
		import ee
		ee.Initialize(project='corestack-gee')
		minx, miny, maxx, maxy = ci_gdf.total_bounds
		ee_roi = ee.Geometry.Rectangle([float(minx), float(miny), float(maxx), float(maxy)])
		print(f"GEE ROI: [{{minx:.4f}}, {{miny:.4f}}, {{maxx:.4f}}, {{maxy:.4f}}]")

		# Step 6: Cloud masking + LST functions for Landsat 8 C2L2
		def cloud_mask_l8(image):
			qa = image.select('QA_PIXEL')
			cloud = qa.bitwiseAnd(1 << 3).eq(0)
			shadow = qa.bitwiseAnd(1 << 4).eq(0)
			return image.updateMask(cloud.And(shadow))

		def compute_lst(image):
			lst = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
			return lst.rename('LST').copyProperties(image, ['system:time_start'])

		# Step 7: Build ANNUAL mean LST composite (ALL months, 2017-2023)
		all_images = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
			.filterBounds(ee_roi) \
			.filterDate('2017-01-01', '2023-12-31') \
			.map(cloud_mask_l8) \
			.map(compute_lst)

		scene_count = all_images.size().getInfo()
		print(f"Total Landsat 8 scenes (2017-2023, all months): {{scene_count}}")
		mean_lst = all_images.mean()

		# Step 8: Build centroid points FeatureCollection
		points = []
		for idx, row in ci_gdf.iterrows():
			pt = ee.Geometry.Point([float(row['centroid_lon']), float(row['centroid_lat'])])
			points.append(ee.Feature(pt, {{'uid': str(row['uid'])}}))
		fc_points = ee.FeatureCollection(points)
		print(f"Created {{len(points)}} centroid points for GEE sampling")

		# Step 9: Sample LST at centroids
		sampled = mean_lst.sampleRegions(collection=fc_points, scale=30, geometries=False)
		result_features = sampled.getInfo()['features']
		print(f"GEE returned {{len(result_features)}} sampled points")

		lst_records = []
		for feat in result_features:
			props = feat['properties']
			uid = props.get('uid')
			lst_val = props.get('LST')
			if uid is not None and lst_val is not None:
				if not np.isnan(lst_val):
					lst_records.append({{'uid': str(uid), 'mean_lst': float(lst_val)}})

		lst_df = pd.DataFrame(lst_records)
		print(f"Valid LST values: {{len(lst_df)}}")
		print(f"LST range: {{lst_df['mean_lst'].min():.2f}} to {{lst_df['mean_lst'].max():.2f}} °C")

		# Step 10: Merge LST with CI
		ci_gdf['uid'] = ci_gdf['uid'].astype(str)
		merged = ci_gdf[['uid', 'mean_ci', 'geometry']].merge(lst_df, on='uid', how='inner')
		merged = merged.dropna(subset=['mean_ci', 'mean_lst'])
		print(f"Merged MWS count: {{len(merged)}}")

		# Step 11: Hypothesis testing
		from scipy import stats as sp_stats

		# 11a: Pearson correlation
		r_val, p_val = sp_stats.pearsonr(merged['mean_lst'], merged['mean_ci'])
		print(f"\n═══ HYPOTHESIS TEST RESULTS ═══")
		print(f"H0: No linear relationship between LST and Cropping Intensity")
		print(f"H1: Higher average temperature → higher cropping intensity")
		print(f"\nPearson r = {{r_val:.4f}}, p-value = {{p_val:.6f}}")
		if p_val < 0.05:
			print(f"→ Reject H0 at α=0.05: significant linear relationship (r={{r_val:.3f}})")
		else:
			print(f"→ Fail to reject H0 at α=0.05: no significant linear relationship")

		# 11b: Split into Hot / Cool groups at median LST
		lst_median = merged['mean_lst'].median()
		merged['lst_group'] = merged['mean_lst'].apply(lambda x: 'Hot' if x >= lst_median else 'Cool')
		hot_ci = merged[merged['lst_group'] == 'Hot']['mean_ci']
		cool_ci = merged[merged['lst_group'] == 'Cool']['mean_ci']
		print(f"\nMedian LST threshold: {{lst_median:.2f}} °C")
		print(f"Hot group (≥ median): n={{len(hot_ci)}}, mean CI={{hot_ci.mean():.1f}}%")
		print(f"Cool group (< median): n={{len(cool_ci)}}, mean CI={{cool_ci.mean():.1f}}%")

		# 11c: Independent two-sample t-test
		t_stat, t_pval = sp_stats.ttest_ind(hot_ci, cool_ci, equal_var=False)
		print(f"\nWelch's t-test: t={{t_stat:.4f}}, p={{t_pval:.6f}}")
		if t_pval < 0.05:
			print(f"→ Reject H0 at α=0.05: Hot and Cool groups have significantly different CI")
		else:
			print(f"→ Fail to reject H0 at α=0.05: no significant difference in CI between groups")

		# 11d: Effect size (Cohen's d)
		pooled_std = np.sqrt((hot_ci.std()**2 + cool_ci.std()**2) / 2)
		if pooled_std > 0:
			cohens_d = (hot_ci.mean() - cool_ci.mean()) / pooled_std
			print(f"Cohen's d = {{cohens_d:.3f}} ({{'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'}} effect)")
		else:
			cohens_d = 0.0
			print(f"Cohen's d = 0 (no variance)")

		# Step 12: Build scatter plot
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(figsize=(14, 10))

		# Color by group
		for grp, color, marker in [('Hot', 'orangered', 'o'), ('Cool', 'steelblue', 's')]:
			subset = merged[merged['lst_group'] == grp]
			ax.scatter(subset['mean_lst'], subset['mean_ci'], c=color, s=70, alpha=0.7,
					edgecolors='black', linewidth=0.5, marker=marker,
					label=f'{{grp}} (n={{len(subset)}}, mean CI={{subset["mean_ci"].mean():.1f}}%)')

		# Annotate each point with uid
		for _, row in merged.iterrows():
			ax.annotate(row['uid'], (row['mean_lst'], row['mean_ci']),
					fontsize=5, alpha=0.6, ha='left', va='bottom',
					xytext=(3, 3), textcoords='offset points')

		# Regression line
		slope, intercept, _, _, _ = sp_stats.linregress(merged['mean_lst'], merged['mean_ci'])
		x_line = np.linspace(merged['mean_lst'].min(), merged['mean_lst'].max(), 100)
		ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5,
				label=f'Regression (r={{r_val:.3f}}, p={{p_val:.4f}})')

		# Median LST threshold line
		ax.axvline(lst_median, color='gray', linestyle=':', alpha=0.7, linewidth=1.2,
				label=f'Median LST = {{lst_median:.1f}} °C')

		# Annotations box with hypothesis test results
		textstr = (f'Pearson r = {{r_val:.3f}} (p={{p_val:.4f}})\n'
				   f"Welch t = {{t_stat:.2f}} (p={{t_pval:.4f}})\n"
				   f"Cohen's d = {{cohens_d:.3f}}\n"
				   f"Hot mean CI = {{hot_ci.mean():.1f}}%\n"
				   f"Cool mean CI = {{cool_ci.mean():.1f}}%")
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
		ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
				verticalalignment='bottom', horizontalalignment='right', bbox=props)

		ax.set_xlabel('Mean Land Surface Temperature (°C)', fontsize=12)
		ax.set_ylabel('Mean Cropping Intensity (%)', fontsize=12)
		ax.set_title(f'Hypothesis Test: Higher LST → Higher Cropping Intensity?\n{{LOCATION}} (2017–2023) | Pearson r={{r_val:.3f}}, p={{p_val:.4f}}', fontsize=13)
		ax.legend(fontsize=9, loc='upper left')
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig('./exports/lst_ci_hypothesis_test.png', dpi=200, bbox_inches='tight')
		print(f"\nScatter plot saved to ./exports/lst_ci_hypothesis_test.png")

		# Step 13: Export GeoJSON
		export_gdf = gpd.GeoDataFrame(merged, geometry='geometry')
		export_gdf.to_file('./exports/lst_ci_hypothesis_test.geojson', driver='GeoJSON')
		print(f"GeoJSON exported to ./exports/lst_ci_hypothesis_test.geojson")

		# Step 14: Build result string and return
		pearson_sig = '(significant)' if p_val < 0.05 else '(not significant)'
		ttest_sig = '(significant)' if t_pval < 0.05 else '(not significant)'
		hot_mean = hot_ci.mean()
		cool_mean = cool_ci.mean()
		result_text = (
			f"Hypothesis test complete for {{len(merged)}} microwatersheds in {{LOCATION}}.\n\n"
			f"Results:\n"
			f"- Pearson r = {{r_val:.4f}}, p = {{p_val:.6f}} {{pearson_sig}}\n"
			f"- Welch t-test: t = {{t_stat:.4f}}, p = {{t_pval:.6f}} {{ttest_sig}}\n"
			f"- Hot group (LST >= {{lst_median:.1f}} C): mean CI = {{hot_mean:.1f}}%\n"
			f"- Cool group (LST < {{lst_median:.1f}} C): mean CI = {{cool_mean:.1f}}%\n"
			f"- Cohen's d = {{cohens_d:.3f}}\n\n"
			f"Exports:\n"
			f"- Scatter: ./exports/lst_ci_hypothesis_test.png\n"
			f"- Vector: ./exports/lst_ci_hypothesis_test.geojson"
		)
		final_answer(result_text)

	# ═══════════════════════════════════════════════════════════════════════
	# ║  EXAMPLE 15: AGRICULTURAL SUITABILITY INDEX — RANK MWS             ║
	# ║  Data: Cropping Intensity (CoreStack) + Landsat 8 LST (GEE)        ║
	# ║        + Surface Water Bodies kharif season (CoreStack)             ║
	# ║  ⚠️ For Query Type 17, COPY THIS CODE ALMOST VERBATIM.             ║
	# ═══════════════════════════════════════════════════════════════════════
	# --- Example query: "Rank microwatersheds in Navalgund, Dharwad, Karnataka
	#     by suitability index considering temperature, cropping intensity,
	#     and water availability during the growing phenological stage."

	import json, os, re, math
	import geopandas as gpd
	import pandas as pd
	import numpy as np
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.colors as mcolors

	LOCATION = "Navalgund Dharwad Karnataka"   # ← Replace with user's tehsil district state

	os.makedirs('./exports', exist_ok=True)

	# ── Step 1: Fetch all CoreStack layers ──────────────────────────────
	result = fetch_corestack_data(f"{{LOCATION}} cropping intensity surface water")
	data = json.loads(result)
	vector_layers = data['spatial_data']['vector_layers']
	for vl in vector_layers:
		print(f"  - {{vl['layer_name']}}")

	# ── Step 2: Load Cropping Intensity ─────────────────────────────────
	ci_layer = None
	for vl in vector_layers:
		if 'cropping' in vl['layer_name'].lower() and 'intensity' in vl['layer_name'].lower():
			ci_layer = vl
			break
	print(f"CI layer: {{ci_layer['layer_name']}}")
	ci_gdf = gpd.read_file(ci_layer['urls'][0]['url'])
	ci_gdf = ci_gdf.to_crs(epsg=4326)
	ci_year_cols = sorted([c for c in ci_gdf.columns if 'cropping_intensity_' in c and re.search(r'\\d{{4}}', c)])
	print(f"CI year columns: {{ci_year_cols}}")
	ci_gdf['mean_ci'] = ci_gdf[ci_year_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
	print(f"Mean CI range: {{ci_gdf['mean_ci'].min():.2f}} to {{ci_gdf['mean_ci'].max():.2f}}")

	# ── Step 3: Load Surface Water Bodies (kharif season) ───────────────
	sw_layer = None
	for vl in vector_layers:
		nm = vl['layer_name'].lower()
		if 'surface water' in nm and 'zoi' not in nm:
			sw_layer = vl
			break
	print(f"SW layer: {{sw_layer['layer_name']}}")
	sw_gdf = gpd.read_file(sw_layer['urls'][0]['url'])
	sw_gdf = sw_gdf.to_crs(epsg=4326)
	print(f"SW columns: {{list(sw_gdf.columns)}}")
	print(f"SW shape: {{sw_gdf.shape}}")

	# Identify kharif-only water columns: start with 'k_' but NOT 'kr_' or 'krz_'
	kharif_cols = [c for c in sw_gdf.columns if re.match(r'^k_\\d', c)]
	print(f"Kharif water columns: {{kharif_cols}}")
	for kc in kharif_cols:
		sw_gdf[kc] = pd.to_numeric(sw_gdf[kc], errors='coerce').fillna(0)

	# Aggregate by MWS_UID
	sw_agg = sw_gdf.groupby('MWS_UID')[kharif_cols].sum().reset_index()
	sw_agg['mean_kharif_sw'] = sw_agg[kharif_cols].mean(axis=1)
	print(f"SW aggregated MWS count: {{len(sw_agg)}}")
	print(f"Mean kharif SW range: {{sw_agg['mean_kharif_sw'].min():.2f}} to {{sw_agg['mean_kharif_sw'].max():.2f}}")

	# ── Step 4: Extract centroids for GEE LST sampling ──────────────────
	ci_gdf['centroid_lon'] = ci_gdf.geometry.centroid.x
	ci_gdf['centroid_lat'] = ci_gdf.geometry.centroid.y

	# ── Step 5: GEE — Landsat 8 mean LST (2017-2023, all months) ────────
	import ee
	ee.Initialize(project='corestack-gee')

	minx, miny, maxx, maxy = ci_gdf.total_bounds
	ee_roi = ee.Geometry.Rectangle([float(minx), float(miny), float(maxx), float(maxy)])
	print(f"GEE ROI: [{{minx:.4f}}, {{miny:.4f}}, {{maxx:.4f}}, {{maxy:.4f}}]")

	l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
		  .filterBounds(ee_roi)
		  .filterDate('2017-01-01', '2024-01-01')
		  .filter(ee.Filter.lt('CLOUD_COVER', 30)))
	print(f"Total Landsat 8 scenes: {{l8.size().getInfo()}}")

	def cloud_mask_l8(img):
		qa = img.select('QA_PIXEL')
		mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
		return img.updateMask(mask)

	def compute_lst(img):
		return img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

	lst_composite = l8.map(cloud_mask_l8).map(compute_lst).mean().clip(ee_roi)

	# Build centroid FeatureCollection
	fc_features = []
	for _, row in ci_gdf.iterrows():
		pt = ee.Geometry.Point([float(row['centroid_lon']), float(row['centroid_lat'])])
		fc_features.append(ee.Feature(pt, {{'uid': str(row['uid'])}}))
	ee_fc = ee.FeatureCollection(fc_features)
	print(f"Created {{len(fc_features)}} centroid points")

	sampled = lst_composite.sampleRegions(collection=ee_fc, scale=30, geometries=False).getInfo()
	print(f"GEE returned {{len(sampled['features'])}} sampled points")

	lst_records = []
	for feat in sampled['features']:
		p = feat['properties']
		if 'LST' in p and p['LST'] is not None:
			lst_records.append({{'uid': p['uid'], 'mean_lst': p['LST']}})
	lst_df = pd.DataFrame(lst_records)
	print(f"Valid LST values: {{len(lst_df)}}")
	print(f"LST range: {{lst_df['mean_lst'].min():.2f}} to {{lst_df['mean_lst'].max():.2f}} °C")

	# ── Step 6: Merge all three datasets ─────────────────────────────────
	merged = ci_gdf[['uid', 'mean_ci', 'geometry']].merge(lst_df, on='uid', how='inner')
	merged['uid_str'] = merged['uid'].astype(str)
	sw_agg['MWS_UID_str'] = sw_agg['MWS_UID'].astype(str)
	merged = merged.merge(sw_agg[['MWS_UID', 'mean_kharif_sw']], left_on='uid', right_on='MWS_UID', how='left')
	merged['mean_kharif_sw'] = merged['mean_kharif_sw'].fillna(0)
	print(f"Merged MWS count: {{len(merged)}}")

	# ── Step 7: Min-max normalisation ────────────────────────────────────
	def minmax(s):
		return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else pd.Series(0.5, index=s.index)

	merged['ci_norm'] = minmax(merged['mean_ci'])
	merged['lst_norm'] = minmax(merged['mean_lst'])
	merged['sw_norm'] = minmax(merged['mean_kharif_sw'])

	# ── Step 8: Composite Agricultural Suitability Index ─────────────────
	W_CI, W_TEMP, W_WATER = 0.40, 0.30, 0.30
	merged['ASI'] = (W_CI * merged['ci_norm']
					 + W_TEMP * (1 - merged['lst_norm'])
					 + W_WATER * merged['sw_norm'])
	merged['rank'] = merged['ASI'].rank(ascending=False, method='min').astype(int)
	merged = merged.sort_values('rank')
	print(f"\n═══ AGRICULTURAL SUITABILITY INDEX (ASI) — TOP 10 ═══")
	print(f"Formula: ASI = {{W_CI}}×CI_norm + {{W_TEMP}}×(1−LST_norm) + {{W_WATER}}×SW_norm")
	print(f"{{'-'*70}}")
	for _, r in merged.head(10).iterrows():
		print(f"  Rank {{r['rank']:>3}}  |  uid={{r['uid']:<12}}  |  ASI={{r['ASI']:.4f}}  |  CI={{r['mean_ci']:.2f}}  LST={{r['mean_lst']:.1f}}°C  SW={{r['mean_kharif_sw']:.1f}} ha")

	# ── Step 9: Bar chart — top 25 with stacked components ───────────────
	top25 = merged.head(25).copy()
	top25['comp_ci'] = W_CI * top25['ci_norm']
	top25['comp_temp'] = W_TEMP * (1 - top25['lst_norm'])
	top25['comp_water'] = W_WATER * top25['sw_norm']

	fig, ax = plt.subplots(figsize=(12, 10))
	y_pos = range(len(top25))
	bar_labels = [str(uid) for uid in top25['uid']]

	ax.barh(y_pos, top25['comp_ci'], color='#2ecc71', label=f'Cropping Intensity (w={{W_CI}})')
	ax.barh(y_pos, top25['comp_temp'], left=top25['comp_ci'], color='#e67e22',
		   label=f'Thermal Suitability (w={{W_TEMP}})')
	ax.barh(y_pos, top25['comp_water'],
		   left=top25['comp_ci'] + top25['comp_temp'], color='#3498db',
		   label=f'Water Availability (w={{W_WATER}})')

	for i, (_, r) in enumerate(top25.iterrows()):
		ax.text(r['ASI'] + 0.005, i, f"{{r['ASI']:.3f}}", va='center', fontsize=8)

	ax.set_yticks(y_pos)
	ax.set_yticklabels(bar_labels, fontsize=8)
	ax.invert_yaxis()
	ax.set_xlabel('Agricultural Suitability Index (ASI)', fontsize=11)
	ax.set_title(f'Top 25 Microwatersheds by Agricultural Suitability — {{LOCATION}}',
				 fontsize=13, fontweight='bold')
	ax.legend(loc='lower right', fontsize=9)
	ax.set_xlim(0, min(1.0, top25['ASI'].max() + 0.08))
	plt.tight_layout()
	plt.savefig('./exports/agricultural_suitability_index.png', dpi=200)
	plt.close()
	print(f"\nBar chart saved to ./exports/agricultural_suitability_index.png")

	# ── Step 10: Export GeoJSON ───────────────────────────────────────────
	export_cols = ['uid', 'mean_ci', 'mean_lst', 'mean_kharif_sw',
				   'ci_norm', 'lst_norm', 'sw_norm', 'ASI', 'rank', 'geometry']
	export_gdf = gpd.GeoDataFrame(merged[export_cols], geometry='geometry')
	export_gdf.to_file('./exports/agricultural_suitability_index.geojson', driver='GeoJSON')
	print(f"GeoJSON exported to ./exports/agricultural_suitability_index.geojson")

	# ── Step 11: Build final answer ───────────────────────────────────────
	top3 = merged.head(3)
	t1 = top3.iloc[0]
	t2 = top3.iloc[1]
	t3 = top3.iloc[2]
	result_text = (
		f"Agricultural Suitability Index ranking complete for"
		f" {{len(merged)}} microwatersheds in {{LOCATION}}.\n\n"
		f"Formula: ASI = {{W_CI}}*CI_norm + {{W_TEMP}}*(1-LST_norm) + {{W_WATER}}*SW_norm\n\n"
		f"Top 3 ranked MWS:\n"
		f"  1. uid={{t1['uid']}}  ASI={{t1['ASI']:.4f}}  CI={{t1['mean_ci']:.2f}}  LST={{t1['mean_lst']:.1f}} C  SW={{t1['mean_kharif_sw']:.1f}} ha\n"
		f"  2. uid={{t2['uid']}}  ASI={{t2['ASI']:.4f}}  CI={{t2['mean_ci']:.2f}}  LST={{t2['mean_lst']:.1f}} C  SW={{t2['mean_kharif_sw']:.1f}} ha\n"
		f"  3. uid={{t3['uid']}}  ASI={{t3['ASI']:.4f}}  CI={{t3['mean_ci']:.2f}}  LST={{t3['mean_lst']:.1f}} C  SW={{t3['mean_kharif_sw']:.1f}} ha\n\n"
		f"Exports:\n"
		f"- Bar chart: ./exports/agricultural_suitability_index.png\n"
		f"- GeoJSON: ./exports/agricultural_suitability_index.geojson"
	)
	final_answer(result_text)

	elif data['success'] and data['data_type'] == 'timeseries':
		# Access timeseries data
		timeseries = data['timeseries_data']
		# Process timeseries for temporal analysis
	```
3. **EARTH ENGINE (SUPPLEMENTARY)**: ONLY use Earth Engine if CoreStack doesn't have the required data or for non-India queries. When using Earth Engine:
   - Initialize with: `ee.Initialize(project='corestack-gee')`
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
- Similarity analysis: Top-K similar microwatersheds based on multiple attributes (cosine similarity OR propensity score matching)
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

# if __name__ == "__main__":
# 	"""
# 	Example usage of the hybrid agent.

# 	All queries in this run share one Langfuse session so they appear as a
# 	single conversation in the dashboard.  Each run_hybrid_agent() call
# 	creates its own trace within that session.
# 	"""
# 	print("Bot TEST")
# 	print("="*70)

# 	# Create a single session ID to group all traces from this run
# 	_session_id = generate_session_id()
# 	print(f"📊 Langfuse session ID: {_session_id}")

# 	print("Running query #1 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	print("="*70)

# 	try:
# 		result = run_hybrid_agent(
# 			"Could you show how cropping intensity has changed over the years in Navalgund, Dharwad, Karnataka?",
# 			session_id=_session_id,
# 			user_id="dev-test",
# 		)

# 		# ── Example: record user feedback after agent finishes ──
# 		# In production this would come from a UI callback.
# 		# score_trace(name="user_feedback", value=1.0, data_type="NUMERIC",
# 		#             comment="Correct answer")

# 	except Exception:
# 		pass  # error already logged to Langfuse inside run_hybrid_agent

# 	finally:
# 		# Ensure all buffered events reach Langfuse before exit
# 		lf_shutdown()

# 	# print("Running query #2 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("Could you show how surface water availability has changed over the years in Navalgund, Dharwad, Karnataka?")

# 	# print("Running query #3 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("Can you show me areas that have lost tree cover in Navalgund, Dharwad, Karnataka since 2018? also hectares of degraded area?")

# 	# print("Running query #4 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("How much cropland in Navalgund, Dharwad, Karnataka has turned into built up since 2018? can you show me those regions?")

#     # print("Running query #5 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("Which villages in Navalgund, Dharwad Karnataka among the ones available on Core Stack have experienced droughts? ")

# 	# print("Running query #6 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest cropping senstivity to drought")

# 	# print("Running query #7 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("find all microwatersheds in Navalgund tehsil, Dharwad district in Karnataka, with highest surface water availability senstivity to drought")


# 	# print("Running query #8 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("find me microwatersheds in Navalgund tehsil, Dharwad district, Karnataka most similar to 18_16157 uid microwatershed, based on its terrain, drought frequency, LULC and cropping intensity")


# 	# print("Running query #9 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("find me microwatersheds most similar to 18_16157 id microwatershed in Navalgund, Dharwad, Karnataka, based on its terrain, drought frequency, and LULC using propensity score matching")


# 	# print("Running query #10 from CSV (Navalgund, Dharwad, Karnataka - correct coords)...")
# 	# print("="*70)
# 	# run_hybrid_agent("From the top-K earlier identified drought-sensitive and surface-water-sensitive microwatersheds in Navalgund, Dharwad, Karnataka, rank them based on their cropping intensity and surface water availability. Use weighted score: cropping_score = 3*triple_crop + 2*double_crop + 1*(single_kharif + single_non_kharif). Water score = 3*perennial_water + 2*winter_water + 1*monsoon_water. Read past exports from ./exports/ to get the MWS UIDs.")

# 	# print("Running query #11 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("In my Navalgund tehsil of Dharwad, Karnataka, compare the SC/ST% population of villages against the number of NREGA works done in the villages. Build a scatter plot.")

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

# 	# print("Running query #16 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("For my Navalgund tehsil in Dharwad, Karnataka, test the hypothesis that villages with higher average temperature have higher cropping intensity. Compute per-MWS average Land Surface Temperature from Landsat 8 and cropping intensity from CoreStack, build a scatterplot, and perform hypothesis testing (Pearson correlation + t-test on hot vs cool groups).")

# 	# print("Running query #17 from CSV (Navalgund, Dharwad, Karnataka)...")
# 	# print("="*70)
# 	# run_hybrid_agent("For my Navalgund tehsil in Dharwad, Karnataka, rank microwatersheds by a composite Agricultural Suitability Index considering temperature (Landsat 8 LST), cropping intensity (CoreStack CI vector), and surface water availability during the growing phenological stage (kharif season from Surface Water Bodies vector). Use weighted linear combination: ASI = 0.40*CI_norm + 0.30*(1-LST_norm) + 0.30*SW_norm. Export ranked bar chart and GeoJSON.")

