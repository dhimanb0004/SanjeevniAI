"""
gemini_client.py — Sanjeevni Gemini integration layer.

Schema, prompts, API calls, and the deterministic post-processing
(enrich_with_totals) that keeps all farm-specific arithmetic out of the
LLM entirely. get_sanjeevni_report() is the single entry point Streamlit
should call -- it hides the main-vs-fallback routing decision and, unlike
the version tested in Day8_GeminiIntegration.ipynb, now builds its own
report_json internally via integration.py rather than requiring the
caller to pre-assemble it. That requirement only existed because
assemble_farmer_report() used to depend on a separate notebook kernel.

MODEL_NAME is read from an environment variable, same pattern as
GEMINI_API_KEY -- lets you switch between gemini-3-flash-preview and
gemini-3.1-flash-lite (e.g. during a quota outage) without touching code.
"""

import os
import json
import concurrent.futures
from typing import Optional, List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from integration import assemble_farmer_report, is_district_covered
from engine3_serving import predict_soil_candidates



# Client + model config

MODEL_NAME = os.environ.get("SANJEEVNI_GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_TIMEOUT_SECONDS = 90  # hard ceiling -- normal calls take 30-40s

api_key = os.environ.get("GEMINI_API_KEY")
assert api_key, "GEMINI_API_KEY not found in environment -- check setx worked and this process was started fresh"
client = genai.Client(
    api_key=api_key,
    # Documented mechanism for a request timeout -- included as a first line of defense, but NOT relied on alone: there are open, confirmed bugs in google-genai where this setting is silently ignored for some request paths, letting calls hang indefinitely instead of timing out. The ThreadPoolExecutor wrapper below is the real guarantee, since it operates entirely in our own code.
    http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000),  # milliseconds
)


def _call_with_timeout(fn, *args, timeout_seconds=GEMINI_TIMEOUT_SECONDS, **kwargs):
    """Hard, guaranteed timeout at the application level. Runs fn in a
    background thread and gives up waiting after timeout_seconds,
    regardless of whether the SDK/HTTP layer itself would ever time out on
    its own. Note: the background thread isn't forcibly killed (Python
    can't do that safely) -- it may keep running after we stop waiting on
    it, but the app itself is unblocked and the user gets a fast, clear
    error instead of an indefinite hang."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Gemini did not respond within {timeout_seconds} seconds. "
                f"This is usually a transient network issue -- please try again."
            )


def get_interaction_text(interaction) -> str:
    """Defensive parsing -- Interactions API responses have shown two slightly
    different shapes across Google's own docs. Try output_text first, fall
    back to walking the steps array for a text content block."""
    text = getattr(interaction, "output_text", None)
    if text:
        return text
    for step in getattr(interaction, "steps", []):
        if getattr(step, "type", None) == "model_output":
            for block in getattr(step, "content", []):
                if getattr(block, "type", None) == "text":
                    return block.text
    raise ValueError("Could not extract text from interaction response -- inspect interaction object directly")


# Output schema -- Gemini reports RATES only (yield per hectare, price per tonne). Farm-specific totals are computed separately, deterministically, by enrich_with_totals() below -- never by Gemini itself.

class CropRecommendation(BaseModel):
    crop_name: str = Field(description="Name of the recommended crop")
    is_category_aggregate: bool = Field(description="True if this is a category-level label like Cereals/Kharif Pulse/Rabi Pulse/Oilseed/Millet rather than a specific variety")
    category_aggregate_note: Optional[str] = Field(default=None, description="If is_category_aggregate is true: name 1-2 plausible specific crops within this category from general agronomic knowledge, and state plainly this is a category-level signal, not a specific-variety recommendation. Null otherwise.")
    rank_rationale: str = Field(description="Why this crop made the top 3 -- reference yield/area-share/stability score, and the soil re-ranking probability if a soil card was supplied. If this crop is not the highest-revenue-potential crop among the top 3, explicitly explain why it's still the stronger recommendation.")
    seed_variety: str = Field(description="Recommended seed variety suited to this district and season")
    cultivation_practices: str = Field(description="Scientific cultivation practices")
    fertilizer_guidance: str = Field(description="Fertilizer/nutrient correction guidance, accounting for cross-nutrient contribution -- e.g. DAP is 18-46-0, so correcting a P gap with it also adds N")
    irrigation_guidance: str = Field(description="Irrigation guidance using the farmer's stated irrigation level directly")
    predicted_yield_tonnes_per_hectare: float = Field(description="Expected yield in tonnes per hectare. If fallback_mode is false, copy the exact predicted_yield_tonnes_per_hectare value given in revenue_estimate for this crop verbatim -- do not alter, round differently, or re-estimate it. If fallback_mode is true, provide your own realistic estimate from general agronomic knowledge for this crop/region/season.")
    price_per_tonne: float = Field(description="Price in Rs per tonne. If fallback_mode is false, copy the exact price_per_tonne given in revenue_estimate verbatim. If fallback_mode is true, provide a rough general market-price estimate.")
    price_confidence_note: str = Field(description="e.g. 'Based on current MSP', 'Derived-average category price -- lower confidence', or for fallback reports, 'Rough general market estimate -- not MSP-backed, no local data available for this district'")
    climate_extrapolation_risk_disclosed: Optional[str] = Field(default=None, description="If climate_extrapolation_risk was true for this crop, plainly disclose that its ranking rests on climate conditions outside anything the model was ever trained on. Null if not applicable or not ML-backed.")
    footprint_flag_disclosed: Optional[str] = Field(default=None, description="If this crop's historical_area_share is notably lower than another shortlisted crop that scored below it, flag that discrepancy explicitly rather than silently presenting the ranking. Null if not applicable.")
    stale_climate_context_disclosed: Optional[str] = Field(default=None, description="If no_engine2_forecast is true for this crop's shortlist entry, disclose plainly that its score was computed using historical climate data, not Engine 2's forward-looking forecast. Null if not applicable.")
    season_mismatch_disclosed: Optional[str] = Field(default=None, description="If this crop's actual shortlist season differs from the farmer's literally-requested season, clearly state which season this crop is actually grown in locally and that it differs from what the farmer selected -- so they understand why a differently-timed crop appears in their results. Null if the season matches or this isn't applicable (e.g. fallback mode).")


class SanjeevniReport(BaseModel):
    fallback_mode: bool = Field(description="True only if this was generated via the direct-fallback path because the district had no local ML data")
    location_summary: str = Field(description="One-line summary: State, District, Season, Irrigation level")
    top_recommendations: List[CropRecommendation] = Field(description="Top 3 recommended crops, ranked -- all 3 get full depth, not tiered by rank")
    soil_card_used: bool = Field(description="True if the farmer's Soil Health Card values were supplied and used")
    climate_context_note: str = Field(description="Narrative from Engine 2's climate label and anomaly z-score; empty string if not available (fallback mode)")
    limitations_disclosed: List[str] = Field(description="Every applicable limitation, plain language -- never an empty list")
    farmer_facing_summary: str = Field(description="A warm, plain-language paragraph (150-250 words) a smallholder farmer can read directly -- introduce all three recommendations, lead with the top choice and why it's #1, briefly note what distinguishes #2 and #3 as genuine alternatives (not just runners-up), and name the key caveats")



# ---------------------- Prompts ----------------------------

PLAIN_LANGUAGE_INSTRUCTION = """Write every farmer-facing text field in simple, everyday English -- as if
explaining to someone with limited formal education and limited English fluency. Use short sentences and
common words instead of technical or academic terms. For example: say "this uses weather patterns we haven't
seen before, so it may be less accurate" instead of "climate extrapolation risk"; say "a rougher, less certain
price estimate" instead of "Derived_Average category price"; say "this crop is grown on much less land here"
instead of "low historical area share". This is about simplifying VOCABULARY AND SENTENCE STRUCTURE, not
removing necessary specifics -- still give exact fertilizer amounts, exact seed variety names, and exact
figures wherever the guidance calls for them; just express everything in plain, clear language a first-time
reader can follow without a dictionary."""


CORE_LIMITATIONS = """These five apply to every report, always -- include the substance of all five regardless of the specific query:
1. Engine 1 is a historical/statistical baseline -- it has no data on seed variety, pest pressure, or real-time input quality.
2. Engine 3's training data (soil-to-crop matching) is synthetic (GPT-generated), a heuristic pending validation against ICAR/Soil Health Card reference tables -- not ground truth.
3. District-level satellite soil data is a static average for the whole district, not the farmer's own field -- a fallback signal only, never a replacement for the farmer's actual Soil Health Card.
4. MSP/revenue figures need annual updates and may already be stale.
5. Nationally, roughly a fifth of District+Crop+Season combinations have very limited (2 years or fewer) yield history -- predictions in general carry this caveat, even though this specific shortlist doesn't expose per-crop history depth.

Conditionally include these, only when the data indicates they apply to this specific query:
6. If a category-aggregate crop (Cereals, Kharif Pulse, Rabi Pulse, Oilseed, or Millet) lands in the top 3, disclose it isn't an individually actionable variety.
7. If climate_extrapolation_risk is true for a recommended crop's shortlist entry, disclose plainly that its ranking rests on climate conditions the model never saw in training -- a confidence flag, not a corrected prediction.
8. If no_engine2_forecast is true for a recommended crop's shortlist entry, disclose that its score was computed using historical climate rather than Engine 2's forward-looking forecast for that season."""


def build_main_prompt(report_json: dict) -> str:
    return f"""You are an agricultural expert advising a smallholder farmer in India on what crop to grow this season.

You are given structured output from three ML models (a yield/suitability engine, a climate forecast engine, and a soil-nutrient re-ranking engine) plus the farmer's raw input. You do not run any ML yourself -- reason only over what's given below.

STRUCTURED INPUT:
{json.dumps(report_json, indent=2)}

SCHEMA NOTES (read carefully before reasoning):
- engine1_output.shortlist entries each carry their own "season" field. This may differ from farmer_input.season due to state-specific season-label handling (e.g. Assam/WB/Kerala record Rice under Autumn/Summer/Winter, not Kharif/Rabi) -- this is intentional, not an error.
- If a shortlisted crop's own "season" differs from farmer_input.season, you MUST fill in season_mismatch_disclosed for that crop: state plainly which season it's actually grown in locally, and that this differs from what the farmer selected. Do not just mention this in passing within rank_rationale -- it needs its own clear, dedicated disclosure so the farmer isn't confused about why a differently-timed crop appears in their results.
- engine2_output is a dict keyed by season (not one flat object). For each shortlisted crop, look up engine2_output[that crop's own "season" field] to get its correct climate context (climate_label, forecast_T_Avg, forecast_Rainfall_Total).
- revenue_estimate[crop] contains predicted_yield_tonnes_per_hectare, already computed by Engine 1 -- copy this number exactly into your response, do not recompute or re-derive it from revenue_per_hectare.

YOUR TASKS:
- Select the final top 3 crops from engine1_output.shortlist. If engine3_output.reranked_within_shortlist is present (non-null), use it as a secondary nutrient-fit signal to help choose among close-scoring shortlist entries -- it is not an independent ranking and cannot introduce a crop absent from the shortlist.
- If a category-aggregate crop (Cereals/Kharif Pulse/Rabi Pulse/Oilseed/Millet) would land in the top 3, relabel it per the schema instructions rather than silently dropping it for the next-ranked crop.
- Follow note_to_gemini in the input exactly, including the footprint-flag instruction.
- Recommend seed variety, cultivation practices, fertilizer/irrigation guidance, and expected yield/price for each of the top 3 -- full depth for all three, not just the top pick.
- All fertilizer and seed-rate quantities must be stated in a single consistent unit -- kilograms per hectare (kg/ha) -- never mixed with or substituted by local land units (bigha, katha, etc.) without an explicit stated conversion.
- {PLAIN_LANGUAGE_INSTRUCTION}
- Do NOT state any specific total production (tonnes) or total revenue (Rs) figures anywhere in your response, including farmer_facing_summary -- those depend on the farmer's individual land area and are computed separately, outside this response. You may speak qualitatively/comparatively about revenue potential per hectare across the three crops.
- {CORE_LIMITATIONS}
- Do not overstate certainty anywhere in the report.

Respond only in the JSON schema provided."""


def build_fallback_prompt(farmer_input: dict, engine3_output: Optional[dict] = None) -> str:
    engine3_block = (
        f"\n\nThe farmer also supplied a Soil Health Card. An independent soil-nutrient "
        f"model (location-independent -- not restricted to any local shortlist, since none "
        f"exists for this district) ranked crops purely by how well this farmer's soil "
        f"chemistry matches them:\n{json.dumps(engine3_output, indent=2)}\n"
        f"Use this as a genuine prioritization signal among your general-knowledge crop "
        f"candidates -- but it reflects soil-nutrient fit only, not climate suitability or "
        f"local market viability, so weigh it alongside your agronomic/regional knowledge, "
        f"not above it."
        if engine3_output else "\n\nNo Soil Health Card was supplied."
    )

    return f"""You are an agricultural expert advising a smallholder farmer in India on what crop to grow this season.

IMPORTANT: This farmer's State+District is not present in our historical yield or climate datasets. No local ML prediction is available. You must reason entirely from general agricultural knowledge for this input profile -- do not imply any ML backing for this recommendation.

FARMER INPUT:
{json.dumps(farmer_input, indent=2)}{engine3_block}

YOUR TASKS:
- Recommend up to 3 crops well-suited to this State, season, and irrigation level, using general agronomic knowledge (typical crops grown in this region/season, water need, market viability).
- If a soil-nutrient signal was supplied above, use it to help choose among candidate crops.
- Recommend seed variety, cultivation practices, fertilizer/irrigation guidance, and a rough expected yield/price estimate (general market knowledge, not MSP-backed -- state this plainly in price_confidence_note).
- All fertilizer and seed-rate quantities must be stated in a single consistent unit -- kilograms per hectare (kg/ha) -- never mixed with or substituted by local land units (bigha, katha, etc.) without an explicit stated conversion.
- {PLAIN_LANGUAGE_INSTRUCTION}
- Do NOT state any specific total production (tonnes) or total revenue (Rs) figures anywhere in your response, including farmer_facing_summary -- those depend on the farmer's individual land area and are computed separately, outside this response.
- Set fallback_mode to true.
- {CORE_LIMITATIONS}
- Above all, make unmistakably clear in farmer_facing_summary that this recommendation is not backed by local historical data for their specific district -- it is general agricultural guidance.

Respond only in the JSON schema provided."""



# ---------------------------- API calls ------------------------------------

def call_gemini_main(report_json: dict) -> SanjeevniReport:
    interaction = _call_with_timeout(
        client.interactions.create,
        model=MODEL_NAME,
        input=build_main_prompt(report_json),
        response_format={"type": "text", "mime_type": "application/json", "schema": SanjeevniReport.model_json_schema()},
        store=False,
    )
    return SanjeevniReport.model_validate_json(get_interaction_text(interaction))


def call_gemini_fallback(farmer_input: dict, engine3_output: Optional[dict] = None) -> SanjeevniReport:
    interaction = _call_with_timeout(
        client.interactions.create,
        model=MODEL_NAME,
        input=build_fallback_prompt(farmer_input, engine3_output),
        response_format={"type": "text", "mime_type": "application/json", "schema": SanjeevniReport.model_json_schema()},
        store=False,
    )
    return SanjeevniReport.model_validate_json(get_interaction_text(interaction))



# Deterministic totals -- Area x Yield x Price computed in Python, never by the LLM. 
# Runs identically for main-path and fallback-path reports.

def enrich_with_totals(report: SanjeevniReport, report_json: dict) -> dict:
    area_hectares = report_json["farmer_input"]["area_hectares"]
    enriched = report.model_dump()
    enriched["farmer_area_hectares"] = area_hectares
    for rec in enriched["top_recommendations"]:
        production_tonnes = round(rec["predicted_yield_tonnes_per_hectare"] * area_hectares, 2)
        total_revenue = round(production_tonnes * rec["price_per_tonne"], 2)
        rec["total_expected_production_tonnes"] = production_tonnes
        rec["total_expected_revenue_rupees"] = total_revenue
    return enriched



# --------------- get_sanjeevni_report() -- THE single entry point Streamlit should call --------------------
# Hides main-vs-fallback routing entirely. Builds its own report_json via integration.py's assemble_farmer_report() when the district is covered -- the caller only ever provides the farmer's raw inputs.

def get_sanjeevni_report(state: str, district: str, season: str, irrigation: str,
                          area_hectares: float, soil_card: dict = None,
                          dry_run: bool = False):
    covered = is_district_covered(state, district)

    if covered:
        report_json = assemble_farmer_report(
            state=state, district=district, season=season,
            irrigation_level=irrigation, area_hectares=area_hectares, soil_card=soil_card,
        )
        prompt = build_main_prompt(report_json)
        route_info = {
            "path": "main (ML-backed)",
            "district_covered": True,
            "prompt_preview": prompt[:500] + " ...[truncated]",
            "prompt_length_chars": len(prompt),
        }
        if dry_run:
            return route_info
        result = call_gemini_main(report_json)
        return enrich_with_totals(result, report_json)

    else:
        farmer_input = {
            "state": state, "district": district, "season": season,
            "irrigation": irrigation, "area_hectares": area_hectares,
            "soil_health_card": soil_card,
        }
        # Engine 3 is location-independent, so it runs here toowhen a soil card is supplied -- via predict_soil_candidates(), NOT rerank_with_soil() (that one requires an existing shortlist, which doesn't exist in fallback mode). Fixed per the UI addendum: soil card is now mandatory in the UI, so this is no longer an edge case -- every fallback-path farmer supplies one.
        engine3_output = None
        if soil_card is not None:
            engine3_output = predict_soil_candidates(
                N=soil_card.get("N"), P=soil_card.get("P"), K=soil_card.get("K"),
                OC=soil_card.get("OC"), pH=soil_card.get("pH"),
            )

        prompt = build_fallback_prompt(farmer_input, engine3_output)
        route_info = {
            "path": "fallback (no local ML data)",
            "district_covered": False,
            "prompt_preview": prompt[:500] + " ...[truncated]",
            "prompt_length_chars": len(prompt),
        }
        if dry_run:
            return route_info
        result = call_gemini_fallback(farmer_input, engine3_output)
        return enrich_with_totals(result, {"farmer_input": farmer_input})
