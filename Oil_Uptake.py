# oil_uptake_app.py
# Streamlit app to estimate frying oil uptake using fixed defaults + per-ingredient WATER (per 100 g)
# MongoDB source: DB 'Fitshield', collection 'Nutrients'
# Coating level is auto-detected from ingredient amounts (no manual selection).
# NOTE: MongoDB URI is configured in code (not exposed in UI).

import json
import math
import re
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# =========================
# Secure-ish Config (kept in code, not UI)
# =========================
# Replace this with your real URI (e.g., MongoDB Atlas or local instance).
MONGO_URI = "mongodb://fitshield:fitshield123@13.235.70.79:27017/Fitshield?directConnection=true&appName=mongosh+2.4.2"

DB_NAME = "Fitshield"
COLL_NAME = "Nutrients"

# =========================
# App Config
# =========================

st.set_page_config(page_title="FitShield • Oil Uptake Estimator", layout="wide")
st.title("🛢️ Oil Uptake Estimator")
st.caption("Fixed-defaults model • per-ingredient WATER (per 100 g) • temperature-aware kinetics (Q10) • Mongo-backed nutrients • auto coating detection")

# =========================
# Fixed Defaults (Single Source of Truth)
# =========================

METHODS_ENABLED = {"Deep Frying", "Pan Frying", "Sautéing"}  # modeled methods
RELEVANT_CATEGORIES = {
    "Meat & Poultry", "Fish & Seafood", "Vegetables & Greens",
    "Legumes & Beans", "Sweets & Confectionery", "Dairy Products"
}
IGNORED_CATEGORIES = {"Beverages", "Fruits", "Nuts & Seeds", "Spices & Seasonings", "Oils & Fats"}

K_REF_PER_MIN = {"Deep Frying": 0.30, "Pan Frying": 0.12, "Sautéing": 0.08}
Q10_BY_METHOD = {"Deep Frying": 2.0, "Pan Frying": 1.8, "Sautéing": 1.6}
GAMMA_BY_METHOD = {"Deep Frying": 0.20, "Pan Frying": 0.15, "Sautéing": 0.12}
TEMP_BANDS_C = {"Deep Frying": (160, 190), "Pan Frying": (150, 185), "Sautéing": (140, 175)}

# Equilibrium oil uptakes (g oil / g dry), defaults by (method, meal_category)
OE_DEFAULTS = {
    "Deep Frying": {
        "Meat & Poultry": 0.26, "Fish & Seafood": 0.24, "Vegetables & Greens": 0.32,
        "Legumes & Beans": 0.28, "Sweets & Confectionery": 0.25, "Dairy Products": 0.17
    },
    "Pan Frying": {
        "Meat & Poultry": 0.10, "Fish & Seafood": 0.10, "Vegetables & Greens": 0.12,
        "Legumes & Beans": 0.11, "Sweets & Confectionery": 0.11, "Dairy Products": 0.08
    },
    "Sautéing": {
        "Meat & Poultry": 0.06, "Fish & Seafood": 0.06, "Vegetables & Greens": 0.07,
        "Legumes & Beans": 0.06, "Sweets & Confectionery": 0.06, "Dairy Products": 0.05
    }
}

# Coating multipliers (applied automatically via detection)
COATING_MULT = {"none": 1.00, "light": 1.30, "crumb_batter": 1.40}

# Ingredients to exclude from "dry solids entering fryer"
EXCLUDE_FREE_KEYWORDS = ("oil", "ghee", "butter", "water", "milk")

# Basic density heuristics for ml -> g (only if qty given in ml)
# Fallback is 1.0 g/ml (water-like) if no match
DENSITY_G_PER_ML = [
    (r"\boil\b|\bvegetable oil\b|\bcanola\b|\bpeanut oil\b|\bghee\b|\bbutter\b", 0.91),
    (r"\bmilk\b|\byogurt\b|\bcurd\b", 1.03),
    (r"\bhoney\b", 1.42),
    (r"\bsoy sauce\b|\bsauce\b", 1.10),
    (r"\bwater\b", 1.00),
]

# Synonyms for nutrient lookup normalization
SYNONYMS = {
    "all-purpose flour": "wheat flour",
    "ap flour": "wheat flour",
    "maida": "wheat flour",
    "corn starch": "cornstarch",
    "garlic pwd": "garlic powder",
}

# Keywords that indicate coating ingredients
COATING_KEYWORDS = (
    "flour", "cornstarch", "starch", "rice flour", "maida", "gram flour", "besan",
    "semolina", "suji", "bread crumb", "breadcrumbs", "panko", "batter", "tempura"
)

# =========================
# Helpers
# =========================

def to_lower(s: str) -> str:
    return s.strip().lower()

def canonical_name(name: str) -> str:
    n = to_lower(name)
    return SYNONYMS.get(n, n)

def grams_from_quantity_unit(name: str, qty: float, unit: str) -> float:
    """
    Convert qty with unit to grams. Supports 'g' and 'ml'.
    For 'ml', multiply by an estimated density based on ingredient keywords.
    """
    unit_l = to_lower(unit)
    if unit_l in {"g", "gram", "grams"}:
        return qty
    if unit_l in {"ml", "milliliter", "milliliters"}:
        name_l = to_lower(name)
        for pattern, dens in DENSITY_G_PER_ML:
            if re.search(pattern, name_l):
                return qty * dens
        return qty * 1.0  # default water-like
    # unsupported unit: return None to ignore safely
    return None

def is_excluded_free_liquid(name: str) -> bool:
    n = to_lower(name)
    return any(k in n for k in EXCLUDE_FREE_KEYWORDS)

# =========================
# Mongo Nutrient DB (WATER per 100 g) loader
# =========================

@st.cache_data(show_spinner=False)
def build_water_lookup_from_mongo() -> Dict[str, float]:
    """
    Connect to MongoDB, read Fitshield.Nutrients, and build:
    normalized_name -> WATER_per_100g (float)

    Expected doc examples:
      {
        "food_name": "Wheat flour",
        "nutrients": [
          {"name": "WATER", "quantity": 12.0, "unit": "g"},
          ...
        ]
      }
    or:
      {
        "name": "Wheat flour",
        "nutrients": {"WATER": 12.0, "PROTEIN": ...}
      }
    """
    lut: Dict[str, float] = {}
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    try:
        client.admin.command("ping")
        coll = client[DB_NAME][COLL_NAME]
        cursor = coll.find({}, {"_id": 0, "food_name": 1, "name": 1, "ingredient": 1, "nutrients": 1})
        for row in cursor:
            fname = row.get("food_name") or row.get("name") or row.get("ingredient") or ""
            if not fname:
                continue
            fname_norm = canonical_name(fname)
            water_val = None
            nutrients = row.get("nutrients")
            if isinstance(nutrients, list):
                for n in nutrients:
                    if to_lower(n.get("name", "")) == "water":
                        try:
                            water_val = float(n.get("quantity"))
                        except Exception:
                            pass
                        break
            elif isinstance(nutrients, dict):
                for k, v in nutrients.items():
                    if to_lower(k) == "water":
                        try:
                            water_val = float(v)
                        except Exception:
                            pass
                        break
            if water_val is not None:
                lut[fname_norm] = water_val
    finally:
        client.close()
    return lut

# =========================
# Core Calculations
# =========================

def clamp_temperature(method: str, T_c: float) -> float:
    band = TEMP_BANDS_C.get(method)
    if not band:
        return T_c
    tmin, tmax = band
    return max(tmin, min(tmax, T_c))

def k_from_temp(method: str, T_c: float) -> float:
    """
    Temperature-aware rate constant using Q10 scaling from 177 °C.
    """
    k_ref = K_REF_PER_MIN[method]
    Q10 = Q10_BY_METHOD[method]
    T_use = clamp_temperature(method, T_c)
    return k_ref * (Q10 ** ((T_use - 177.0) / 10.0))

def compute_dry_solids_and_coating(
    dish: Dict[str, Any],
    water_lut: Dict[str, float]
) -> Tuple[pd.DataFrame, float, float, float, str]:
    """
    Returns:
      - breakdown_df
      - dry_total_g
      - dry_from_db_fraction
      - coating_dry_g (sum of dry solids from coating ingredients)
      - coating_level ('none'|'light'|'crumb_batter')

    Coating level rules (amount + keywords):
      - If any 'batter'/'panko'/'bread crumb' present -> 'crumb_batter'
      - Else compute coating_dry_ratio = coating_dry_g / dry_total_g:
            >= 0.20 -> 'crumb_batter'
       else if >= 0.08 -> 'light'
       else -> 'none'
    """
    rows = []
    dry_total = 0.0
    dry_from_db = 0.0
    coating_dry = 0.0

    ingredient_names_lower = [to_lower(x.get("name", "")) for x in dish.get("ingredients", [])]
    has_crumb_batter_kw = any(re.search(r"bread\s*crumb|breadcrumbs|panko|batter|tempura", n) for n in ingredient_names_lower)

    for ing in dish.get("ingredients", []):
        name = ing.get("name", "")
        qty = float(ing.get("quantity", 0) or 0)
        unit = ing.get("unit", "g")

        grams = grams_from_quantity_unit(name, qty, unit)
        if grams is None or grams <= 0:
            continue
        if is_excluded_free_liquid(name):
            rows.append({
                "ingredient": name, "qty_input": f"{qty} {unit}", "qty_g": grams,
                "WATER_per100g": None, "moisture_frac": None, "dry_g": 0.0, "is_coating": False,
                "note": "excluded (free liquid)"
            })
            continue

        key = canonical_name(name)
        water_per100 = water_lut.get(key)
        if water_per100 is None:
            dry_g = None
            note = "WATER not found in DB (ignored for dry solids)"
            is_coating = any(kw in to_lower(name) for kw in COATING_KEYWORDS)
        else:
            moisture = max(0.0, min(1.0, water_per100 / 100.0))
            dry_g = grams * (1.0 - moisture)
            dry_total += dry_g
            dry_from_db += dry_g
            is_coating = any(kw in to_lower(name) for kw in COATING_KEYWORDS)
            if is_coating:
                coating_dry += dry_g
            note = ""

        rows.append({
            "ingredient": name,
            "qty_input": f"{qty} {unit}",
            "qty_g": grams,
            "WATER_per100g": water_per100,
            "moisture_frac": None if water_per100 is None else round(water_per100 / 100.0, 4),
            "dry_g": 0.0 if dry_g is None else round(dry_g, 3),
            "is_coating": bool(is_coating and dry_g is not None),
            "note": note
        })

    dry_from_db_fraction = 0.0 if dry_total <= 0 else (dry_from_db / dry_total)
    df = pd.DataFrame(rows)

    # Coating level decision
    coating_ratio = (coating_dry / dry_total) if dry_total > 0 else 0.0
    if has_crumb_batter_kw or coating_ratio >= 0.20:
        coating_level = "crumb_batter"
    elif coating_ratio >= 0.08:
        coating_level = "light"
    else:
        coating_level = "none"

    return df, dry_total, dry_from_db_fraction, coating_dry, coating_level

def oil_uptake_fixed_defaults(
    dish: Dict[str, Any],
    meal_category: str,
    method: str,
    water_lut: Dict[str, float]
) -> Dict[str, Any]:
    """
    Main pipeline with auto coating detection:
    - gates (method/category)
    - dry solids + coating amounts from per-ingredient WATER
    - Oe selection + auto coating multiplier
    - temperature-aware k(T) using Q10
    - fixed gamma by method
    - output grams absorbed + context
    """
    out = {"ok": False, "message": "", "result": None}

    # Gates
    if method not in METHODS_ENABLED:
        out["message"] = f"Method '{method}' not modeled (returns 0)."
        out["result"] = {"oil_absorbed_g": 0.0, "reason": "method"}
        return out
    if meal_category not in RELEVANT_CATEGORIES:
        out["message"] = f"Meal category '{meal_category}' considered negligible (returns 0)."
        out["result"] = {"oil_absorbed_g": 0.0, "reason": "category"}
        return out

    # Dry solids & coating auto-detection
    df, dry_total_g, dry_db_frac, coating_dry_g, coating_level = compute_dry_solids_and_coating(dish, water_lut)
    if dry_total_g <= 0:
        out["message"] = "No dry solids computed (check WATER in nutrient DB, units, or exclusions)."
        out["result"] = {"oil_absorbed_g": 0.0, "reason": "no_dry_solids", "dry_total_g": 0.0, "breakdown": df.to_dict("records")}
        return out

    # Parameters
    Oe_base = OE_DEFAULTS[method][meal_category]
    Oe = Oe_base * COATING_MULT.get(coating_level, 1.0)

    # Temperature & time
    temp_field = str(dish.get("cooking_temperature", "")).strip()
    try:
        T_c = float(temp_field.split()[0])
    except Exception:
        T_c = 177.0  # safe default
    t_field = str(dish.get("cooking_time", "")).strip()
    try:
        t_min = float(t_field.split()[0])
    except Exception:
        t_min = 6.0  # safe default

    kT = k_from_temp(method, T_c)
    gamma = GAMMA_BY_METHOD[method]

    # Kinetics
    O_cook = Oe * (1.0 - math.exp(-kT * t_min))
    O_final = O_cook + gamma * (Oe - O_cook)
    O_final = min(O_final, Oe)  # safety clamp

    oil_absorbed_g = O_final * dry_total_g

    out["ok"] = True
    out["message"] = "Success"
    out["result"] = {
        "dish_name": dish.get("dish_name"),
        "method": method,
        "meal_category": meal_category,
        "coating": coating_level,
        "dry_total_g": round(dry_total_g, 3),
        "coating_dry_g": round(coating_dry_g, 3),
        "coating_ratio": 0.0 if dry_total_g == 0 else round(coating_dry_g / dry_total_g, 3),
        "dry_from_db_fraction": round(dry_db_frac, 3),
        "Oe_g_per_g_dry": round(Oe, 4),
        "k_per_min": round(kT, 4),
        "gamma": gamma,
        "O_final_g_per_g_dry": round(O_final, 4),
        "oil_absorbed_g": round(oil_absorbed_g, 2),
        "breakdown_rows": df.to_dict("records"),
    }
    return out

# =========================
# Nutrient lookup build (no UI for URI)
# =========================

water_lookup: Dict[str, float] = {}
try:
    water_lookup = build_water_lookup_from_mongo()
    if water_lookup:
        st.success(f"Connected to Mongo • WATER lookup built for {len(water_lookup)} items from {DB_NAME}.{COLL_NAME}.")
    else:
        st.warning(f"Connected to Mongo, but found 0 items with WATER in {DB_NAME}.{COLL_NAME}.")
except PyMongoError as e:
    st.error(f"Mongo connection error: {e}")
except Exception as e:
    st.error(f"Failed to build WATER lookup: {e}")

# =========================
# UI: Dish JSON Input
# =========================

st.subheader("Dish JSON Input")
st.caption("Required keys: dish_name, cooking_style, cooking_temperature, cooking_time, cooking_method, meal_category, ingredients[]")

example_json = {
  "dish_name": "Deep Fried Chicken Wings",
  "cooking_style": "Deep Frying",
  "cooking_method": "Deep Frying",
  "cooking_temperature": "177 C",
  "cooking_time": "8 min",
  "meal_category": "Meat & Poultry",
  "ingredients": [
    {
      "name": "Chicken wing",
      "quantity": 500.0,
      "unit": "g"
    },
    {
      "name": "Flour all purpose",
      "quantity": 80.0,
      "unit": "g"
    },
    {
      "name": "Cornstarch",
      "quantity": 30.0,
      "unit": "g"
    },
    {
      "name": "Spices garlic powder",
      "quantity": 2.0,
      "unit": "g"
    },
    {
      "name": "Salt",
      "quantity": 2.5,
      "unit": "g"
    },
    {
      "name": "Black pepper",
      "quantity": 1.0,
      "unit": "g"
    },
    {
      "name": "Vegetable oil",
      "quantity": 2000.0,
      "unit": "ml"
    }
  ]
}
input_str = st.text_area("Paste your dish JSON here", value=json.dumps(example_json, indent=2), height=350)

# Run button
run = st.button("Compute Oil Uptake")

# =========================
# Execution
# =========================

if run:
    # Parse JSON
    try:
        dish = json.loads(input_str)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    # Pull method/category (prefer explicit fields)
    method = dish.get("cooking_method") or dish.get("cooking_style") or "Deep Frying"
    category = dish.get("meal_category") or "Vegetables & Greens"

    if not water_lookup:
        st.warning(f"No WATER data available from {DB_NAME}.{COLL_NAME}. Dry solids will be 0 because the lookup is empty.")

    # Compute
    result_pack = oil_uptake_fixed_defaults(
        dish=dish,
        meal_category=category,
        method=method,
        water_lut=water_lookup
    )

    # Output
    if not result_pack["ok"]:
        st.error(result_pack["message"])
        res = result_pack.get("result", {})
        if "breakdown" in res:
            st.dataframe(pd.DataFrame(res["breakdown"]))
        st.stop()

    res = result_pack["result"]

    # Summary
    st.success("Oil uptake calculated.")
    cols = st.columns(5)
    cols[0].metric("Dry solids (g)", res["dry_total_g"])
    cols[1].metric("Coating dry (g)", res["coating_dry_g"])
    cols[2].metric("Coating level", res["coating"].upper())
    cols[3].metric("k(T) (1/min)", res["k_per_min"])
    cols[4].metric("Oil absorbed (g)", res["oil_absorbed_g"])

    st.caption(
        f"Method: {method} • Category: {category} • Coating: {res['coating']} "
        f"• Oe={res['Oe_g_per_g_dry']} g/g • γ={res['gamma']} • Dry mass from DB fraction={res['dry_from_db_fraction']}"
    )

    # Breakdown table
    st.subheader("Ingredient Dry-Solids Breakdown")
    df_break = pd.DataFrame(res["breakdown_rows"])
    st.dataframe(df_break, use_container_width=True)

    # Raw JSON output (for API integration/testing)
    st.subheader("Raw Result JSON")
    st.code(json.dumps(res, indent=2), language="json")

# =========================
# Footer notes
# =========================
with st.expander("Model Notes & Guardrails"):
    st.markdown(f"""
- **Nutrient Source (Mongo):** DB **{DB_NAME}**, collection **{COLL_NAME}**. We read **WATER per 100 g** to compute moisture and dry solids.
- **Auto Coating:** Detected from amounts. We sum dry solids of coating-like ingredients (flour/starch/crumb/batter). Rules:
  - If any of 'batter', 'panko', or 'bread crumb' present **or** coating_dry / total_dry ≥ **0.20** → **crumb_batter**
  - Else if coating_dry / total_dry ≥ **0.08** → **light**
  - Else → **none**
- **Units:** Accepts `g` and `ml`. For `ml`, converts to grams by keyword-based densities (oil≈0.91 g/ml, milk≈1.03 g/ml, water≈1.0 g/ml).
- **Gates:** Only runs for `Deep Frying`, `Pan Frying`, `Sautéing` and oil-relevant meal categories (meat, fish, veg, legumes, sweets, dairy). Others return **0**.
- **Kinetics:** First-order approach to Oe with temperature scaling via Q10; fixed cooling fraction `γ` per method; clamp `O_final ≤ Oe` and temperature to realistic bands.
- **Consistency:** Dry solids include **food + coating** (not fryer oil / plain water/milk). Missing WATER entries are ignored (conservative).
- **Tuning:** Adjust the coating thresholds (0.08 / 0.20) and keyword lists to match your recipe conventions; update OE defaults per SKU when calibration data becomes available.
""")
