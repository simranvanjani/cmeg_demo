"""
============================================================================
 CMEG Demo — Configuration
============================================================================

Edit this file BEFORE running the demo. These are the only knobs you need.

Save the file, then open `00-CMEG-Demo-Intro.py` and click Run All.
"""

# --- Required ----------------------------------------------------------------

# Unity Catalog name. Must exist OR you must have CREATE CATALOG privileges.
# `main` exists in every workspace and is a safe default.
CATALOG = "main"

# Schema name. Will be created under CATALOG if it doesn't exist.
# All demo tables live here, prefixed by their medallion layer:
#   bronze_users, silver_users, gold_user_360, gold_interactions, ...
SCHEMA = "cmeg_demo"


# --- Optional (sensible defaults) --------------------------------------------

# Synthetic data scale. Controls how much fake data is generated.
#   "small"  = 10k users  / 5k items   / 500k interactions  (~15 min end-to-end)
#   "medium" = 100k users / 20k items  / 10M interactions   (~45 min)
#   "large"  = 1M users   / 100k items / 100M interactions  (heavy compute)
DATA_SCALE = "small"

# Foundation Model used for the "why we recommend this" GenAI explanations.
# Must be a Databricks Foundation Model endpoint available in your workspace.
GENAI_MODEL = "databricks-meta-llama-3-3-70b-instruct"

# Whether to create the real-time Model Serving endpoint.
# Set to False if you only care about batch scoring (saves DBUs).
SERVING_ENDPOINT_ENABLED = True
