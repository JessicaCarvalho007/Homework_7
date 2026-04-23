# =============================================================================
# DEMO 2 — Water Table Depth Observations in Arizona (Last 5 Years)
# =============================================================================
# Course  : HWRS 564b — Hydrogeologic Analysis Tools & Methods II
# Topic   : Accessing point observation data through HydroFrame / HydroData
#
# What this script does:
#   1. Authenticates with the HydroFrame data platform
#   2. Downloads daily water table depth (WTD) observations from USGS NWIS
#      for all monitoring wells in Arizona over the last 5 years
#   3. Explores the spatial distribution and temporal coverage of the wells
#   4. Focuses on the Tucson area and plots time series for nearby wells
#
# Data source : USGS National Water Information System (NWIS), served through
#               HydroFrame's unified point observation API
#
# Package documentation: https://hydroframe.org/hydrodata
# USGS groundwater data: https://groundwaterwatch.usgs.gov/
# =============================================================================

# --- Imports -----------------------------------------------------------------
import numpy as np
from datetime import date
import hf_hydrodata as hf          # HydroFrame data access package

# =============================================================================
# STEP 1 — Authenticate with the HydroFrame platform
# =============================================================================
# You need a free account at: https://hydrogen.princeton.edu/register
# Once registered, you can get a PIN.
#
# Replace the placeholders below with your actual credentials.

HYDROFRAME_EMAIL = "your_email@example.com"   # <-- replace with your email
HYDROFRAME_PIN   = "your_pin_here"            # <-- replace with your PIN

hf.register_api_pin(HYDROFRAME_EMAIL, HYDROFRAME_PIN)
print("Authentication successful.")

# =============================================================================
# STEP 2 — Define the study domain and time period
# =============================================================================
# We will query ALL wells in Arizona over the last 5 years.
# The 'state' parameter is the most convenient geographic filter here —
# it accepts the two-letter postal abbreviation.
#
# Water table depth (WTD) is measured as depth-to-water below the land surface
# (positive values = deeper water table, in meters).

DATE_END   = date.today().strftime("%Y-%m-%d")    # today
DATE_START = str(date.today().year - 5) + DATE_END[4:]  # same month/day, 5 years ago

print(f"\nQuerying water table depth data:")
print(f"  State      : Arizona (AZ)")
print(f"  Date range : {DATE_START} to {DATE_END}")

# =============================================================================
# STEP 3 — Download WTD point observations (all of Arizona)
# =============================================================================
# get_point_data() returns a pandas DataFrame where:
#   - Each COLUMN is a monitoring site (identified by its USGS site ID)
#   - Each ROW    is a date
#   - Values are the water table depth in meters (depth below ground surface)
#
# 'usgs_nwis' is the USGS National Water Information System, the largest
# publicly available groundwater observation network in the U.S.

print("\nDownloading WTD data (this may take a moment for a full state query)...")
df_wtd = hf.get_point_data(
    dataset            = "usgs_nwis",        # USGS NWIS observation network
    variable           = "water_table_depth",# depth to water below land surface
    temporal_resolution= "daily",            # daily observations
    aggregation        = "mean",             # daily mean if multiple readings/day
    date_start         = DATE_START,
    date_end           = DATE_END,
    state              = "AZ",               # two-letter state code for Arizona
)

print(f"\nData shape: {df_wtd.shape}")
print(f"  → {df_wtd.shape[0]} days × {df_wtd.shape[1]} monitoring wells")
print(f"\nFirst 5 rows, first 10 columns:")
print(df_wtd.iloc[:5, :10].round(2))

# =============================================================================
# STEP 4 — Download site metadata (location of each well)
# =============================================================================
# get_point_metadata() returns a DataFrame with one row per site, containing
# coordinates, site name, elevation, and other attributes.
# This is essential for knowing WHERE each column of df_wtd is located.

print("\nDownloading site metadata...")
df_meta = hf.get_point_metadata(
    dataset            = "usgs_nwis",
    variable           = "water_table_depth",
    temporal_resolution= "daily",
    aggregation        = "mean",
    state              = "AZ",
)

print(f"Metadata shape: {df_meta.shape}")
print(f"Available columns: {list(df_meta.columns)}")
print(f"\nSample metadata (first 3 sites):")
print(df_meta.head(3))

# =============================================================================
# STEP 5 — Basic data quality check
# =============================================================================
# Real observation data is messy! Let's understand data coverage before plotting.

# Fraction of days with valid (non-NaN) observations, per site
data_coverage = df_wtd.notna().mean()  # values from 0 (no data) to 1 (complete)

n_good_sites = (data_coverage > 0.5).sum()
print(f"\nData coverage summary:")
print(f"  Total sites          : {len(df_wtd.columns)}")
print(f"  Sites with >50% data : {n_good_sites}")
print(f"  Sites with >90% data : {(data_coverage > 0.9).sum()}")


# =============================================================================
# END OF DEMO
# =============================================================================
# Key concepts covered:
#   - hf.register_api_pin()     → authentication
#   - hf.get_point_data()       → download point observation DataFrame
#   - hf.get_point_metadata()   → get coordinates and attributes of sites
#
# Differences from Demo 1 (gridded data):
#   - get_point_data() returns a DataFrame (sites as columns, dates as rows)
#   - get_gridded_data() returns a numpy array (dimensions: time, y, x)
#   - Point data needs metadata to know where observations are located
# =============================================================================
