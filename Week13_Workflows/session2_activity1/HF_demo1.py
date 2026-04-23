# =============================================================================
# DEMO 1 — Atmospheric Forcing Data for the Verde River Watershed (Arizona)
# =============================================================================
# Course  : HWRS 564b — Hydrogeologic Analysis Tools & Methods II
# Topic   : Accessing gridded data through HydroFrame / HydroData
#
# What this script does:
#   1. Authenticates with the HydroFrame data platform
#   2. Downloads daily precipitation and air temperature data for the
#      Verde River watershed from the CW3E atmospheric forcing dataset
#   3. Summarizes the data spatially (mean over the watershed) and plots
#      a time series to verify the download was successful
#
# Package documentation: https://hydroframe.org/hydrodata
# Data catalog browser : https://hydrogen.princeton.edu/data-catalog
# =============================================================================

# --- Imports -----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import hf_hydrodata as hf          # HydroFrame data access package

# =============================================================================
# STEP 1 — Authenticate with the HydroFrame platform
# =============================================================================
# You need a free account at: https://hydrogen.princeton.edu/register
# Once registered, you can get a PIN.
#
# Replace the placeholders below with your actual credentials.
# WARNING: never share your PIN or push it to a public repository!

HYDROFRAME_EMAIL = "your_email@example.com"   # <-- replace with your email
HYDROFRAME_PIN   = "your_pin_here"            # <-- replace with your PIN

hf.register_api_pin(email=HYDROFRAME_EMAIL, pin=HYDROFRAME_PIN)
print("Authentication successful.")

# =============================================================================
# STEP 2 — Define the study domain and time period
# =============================================================================
# The Verde River watershed in central Arizona, flowing from the
# Prescott area south to its confluence with the Salt River near Phoenix.
#
# We define a bounding box in geographic coordinates (decimal degrees).
# Format for latlng_bounds: [lat_min, lon_min, lat_max, lon_max]

LATLNG_BOUNDS = [33.5, -113.0, 35.0, -111.0]   # Verde River watershed

# Time range for the data request.
DATE_START = "2019-01-21"
DATE_END   = "2019-01-23"

print(f"\nStudy domain : {LATLNG_BOUNDS}")
print(f"Time period  : {DATE_START} to {DATE_END}")

# =============================================================================
# STEP 3 — Download daily PRECIPITATION data
# =============================================================================
# Dataset   : CW3E — Center for Western Weather and Water Extremes
#             A high-resolution atmospheric forcing dataset developed at
#             Scripps Institution of Oceanography, calibrated for the
#             western United States.
# Variable  : precipitation — total liquid water input at the surface (mm/day)
# Resolution: daily, ~1 km grid over CONUS2 domain
#
# The function returns a numpy array with dimensions [day, y, x]
# where y and x are the number of grid cells in the lat and lon directions.

print("\nDownloading precipitation data...")
precip = hf.get_gridded_data(
    dataset            = "CW3E",          # atmospheric forcing dataset
    variable           = "precipitation", # total precipitation
    temporal_resolution= "daily",         # daily values
    aggregation        = "sum",           # sum over the day (for precip)
    date_start         = DATE_START,
    date_end           = DATE_END,
    latlng_bounds      = LATLNG_BOUNDS,
)

print(f"  Downloaded array shape: {precip.shape}  → (days, y_cells, x_cells)")
print(f"  Total days            : {precip.shape[0]}")
print(f"  Spatial grid          : {precip.shape[1]} rows × {precip.shape[2]} columns")

# =============================================================================
# STEP 4 — Download daily AIR TEMPERATURE data
# =============================================================================
# Variable  : air_temp — near-surface (2m) air temperature (K)
# Aggregation: mean — daily average temperature
#              (alternatives: 'min', 'max' for daily min/max temperature)

print("\nDownloading air temperature data...")
temp = hf.get_gridded_data(
    dataset            = "CW3E",
    variable           = "air_temp",
    temporal_resolution= "daily",
    aggregation        = "mean",          # daily mean temperature
    date_start         = DATE_START,
    date_end           = DATE_END,
    latlng_bounds      = LATLNG_BOUNDS,
)

print(f"  Downloaded array shape: {temp.shape}  → (days, y_cells, x_cells)")

# =============================================================================
# STEP 5 — Compute cummulative precipitation and mean temperature
# =============================================================================
# Each time step contains a 2D grid of values (one per grid cell).
# We accumulate/average across all time steps to get a single
# map of each variable for the entire period.

# Temporal sum/mean over all the timesteps
precip_cummulative = np.nansum(precip, axis = 0)   # shape: (days,)
temp_mean   = np.nanmean(temp,   axis = 0)   # shape: (days,)

# =============================================================================
# STEP 6 — Plot the results
# =============================================================================
# Two-panel figure: daily precipitation (bar) + daily temperature (line)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
fig.suptitle(
    "Verde River Watershed — Atmospheric Forcings (CW3E)\n"
    f"{DATE_START} to {DATE_END}",
    fontsize=14, fontweight="bold"
)

# Panel 1 — Precipitation
im0 = axes[0].imshow(precip_cummulative, cmap = 'Blues', origin = 'lower')
cbar = fig.colorbar(im0, ax=axes[0], orientation='horizontal', 
                    shrink = 0.7, label='Cummulative Precipitation (mm)')


# Panel 2 — Air Temperature
im1 = axes[1].imshow(temp_mean, cmap = 'Reds', origin = 'lower')
cbar = fig.colorbar(im1, ax=axes[1], orientation='horizontal', 
                    shrink = 0.7, label='Mean Air Temperature (K)')


plt.tight_layout()
plt.savefig("verde_river_forcings.png", dpi=150)
plt.show()
print("\nFigure saved as: verde_river_forcings.png")

# =============================================================================
# END OF DEMO
# =============================================================================
# Keys covered:
#   - hf.register_api_pin()     → authentication
#   - hf.get_gridded_data()     → download a gridded numpy array
#   - latlng_bounds             → spatial filter using lat/lon coordinates
#   - date_start / date_end     → temporal filter
# =============================================================================
