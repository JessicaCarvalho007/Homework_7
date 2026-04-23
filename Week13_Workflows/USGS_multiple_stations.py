import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stations = ['09380000', '09498500', '09482000', '09506000', '09471000']


for station in stations:

    print(f"I'm processing station {station}")
    # Build the request
    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format": "json",
        "sites": station,        # Replace code for any station or list of stations
        "parameterCd": "00060",     # 00060 = streamflow (ft³/s)
        "startDT": "2020-01-01",    # Start date
        "endDT":   "2024-12-31",    # End date
        "siteStatus": "all"
    }

    # Make the request
    response = requests.get(url, params=params)
    data = response.json()

    # Parse — the response is nested JSON; this navigates to the values
    records = data["value"]["timeSeries"][0]["values"][0]["value"]
    df = pd.DataFrame(records)
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df["value"]    = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("dateTime")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["value"])
    plt.title(f"Daily Streamflow — Station {params['sites']}")
    plt.ylabel("Discharge (ft³/s)")
    plt.tight_layout()
    plt.savefig(f"streamflow_{station}.png", dpi=300)