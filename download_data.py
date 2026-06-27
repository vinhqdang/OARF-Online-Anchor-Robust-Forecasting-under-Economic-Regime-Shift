#!/usr/bin/env python3
"""Download the public data panel (Sec 3) into ``data/``.

Sources (all public, citation-requested):
* FRED CSV endpoint — Brent, Henry Hub, gold, VIX, EUR/USD, 10y Treasury;
* Caldara & Iacoviello Geopolitical Risk (GPR) daily Excel;
* Baker, Bloom & Davis Economic Policy Uncertainty (EPU) daily CSV.

EUA carbon has no clean programmatic URL (Sec 3.4): export it by hand and place
it at ``data/eua.csv`` (a date column and a price/close column); the loader
picks it up automatically.  Requires outbound HTTPS.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests

DATA = "data"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_IDS = ["DCOILBRENTEU", "DHHNGSP", "GOLDPMGBD228NLBM", "VIXCLS",
            "DEXUSEU", "DGS10"]
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (OARF research data fetch)"}


def _get(url, timeout=90):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def main():
    os.makedirs(DATA, exist_ok=True)
    for sid in FRED_IDS:
        try:
            content = _get(FRED.format(sid))
            df = pd.read_csv(io.BytesIO(content))
            df.to_csv(os.path.join(DATA, f"{sid}.csv"), index=False)
            print(f"[ok] FRED {sid}: {len(df)} rows")
        except Exception as e:                # pragma: no cover - network
            print(f"[skip] FRED {sid}: {e}")
    try:
        open(os.path.join(DATA, "gpr_daily.xls"), "wb").write(_get(GPR_URL))
        print("[ok] GPR daily Excel")
    except Exception as e:                    # pragma: no cover - network
        print(f"[skip] GPR: {e}")
    try:
        open(os.path.join(DATA, "epu_daily.csv"), "wb").write(_get(EPU_URL))
        print("[ok] EPU daily CSV")
    except Exception as e:                    # pragma: no cover - network
        print(f"[skip] EPU: {e}")
    if not os.path.exists(os.path.join(DATA, "eua.csv")):
        print("[note] data/eua.csv not present — export EUA manually (Sec 3.4); "
              "the loader runs without it.")


if __name__ == "__main__":
    main()
