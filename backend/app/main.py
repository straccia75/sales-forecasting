# Prophet-only, hardened
import os
import math
from typing import List, Optional, Dict, Any
from datetime import timedelta
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
# Configure a module-level logger early so every function can use consistent,
# timestamped messages. This is helpful both for debugging locally and for
# observability when deployed to production.
logger = logging.getLogger("forecast")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# Prophet import check
# -----------------------------------------------------------------------------
# Prophet is optional in some environments (like limited build pipelines).
# Here we safely attempt to import it and flag availability for runtime checks.
try:
    from prophet import Prophet  # type: ignore
    _PROPHET_AVAILABLE = True
except Exception as e:
    _PROPHET_AVAILABLE = False
    logger.error("Prophet is not available: %s", e)

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------
# Read environment variables to adapt runtime behavior.
# - APP_ENV controls whether we run in 'dev' (open CORS) or 'prod' (restricted CORS).
# - ALLOWED_ORIGINS defines allowed front-end domains.
APP_ENV = os.getenv("APP_ENV", "dev")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# -----------------------------------------------------------------------------
# FastAPI app initialization
# -----------------------------------------------------------------------------
# This service exposes forecasting endpoints for client-side dashboards or data
# pipelines. The app name and version appear in the interactive Swagger docs.
app = FastAPI(title="Sales Forecasting API", version="2.0.0")

# CORS configuration — permissive in dev, restrictive in prod.
if APP_ENV.lower() == "prod":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
# These Pydantic models validate input and output payloads for the API,
# providing strong typing and autogeneration of Swagger documentation.
class SchemaMap(BaseModel):
    """Defines column names in the uploaded dataset."""
    model_config = {"protected_namespaces": ()}
    date: str
    target: str
    group_by: List[str] = Field(default_factory=list)
    regressors: List[str] = Field(default_factory=list)

class ForecastParams(BaseModel):
    """Runtime configuration parameters for the forecast."""
    model_config = {"protected_namespaces": ()}
    horizon: Optional[int] = 30                # Number of periods to predict ahead
    confidence: Optional[float] = 0.8          # Confidence interval for Prophet
    frequency: Optional[str] = None            # 'D', 'W', or 'M'; inferred if None
    country_holidays: Optional[str] = None     # ISO country code for holiday effects
    model_preference: Optional[str] = "prophet"  # Legacy param kept for UI compatibility
    exog_future_policy: Optional[str] = "ffill"  # Legacy param, ignored by Prophet

class ForecastRequest(BaseModel):
    """Top-level request schema consumed by POST /forecast."""
    model_config = {
        "protected_namespaces": (),
        "populate_by_name": True,
        "validate_assignment": True,
    }
    data: List[Dict[str, Any]]
    schema: SchemaMap = Field(alias="schema")
    params: Optional[ForecastParams] = ForecastParams()

class ForecastPoint(BaseModel):
    """Single time-series point, historical or forecasted."""
    model_config = {"protected_namespaces": ()}
    date: str
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None
    group: Optional[Dict[str, Any]] = None
    kind: str  # "history" | "forecast"

class Metrics(BaseModel):
    """Evaluation metrics computed from lightweight backtests."""
    model_config = {"protected_namespaces": ()}
    mape: Optional[float] = None
    mae: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None

class ForecastResponse(BaseModel):
    """Complete response payload returned to clients."""
    model_config = {"protected_namespaces": ()}
    used_model: str
    detected_frequency: str
    warnings: List[str] = Field(default_factory=list)
    metrics_overall: Optional[Metrics] = None
    forecast: List[ForecastPoint]

# -----------------------------------------------------------------------------
# Utility: _coerce_datetime
# -----------------------------------------------------------------------------
def _coerce_datetime(s: pd.Series) -> pd.Series:
    """
    Attempts to robustly parse any column into timezone-naive datetimes.

    This helper handles multiple messy real-world cases:
    - epoch timestamps (seconds / milliseconds)
    - Excel serial numbers
    - multilingual month names (ES, IT → EN)
    - various string formats and localized ordinals
    - auto-scoring to pick the best parse strategy

    The function returns a UTC-normalized, tz-naive datetime series and logs
    informative warnings about dropped or invalid rows.
    """
    import re
    import numpy as np
    import pandas as pd

    src = s.copy()  # Keep original reference for logging

# --- Helper: numeric coercion -------------------------------------------------
    def _to_numeric(x: pd.Series) -> pd.Series:
        # Coerce any string or mixed column to numeric, keeping NaN where it fails.
        return pd.to_numeric(x, errors="coerce")

    # --- Helper: remove timezone information -------------------------------------
    def _tz_naive(dt: pd.Series) -> pd.Series:
        # Prophet expects tz-naive datetimes. Here we drop any timezone component
        # safely, trying both tz_convert and tz_localize depending on dtype.
        try:
            return dt.dt.tz_convert(None)
        except Exception:
            try:
                return dt.dt.tz_localize(None)
            except Exception:
                return dt

    # --- Helper: evaluate parsing quality ----------------------------------------
    def _score_series(dt: pd.Series) -> tuple:
        """Return a tuple (valid_ratio, year_penalty, uniq_ratio) — higher is better."""
        valid = dt.notna()
        valid_ratio = float(valid.mean())
        if valid_ratio == 0.0:
            return (0.0, -1.0, 0.0)
        years = dt[valid].dt.year
        in_range = ((years >= 1900) & (years <= 2100)).mean()
        uniq_ratio = dt[valid].nunique() / max(1, valid.sum())
        return (valid_ratio, float(in_range), float(uniq_ratio))

    # --- Helper: log invalid samples ---------------------------------------------
    def _log_invalid(dt: pd.Series, label: str):
        invalid = dt.isna()
        n = int(invalid.sum())
        if n:
            examples = (
                src[invalid]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            logger.warning(
                "%s: dropped %d rows with invalid dates. Examples: %s",
                label, n, examples
            )

    # --- Fast path: numeric epochs or Excel serials ------------------------------
    nums = _to_numeric(src)
    numeric_share = float(nums.notna().mean())
    candidates = []

    if numeric_share >= 0.85:
        # Try to guess whether numbers are Excel serials or UNIX epochs.
        serial_mask = nums.between(20000, 60000)
        if serial_mask.any():
            try:
                dt_excel = pd.to_datetime(nums, unit="D", origin="1899-12-30", utc=True)
                candidates.append(("excel_serial", _tz_naive(dt_excel)))
            except Exception:
                pass

        finite = np.isfinite(nums)
        if finite.any():
            q95 = float(np.nanquantile(nums[finite], 0.95))
            if q95 > 1e11:  # epoch milliseconds
                try:
                    dt_ms = pd.to_datetime(nums, unit="ms", utc=True)
                    candidates.append(("epoch_ms", _tz_naive(dt_ms)))
                except Exception:
                    pass
            if q95 > 1e9 or q95 <= 1e11:  # epoch seconds
                try:
                    dt_s = pd.to_datetime(nums, unit="s", utc=True)
                    candidates.append(("epoch_s", _tz_naive(dt_s)))
                except Exception:
                    pass


    # --- Normalize strings before parsing ---------------------------------------
    as_str = src.astype(str)
    norm = as_str.str.replace(
        r"(\b\d{1,2})(?:st|nd|rd|th|º|ª)\b", r"\1", regex=True, flags=re.IGNORECASE
    )
    norm = norm.str.replace(r"[.\u00B7]", "/", regex=True)  # Replace dots with slashes
    norm = norm.str.replace(r"\s+", " ", regex=True).str.strip()

    # Map month names from Spanish/Italian to English to improve date parsing
    month_map ={
        # Español
        "enero": "January", "ene": "Jan",
        "febrero": "February", "feb": "Feb",
        "marzo": "March", "mar": "Mar",
        "abril": "April", "abr": "Apr",
        "mayo": "May", "may": "May",
        "junio": "June", "jun": "Jun",
        "julio": "July", "jul": "Jul",
        "agosto": "August", "ago": "Aug",
        "septiembre": "September", "sep": "Sep", "setiembre": "September",
        "octubre": "October", "oct": "Oct",
        "noviembre": "November", "nov": "Nov",
        "diciembre": "December", "dic": "Dec",
        # Italiano
        "gennaio": "January", "gen": "Jan",
        "febbraio": "February", "feb": "Feb",
        "marzo": "March", "mar": "Mar",
        "aprile": "April", "apr": "Apr",
        "maggio": "May", "mag": "May",
        "giugno": "June", "giu": "Jun",
        "luglio": "July", "lug": "Jul",
        "agosto": "August", "ago": "Aug",
        "settembre": "September", "set": "Sep",
        "ottobre": "October", "ott": "Oct",
        "novembre": "November", "nov": "Nov",
        "dicembre": "December", "dic": "Dec",
    }
    # Replacing Caps
    def _replace_months(text: str) -> str:
        t = text
        for k, v in month_map.items():
            t = re.sub(rf"\b{k}\b", v, t, flags=re.IGNORECASE)
        return t

    norm = norm.apply(_replace_months)

    # --- Multiple Parsing Attempts ------------------------------------------
    attempts = []

    def _try(label: str, **kwargs):
        try:
            dt = pd.to_datetime(norm, errors="coerce", utc=True, **kwargs)
            attempts.append((label, _tz_naive(dt)))
        except Exception:
            pass

    # default/dateutil
    _try("pandas_default")
    _try("dayfirst", dayfirst=True)
    _try("yearfirst", yearfirst=True)

    #  common formats (avoids anbiguity dd/mm vs mm/dd)
    for fmt_label, fmt in [
        ("fmt_%d/%m/%Y", "%d/%m/%Y"),
        ("fmt_%m/%d/%Y", "%m/%d/%Y"),
        ("fmt_%Y-%m-%d", "%Y-%m-%d"),
        ("fmt_%d-%b-%Y", "%d-%b-%Y"),
        ("fmt_%d-%B-%Y", "%d-%B-%Y"),
        ("fmt_%d.%m.%Y", "%d.%m.%Y"),
    ]:
        try:
            dtf = pd.to_datetime(norm, format=fmt, errors="coerce", utc=True)
            attempts.append((fmt_label, _tz_naive(dtf)))
        except Exception:
            pass

    # dateparser (optional) — handles languages better and dirty texts
    try:
        import dateparser  # type: ignore
        def _dp_parse(x: str):
            return dateparser.parse(x, settings={"RETURN_AS_TIMEZONE_AWARE": True})
        dt_dp = norm.apply(_dp_parse)
        dt_dp = pd.to_datetime(dt_dp, errors="coerce", utc=True)
        attempts.append(("dateparser", _tz_naive(dt_dp)))
    except Exception:
        pass

    # Merging all candidates (numéricas + string)
    candidates += attempts

    if not candidates:
        # Last resource : previous behavior
        dt = pd.to_datetime(src, errors="coerce", utc=True)
        dt = _tz_naive(dt)
        _log_invalid(dt, "coerce_datetime(default)")
        return dt

    #Picks the best (valid_ratio, in_range, uniq_ratio)
    scored = [(name, dt, _score_series(dt)) for (name, dt) in candidates]
    scored.sort(key=lambda x: x[2], reverse=True)
    best_name, best_dt, best_score = scored[0]

    # Log to inform and invalidad examples
    logger.info(
        "coerce_datetime picked '%s' with scores (valid=%.3f, in_range=%.3f, uniq=%.3f)",
        best_name, best_score[0], best_score[1], best_score[2]
    )
    _log_invalid(best_dt, f"coerce_datetime[{best_name}]")

    return best_dt


# -----------------------------------------------------------------------------
# Utility: Frequency and date helpers
# -----------------------------------------------------------------------------
def _infer_frequency(dts: pd.Series) -> str:
    """
    Infer the dominant frequency of a datetime series.

    Prophet and most forecasting models require a consistent frequency
    ('D', 'W', or 'M'). This function uses pandas’ inference first,
    then falls back to manual delta inspection for robustness.
    """
    dts = dts.sort_values().dropna().unique()
    if len(dts) < 3:
        return "D"
    try:
        inferred = pd.infer_freq(pd.DatetimeIndex(dts))
        if inferred:
            if inferred.startswith("W"): return "W"
            if inferred.startswith("M") or inferred in ["MS", "ME"]: return "M"
            if inferred.startswith("D"): return "D"
        # Manual fallback — compute mode of deltas
        deltas = pd.Series(dts[1:]) - pd.Series(dts[:-1])
        delta = deltas.mode().iloc[0]
        days = delta / np.timedelta64(1, "D")
        if abs(days - 7) < 1: return "W"
        if 27 <= days <= 31: return "M"
        return "D"
    except Exception:
        return "D"

def _weekly_anchor(dts: pd.Series) -> int:
    """Find the most common weekday (0=Monday) to anchor weekly forecasts."""
    try:
        return int(pd.Series(pd.to_datetime(dts)).dt.weekday.mode().iloc[0])
    except Exception:
        return 0

def _monthly_anchor(dts: pd.Series) -> str:
    """
    Detect whether the data aligns better to month-start (MS)
    or month-end (ME) conventions, based on a majority rule.
    """
    idx = pd.to_datetime(dts).dropna()
    if idx.empty: return "ME"
    month_end_share = (idx == idx + pd.offsets.MonthEnd(0)).mean()
    return "ME" if month_end_share >= 0.5 else "MS"

def _future_dates(last_date: pd.Timestamp, horizon: int, freq: str,
                  week_anchor: int, month_anchor_rule: str):
    """
    Build a consistent future date range for forecasting, respecting
    weekly or monthly anchors. Keeps Prophet projections visually aligned
    with historical granularity.
    """
    if freq == "W":
        start = (last_date + pd.Timedelta(days=1))
        start += pd.offsets.Week(weekday=week_anchor)
        return pd.date_range(start=start, periods=horizon, freq=pd.offsets.Week(weekday=week_anchor))
    if freq == "M":
        if month_anchor_rule == "ME":
            first = (last_date + pd.offsets.MonthEnd(0)) + pd.offsets.MonthEnd(1)
            return pd.date_range(start=first, periods=horizon, freq=pd.offsets.MonthEnd(1))
        else:
            first = (last_date + pd.offsets.MonthBegin(1))
            return pd.date_range(start=first, periods=horizon, freq=pd.offsets.MonthBegin(1))
    return pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")



# -----------------------------------------------------------------------------
# Utility: Duplicate aggregation and metrics
# -----------------------------------------------------------------------------
def _aggregate_duplicates(df: pd.DataFrame, date_col: str, y_col: str,
                          group_cols: List[str], regs: List[str]) -> pd.DataFrame:
    """
    Safely aggregate duplicate (date + group) combinations.
    - Target values are summed.
    - Regressors are averaged.
    """
    use_cols = []
    for c in [date_col, y_col] + group_cols + regs:
        if c in df.columns and c not in use_cols:
            use_cols.append(c)
    df2 = df[use_cols].copy()
    df2 = df2.loc[:, ~df2.columns.duplicated()]
    df2.columns = pd.Index([str(c) for c in df2.columns]).rename(None)

    for g in group_cols:
        if g in df2.columns:
            # Normalize non-string group keys for consistent grouping
            df2[g] = df2[g].apply(lambda v: v if (isinstance(v, (str, int, float)) or pd.isna(v)) else str(v))

    agg_map = {y_col: "sum"}
    for r in regs:
        if r in df2.columns:
            agg_map[r] = "mean"

    gb = df2.groupby([c for c in [date_col] + group_cols if c in df2.columns],
                     dropna=False, as_index=False)
    return gb.agg(agg_map).sort_values(date_col).reset_index(drop=True)

def _metrics_from_arrays(true: np.ndarray, pred: np.ndarray) -> Metrics:
    """
    Compute simple regression metrics (MAE, MAPE, sMAPE, RMSE, Bias)
    for Prophet validation and lightweight backtests.
    """
    if len(true) == 0:
        return Metrics()
    err = pred - true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    denom = np.maximum(np.abs(true), 1e-9)
    mape = float(np.mean(abs_err / denom))
    smape = float(np.mean(2 * abs_err / (np.abs(true) + np.abs(pred) + 1e-9)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(-err))  # Positive bias ⇒ under-forecast
    return Metrics(mape=mape, mae=mae, smape=smape, rmse=rmse, bias=bias)


# -----------------------------------------------------------------------------
# Prophet model helpers
# -----------------------------------------------------------------------------
def _min_required_points(freq: str) -> int:
    """Define minimal history lengths to stabilize Prophet fits."""
    if freq == "D": return 14
    if freq == "W": return 26
    if freq == "M": return 12
    return 14

def _prepare_prophet_frame(dfg: pd.DataFrame, regressors: List[str]) -> pd.DataFrame:
    """
    Clean and rename data before fitting Prophet.
    Ensures required columns ('ds', 'y') exist, numeric targets are valid,
    and regressors are filled or dropped as needed.
    """
    d = dfg.rename(columns={"date": "ds", "y": "y"}).copy()
    d = d[["ds", "y"] + list(regressors or [])].sort_values("ds").reset_index(drop=True)
    mask_core = d[["ds", "y"]].isna().any(axis=1)
    if mask_core.any():
        logger.warning("Dropping %d rows with missing ds/y before Prophet fit", int(mask_core.sum()))
        d = d.loc[~mask_core]
    if d.empty:
        raise ValueError("No valid data after cleaning ds/y")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=["y"])
    if d.empty:
        raise ValueError("No valid numeric y after cleaning")
    if regressors:
        d[regressors] = d[regressors].fillna(0.0)
    return d

def _fit_prophet(df: pd.DataFrame, regressors: List[str], confidence: float,
                 country_holidays: Optional[str], freq: str,
                 prophet_params: Optional[Dict[str, Any]] = None) -> Prophet:
    """
    Configure, fit, and return a Prophet model tailored to the dataset.

    Dynamically tunes:
    - seasonality activation (weekly/yearly)
    - changepoint flexibility based on volatility
    - additive vs. multiplicative mode
    - optional holiday and regressor support
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if not {"date", "y"}.issubset(df.columns):
        raise ValueError("Dataframe must contain {'date','y'}")
    if not 0 < confidence < 1:
        raise ValueError("Confidence interval must be between 0 and 1")

    dfit = _prepare_prophet_frame(df, regressors)

    y = dfit["y"].astype(float).to_numpy()
    span_days = (pd.to_datetime(dfit["ds"]).max() - pd.to_datetime(dfit["ds"]).min()).days
    has_nonpos = np.any(y <= 0)
    cv = float(np.std(y) / max(np.mean(y), 1e-9))

    enable_weekly = (freq == "D") and (span_days >= 14)
    enable_yearly = (
        (freq == "D" and span_days >= 730) or
        (freq == "W" and len(dfit) >= 104) or
        (freq == "M" and len(dfit) >= 24)
    )
    seasonality_mode = "multiplicative" if (not has_nonpos) else "additive"

    # Changepoint flexibility based on coefficient of variation
    cps_auto = float(np.clip(0.05 + 0.6 * min(cv, 1.0), 0.05, 0.4))

    default_params = {
        "interval_width": float(confidence),
        "seasonality_mode": seasonality_mode,
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "changepoint_prior_scale": cps_auto,
        "mcmc_samples": 0,
        "uncertainty_samples": 300,
    }
    if prophet_params:
        default_params.update(prophet_params)

    m = Prophet(**default_params)

    # Enable relevant seasonalities
    if enable_weekly:
        m.add_seasonality(name="weekly", period=7, fourier_order=6 if len(dfit) >= 60 else 3)
    if enable_yearly:
        yearly_fo = 10 if (
            (freq == "D" and len(dfit) >= 365) or
            (freq == "W" and len(dfit) >= 156) or
            (freq == "M" and len(dfit) >= 36)
        ) else 5
        m.add_seasonality(name="yearly", period=365.25, fourier_order=yearly_fo)
        if freq == "W" and yearly_fo <= 8:
            m.add_seasonality(name="yearly_extra", period=365.25, fourier_order=yearly_fo + 2)
    if freq == "M" and len(dfit) >= 36:
        m.add_seasonality(name="quarterly", period=91.25, fourier_order=5)

    # Holidays and regressors
    if country_holidays:
        try:
            m.add_country_holidays(country_name=country_holidays)
        except Exception as e:
            logger.warning("Holidays not added for %s: %s", country_holidays, e)
    for r in [r for r in regressors if r in dfit.columns and r.lower() not in {"date", "ds", "y"}]:
        try:
            m.add_regressor(r, prior_scale=0.5)
        except TypeError:
            m.add_regressor(r)

    if np.nanstd(y) < 1e-8:
        raise ValueError("Series is near-constant; Prophet not suitable")

    m.fit(dfit)
    return m

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """Simple health check for container or monitoring probes."""
    return {"ok": True, "prophet": _PROPHET_AVAILABLE}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Main forecasting endpoint.
    Cleans, validates, and aggregates input data, then fits Prophet models
    per group (if grouping columns exist) to produce structured forecasts.
    """
    if not _PROPHET_AVAILABLE:
        raise HTTPException(status_code=500, detail="Prophet not installed.")
    if not req.data:
        raise HTTPException(status_code=400, detail="No data provided.")

    # --- Data preparation ------------------------------------------------------
    df = pd.DataFrame(req.data)
    if df.columns.duplicated().any():
        dupes = [str(c) for c, d in zip(df.columns, df.columns.duplicated()) if d]
    else:
        dupes = []
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df.columns = pd.Index([str(c).strip() for c in df.columns]).rename(None)

    warnings: List[str] = []
    if dupes:
        warnings.append(f"Dropped duplicate columns: {', '.join(sorted(set(dupes)))}")

    schema = req.schema
    params = req.params or ForecastParams()
    original_date = schema.date.strip()
    original_target = schema.target.strip()

    if original_date not in df.columns:
        raise HTTPException(status_code=400, detail=f"Missing required column: {original_date}")
    if original_target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Missing required column: {original_target}")

    # --- Column and regressor sanity checks ------------------------------------
    group_cols, seen = [], set()
    for g in (schema.group_by or []):
        g2 = str(g).strip()
        if g2 and g2 != original_target and g2 in df.columns and g2 not in seen:
            seen.add(g2)
            group_cols.append(g2)

    reg_ok = []
    for r in (schema.regressors or []):
        r2 = str(r).strip()
        if r2 and r2 != original_target and r2 in df.columns and r2 not in group_cols and r2 not in reg_ok:
            reg_ok.append(r2)
    regressors = reg_ok

    numeric_regs = []
    for r in regressors:
        if pd.api.types.is_numeric_dtype(df[r]):
            if df[r].nunique(dropna=True) > 5:
                numeric_regs.append(r)
            else:
                warnings.append(f"Ignoring constant/low-variance regressor: {r}")
        else:
            warnings.append(f"Ignoring non-numeric regressor: {r}")
    regressors = numeric_regs

    # --- Type coercion and cleanup ---------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df.rename(columns={original_date: "date", original_target: "y"}, inplace=True)
    df["date"] = _coerce_datetime(df["date"])
    bad = df["date"].isna().sum()
    if bad:
        warnings.append(f"Dropped {bad} rows with invalid dates.")
    df = df.dropna(subset=["date"])

    for c in ["y"] + regressors:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_y = df["y"].isna().sum()
    if bad_y:
        warnings.append(f"Dropped {bad_y} rows with non-numeric target.")
    df = df.dropna(subset=["y"])
    df = _aggregate_duplicates(df, "date", "y", group_cols, regressors)

    # --- Frequency and anchors -------------------------------------------------
    freq = (params.frequency or _infer_frequency(df["date"])).upper()
    if freq not in ["D", "W", "M"]:
        freq = "D"
    week_anchor = _weekly_anchor(df["date"]) if freq == "W" else 0
    month_anchor_rule = _monthly_anchor(df["date"]) if freq == "M" else "ME"
    detected_frequency = freq

        # --- Forecast configuration -------------------------------------------------
    horizon = int(params.horizon or 30)
    confidence = float(params.confidence or 0.8)

    # Split by groups if specified, else treat the dataset as a single series
    if group_cols:
        all_groups = [
            dict(zip(group_cols, vals))
            for vals in df[group_cols].drop_duplicates().itertuples(index=False, name=None)
        ]
    else:
        all_groups = [None]

    # Containers for all forecast results and evaluation stats
    out_rows: List[ForecastPoint] = []
    overall_abs_errs, overall_abs_pct, smapes, overall_rmses, biases = ([] for _ in range(5))
    per_group_used_model: List[str] = []

    # --- Helper: Regressor filling ---------------------------------------------
    def _fill_regs(dfg: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing regressor values within each group.
        Ensures Prophet never receives NaNs — forward fill, then backfill,
        finally defaulting to 0.0.
        """
        if not regressors:
            return dfg
        dfg = dfg.sort_values("date").reset_index(drop=True)
        dfg[regressors] = dfg[regressors].ffill().bfill()
        dfg[regressors] = dfg[regressors].fillna(0.0)
        return dfg

    # -----------------------------------------------------------------------------
    # Group-level forecasting loop
    # -----------------------------------------------------------------------------
    for grp in all_groups:
        if grp:
            # Mask dataset to the current group (multi-condition filter)
            mask = np.logical_and.reduce([df[k].astype(str) == str(v) for k, v in grp.items()])
            dfg = df.loc[mask, ["date", "y"] + regressors].sort_values("date").reset_index(drop=True)
        else:
            dfg = df[["date", "y"] + regressors].sort_values("date").reset_index(drop=True)

        # Emit history points (useful for charting original data)
        for _, r in dfg.iterrows():
            out_rows.append(ForecastPoint(
                date=r["date"].strftime("%Y-%m-%d"),
                yhat=float(r["y"]),
                group=grp,
                kind="history",
            ))

        # If insufficient data, skip and warn
        if len(dfg) < _min_required_points(freq):
            warnings.append(
                f"Group {grp or 'ALL'} skipped: insufficient history for Prophet "
                f"({len(dfg)} pts, freq={freq})."
            )
            per_group_used_model.append("skipped")
            continue

        # Ensure regressors are properly filled
        dfg = _fill_regs(dfg)

        # --- Model training -----------------------------------------------------
        try:
            model = _fit_prophet(dfg, regressors, confidence, params.country_holidays, freq)
        except Exception as e:
            warnings.append(f"Prophet failed for group {grp or 'ALL'}: {e}")
            per_group_used_model.append("skipped")
            continue

        # --- Forecast generation ------------------------------------------------
        last_date = dfg["date"].max()
        future_idx = _future_dates(last_date, horizon, freq, week_anchor, month_anchor_rule)

        # Build a "future" dataframe with the same regressor structure
        future = pd.DataFrame({"ds": future_idx})
        if regressors:
            latest = dfg.iloc[-1][regressors].to_dict()
            for r in regressors:
                future[r] = latest.get(r, 0.0)

        # Predict on both history + future to maintain Prophet’s internal components
        dfit = dfg.rename(columns={"date": "ds"})
        full_df = pd.concat(
            [dfit[["ds"] + regressors] if regressors else dfit[["ds"]], future],
            ignore_index=True,
        )
        fcst = model.predict(full_df)

        # Extract and sanitize only the forecasted part
        fut = fcst.iloc[len(dfit):].copy().reset_index(drop=True)
        for _, r in fut.iterrows():
            ds = pd.Timestamp(r["ds"]).strftime("%Y-%m-%d")
            yhat = float(r["yhat"]) if np.isfinite(r["yhat"]) else None
            lo = float(r["yhat_lower"]) if np.isfinite(r.get("yhat_lower", np.nan)) else None
            hi = float(r["yhat_upper"]) if np.isfinite(r.get("yhat_upper", np.nan)) else None
            if yhat is None:
                continue  # Skip invalid rows
            out_rows.append(ForecastPoint(
                date=ds,
                yhat=yhat,
                yhat_lower=lo,
                yhat_upper=hi,
                group=grp,
                kind="forecast",
            ))

        # -----------------------------------------------------------------------------
        # Lightweight backtesting (sanity check of model performance)
        # -----------------------------------------------------------------------------
        try:
            back_h = min(horizon, max(7, math.ceil(len(dfg) * 0.1)))
            if back_h < len(dfg) - 5:
                train = dfg.iloc[:-back_h]
                test = dfg.iloc[-back_h:]
                m2 = _fit_prophet(train, regressors, confidence, params.country_holidays, freq)
                pred = m2.predict(
                    test.rename(columns={"date": "ds"})[["ds"] + (regressors or [])]
                )["yhat"].values
                met = _metrics_from_arrays(test["y"].values, pred)

                # Collect metrics for overall aggregation
                if met.mae is not None: overall_abs_errs.append(met.mae)
                if met.mape is not None: overall_abs_pct.append(met.mape)
                if met.smape is not None: smapes.append(met.smape)
                if met.rmse is not None: overall_rmses.append(met.rmse)
                if met.bias is not None: biases.append(met.bias)

        except Exception as e:
            logger.warning("Backtest failed for group %s: %s", grp, e)

        per_group_used_model.append("prophet")

    # -----------------------------------------------------------------------------
    # Metrics and final response
    # -----------------------------------------------------------------------------
    metrics = Metrics(
        mae=float(np.nanmean(overall_abs_errs)) if overall_abs_errs else None,
        mape=float(np.nanmean(overall_abs_pct)) if overall_abs_pct else None,
        smape=float(np.nanmean(smapes)) if smapes else None,
        rmse=float(np.nanmean(overall_rmses)) if overall_rmses else None,
        bias=float(np.nanmean(biases)) if biases else None,
    )

    # If multiple models were used (some skipped), fallback to Prophet
    uniq = sorted(set([m for m in per_group_used_model if m != "skipped"])) or ["prophet"]
    used_model_final = uniq[0] if len(uniq) == 1 else "prophet"

    # Assemble the final structured response
    return ForecastResponse(
        used_model=used_model_final,
        detected_frequency=detected_frequency,
        warnings=warnings,
        metrics_overall=metrics,
        forecast=out_rows,
    )

# -----------------------------------------------------------------------------
# Local entrypoint (for manual testing)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # This allows `python main.py` to start a local dev server easily
    uvicorn.run(app, host="0.0.0.0", port=8000)
