
from flask import Flask, render_template, request, jsonify
import datetime as dt
from datetime import timedelta
import os

import joblib
import numpy as np
import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry

import tensorflow as tf
from tensorflow.keras.models import load_model

load_dotenv()
app = Flask(__name__)

try:
    from chatbot import load_thesis, ask_thesis, get_status as chatbot_status
    load_thesis()
except ImportError as e:
    print(f"[WARN] Chatbot module not available: {e}")
    ask_thesis = None
    chatbot_status = None

TIMEZONE = "Asia/Manila"

MODEL_PATH = "LSTM_Weather_Forecast_Direct2_x7.keras"
X_SCALER_PATH = "x_scaler_x7.pkl"
Y_SCALER_PATH = "y_scaler_x7.pkl"

MODEL_INPUT_STEPS = 7
TARGET_HORIZON = 2

TARGET_COLS_RAW = [
    "temperature_2m_min",
    "temperature_2m_max",
    "rain_sum",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "sunshine_duration",
]

FEATURE_COLS_MODEL = [
    "temperature_2m_min",
    "temperature_2m_max",
    "rain_sum_log1p",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "sunshine_duration",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]

TARGET_COLS_MODEL = [
    "temperature_2m_min",
    "temperature_2m_max",
    "rain_sum_log1p",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "sunshine_duration",
]

DEFAULT_LOCATION = {
    "name": "Iloilo City",
    "lat": 10.6969,
    "lon": 122.5644,
    "elevation": 8.0,
    "admin1": "Western Visayas",
    "admin2": "Iloilo",
    "admin3": "Iloilo City",
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

@tf.keras.utils.register_keras_serializable()
class R2Score3D(tf.keras.metrics.Metric):
    def __init__(self, name="r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.r2_metric = tf.keras.metrics.R2Score()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        self.r2_metric.update_state(y_true_flat, y_pred_flat, sample_weight=sample_weight)

    def result(self):
        return self.r2_metric.result()

    def reset_state(self):
        self.r2_metric.reset_state()


def load_artifacts():
    model = None
    x_scaler = None
    y_scaler = None

    try:
        model = load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"R2Score3D": R2Score3D},
        )
        print(f"[OK] Loaded model: {MODEL_PATH}")
        print(f"[OK] Model input shape : {model.input_shape}")
        print(f"[OK] Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load model '{MODEL_PATH}': {e}")

    try:
        x_scaler = joblib.load(X_SCALER_PATH)
        print(f"[OK] Loaded X scaler: {X_SCALER_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load X scaler '{X_SCALER_PATH}': {e}")

    try:
        y_scaler = joblib.load(Y_SCALER_PATH)
        print(f"[OK] Loaded Y scaler: {Y_SCALER_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load Y scaler '{Y_SCALER_PATH}': {e}")

    return model, x_scaler, y_scaler


model, x_scaler, y_scaler = load_artifacts()


def convert_to_native_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [convert_to_native_types(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native_types(x) for x in obj]
    return obj


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp_non_negative(value):
    return float(max(0.0, safe_float(value, 0.0)))


def search_locations(query):
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": query,
        "count": 10,
        "language": "en",
        "countryCode": "PH",
    }

    try:
        response = requests.get(geo_url, params=geo_params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = []
        for place in data.get("results", []):
            if place.get("admin1", "") == "Western Visayas":
                results.append({
                    "name": place.get("name", ""),
                    "lat": float(place.get("latitude", 0)),
                    "lon": float(place.get("longitude", 0)),
                    "elevation": float(place.get("elevation", 0)) if place.get("elevation") is not None else "N/A",
                    "admin1": place.get("admin1", ""),
                    "admin2": place.get("admin2", ""),
                    "admin3": place.get("admin3", ""),
                })
        return results
    except Exception as e:
        print(f"[WARN] search_locations failed: {e}")
        return []


def fetch_openmeteo_daily_archive(start_date, end_date, location_data):
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location_data["lat"],
        "longitude": location_data["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_min",
            "temperature_2m_max",
            "rain_sum",
            "wind_speed_10m_mean",
            "relative_humidity_2m_mean",
            "dew_point_2m_mean",
            "sunshine_duration",
        ],
        "timezone": TIMEZONE,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
        "temperature_2m_min": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
        "rain_sum": daily.Variables(2).ValuesAsNumpy(),
        "wind_speed_10m_mean": daily.Variables(3).ValuesAsNumpy(),
        "relative_humidity_2m_mean": daily.Variables(4).ValuesAsNumpy(),
        "dew_point_2m_mean": daily.Variables(5).ValuesAsNumpy(),
        "sunshine_duration": daily.Variables(6).ValuesAsNumpy(),
    })
    return df


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    if index.tz is not None:
        idx_local = index.tz_convert(TIMEZONE)
    else:
        idx_local = index.tz_localize(TIMEZONE)

    day_of_year = idx_local.dayofyear.astype(float)
    month = idx_local.month.astype(float)

    cal = pd.DataFrame(index=index)
    cal["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    cal["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    cal["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    cal["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return cal


def preprocess_daily_weather(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date")

    df["sunshine_duration"] = df["sunshine_duration"] / 3600.0
    df["sunshine_duration"] = df["sunshine_duration"].interpolate(method="time").bfill().ffill()
    df["rain_sum"] = df["rain_sum"].clip(lower=0)
    df["sunshine_duration"] = df["sunshine_duration"].clip(lower=0)
    df["rain_sum_log1p"] = np.log1p(df["rain_sum"])

    cal = add_calendar_features(df.index)
    df = pd.concat([df, cal], axis=1)
    df = df.dropna().copy()
    return df


def get_sample_data(start_date, end_date):
    rng = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
    raw_df = pd.DataFrame({
        "date": rng,
        "temperature_2m_min": np.random.uniform(23, 25, len(rng)),
        "temperature_2m_max": np.random.uniform(28, 32, len(rng)),
        "rain_sum": np.random.uniform(0, 10, len(rng)),
        "wind_speed_10m_mean": np.random.uniform(5, 15, len(rng)),
        "relative_humidity_2m_mean": np.random.uniform(80, 95, len(rng)),
        "dew_point_2m_mean": np.random.uniform(23, 25, len(rng)),
        "sunshine_duration": np.random.uniform(6, 12, len(rng)) * 3600.0,
    })
    return preprocess_daily_weather(raw_df)


def fetch_weather_data(start_date, end_date, location_data=DEFAULT_LOCATION):
    try:
        raw_df = fetch_openmeteo_daily_archive(start_date, end_date, location_data)
        return preprocess_daily_weather(raw_df)
    except Exception as e:
        print(f"[WARN] fetch_weather_data fallback to sample data: {e}")
        return get_sample_data(start_date, end_date)


def prepare_model_input(df_model: pd.DataFrame) -> np.ndarray:
    if len(df_model) < MODEL_INPUT_STEPS:
        raise ValueError(f"Need at least {MODEL_INPUT_STEPS} days of data, got {len(df_model)}.")

    x_values = df_model[FEATURE_COLS_MODEL].values
    x_scaled = x_scaler.transform(x_values)
    return x_scaled[-MODEL_INPUT_STEPS:].reshape(1, MODEL_INPUT_STEPS, len(FEATURE_COLS_MODEL))


def inverse_target_postprocess(y_pred_target_raw: np.ndarray, target_cols_transformed):
    out = pd.DataFrame(y_pred_target_raw, columns=target_cols_transformed)

    if "rain_sum_log1p" in out.columns:
        out["rain_sum"] = np.expm1(out["rain_sum_log1p"]).clip(lower=0)
        out = out.drop(columns=["rain_sum_log1p"])

    if "sunshine_duration" in out.columns:
        out["sunshine_duration"] = out["sunshine_duration"].clip(lower=0)

    ordered = out.reindex(columns=TARGET_COLS_RAW)
    return ordered.values


def inverse_targets_to_original_units(y_scaled_seq: np.ndarray) -> np.ndarray:
    shape = y_scaled_seq.shape
    y_flat = y_scaled_seq.reshape(-1, shape[-1])
    y_unscaled_transformed = y_scaler.inverse_transform(y_flat)
    y_raw = inverse_target_postprocess(y_unscaled_transformed, TARGET_COLS_MODEL)
    return y_raw.reshape(shape[0], shape[1], len(TARGET_COLS_RAW))


def generate_predictions(sequence: np.ndarray) -> np.ndarray:
    if model is None or x_scaler is None or y_scaler is None:
        raise RuntimeError("Model or scalers are not loaded.")

    y_pred_scaled = model.predict(sequence, verbose=0)

    if y_pred_scaled.ndim == 2 and y_pred_scaled.shape[1] == TARGET_HORIZON * len(TARGET_COLS_MODEL):
        y_pred_scaled = y_pred_scaled.reshape(1, TARGET_HORIZON, len(TARGET_COLS_MODEL))

    if y_pred_scaled.ndim != 3:
        raise ValueError(f"Unexpected prediction shape: {y_pred_scaled.shape}")

    y_pred_raw = inverse_targets_to_original_units(y_pred_scaled)
    y_pred_raw[..., 2] = np.maximum(y_pred_raw[..., 2], 0.0)
    return y_pred_raw[0]


def get_historical_weather(day, month, years_range=range(2014, 2026), location_data=DEFAULT_LOCATION):
    historical_rows = []

    for year in years_range:
        try:
            target_date = dt.date(year, month, day)

            cache_session = requests_cache.CachedSession(".cache", expire_after=86400)
            retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)

            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": location_data["lat"],
                "longitude": location_data["lon"],
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "daily": [
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "rain_sum",
                    "wind_speed_10m_mean",
                    "relative_humidity_2m_mean",
                    "dew_point_2m_mean",
                    "sunshine_duration",
                ],
                "timezone": TIMEZONE,
            }

            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()

            historical_rows.append({
                "year": int(year),
                "temperature_2m_min": float(daily.Variables(0).ValuesAsNumpy()[0]),
                "temperature_2m_max": float(daily.Variables(1).ValuesAsNumpy()[0]),
                "rain_sum": clamp_non_negative(daily.Variables(2).ValuesAsNumpy()[0]),
                "wind_speed_10m_mean": float(daily.Variables(3).ValuesAsNumpy()[0]),
                "relative_humidity_2m_mean": float(daily.Variables(4).ValuesAsNumpy()[0]),
                "dew_point_2m_mean": float(daily.Variables(5).ValuesAsNumpy()[0]),
                "sunshine_duration": float(daily.Variables(6).ValuesAsNumpy()[0]),
            })
        except Exception as e:
            print(f"[WARN] Historical fetch failed for {year}-{month:02d}-{day:02d}: {e}")

    return pd.DataFrame(historical_rows)


def build_dashboard_context(location_data, start_date=None, end_date=None):
    if end_date is None:
        end_date_obj = dt.date.today() - dt.timedelta(days=1)
    else:
        end_date_obj = pd.to_datetime(end_date).date()- dt.timedelta(days=1)

    if start_date is None:
        start_date_obj = end_date_obj - dt.timedelta(days=MODEL_INPUT_STEPS - 1)
    else:
        start_date_obj = pd.to_datetime(start_date).date()

    weather_df = fetch_weather_data(
        start_date_obj.strftime("%Y-%m-%d"),
        end_date_obj.strftime("%Y-%m-%d"),
        location_data,
    )

    if len(weather_df) < MODEL_INPUT_STEPS:
        raise ValueError(f"Need at least {MODEL_INPUT_STEPS} days of history, got {len(weather_df)}.")

    historical_data = []
    for date_idx, row in weather_df.tail(MODEL_INPUT_STEPS).iterrows():
        historical_data.append({
            "date": date_idx.date().isoformat(),
            "temperature_2m_min": float(round(row["temperature_2m_min"], 3)),
            "temperature_2m_max": float(round(row["temperature_2m_max"], 3)),
            "rain_sum": float(round(clamp_non_negative(row["rain_sum"]), 3)),
            "wind_speed_10m_mean": float(round(row["wind_speed_10m_mean"], 3)),
            "relative_humidity_2m_mean": float(round(row["relative_humidity_2m_mean"], 3)),
            "dew_point_2m_mean": float(round(row["dew_point_2m_mean"], 3)),
            "sunshine_duration": float(round(row["sunshine_duration"] * 3600.0, 3)),
        })

    try:
        sequence = prepare_model_input(weather_df)
        predictions = generate_predictions(sequence)
    except Exception as e:
        print(f"[WARN] Prediction failed; using fallback predictions: {e}")
        predictions = np.array([
            [24.0, 30.0, 0.0, 8.0, 84.0, 23.5, 8.0],
            [24.2, 30.4, 0.0, 8.5, 83.0, 23.6, 8.5],
        ])

    forecast_data = []
    last_date = weather_df.index[-1].date()
    future_dates = [last_date + timedelta(days=i + 1) for i in range(TARGET_HORIZON)]

    for i in range(min(TARGET_HORIZON, len(predictions))):
        pred = predictions[i]
        forecast_data.append({
            "date": future_dates[i],
            "temperature_2m_min": float(round(pred[0], 3)),
            "temperature_2m_max": float(round(pred[1], 3)),
            "rain_sum": float(round(clamp_non_negative(pred[2]), 3)),
            "wind_speed_10m_mean": float(round(pred[3], 3)),
            "relative_humidity_2m_mean": float(round(pred[4], 3)),
            "dew_point_2m_mean": float(round(pred[5], 3)),
            "sunshine_duration": float(round(max(0.0, pred[6]) * 3600.0, 3)),
        })

    historical_data_list = []
    for forecast_day in forecast_data:
        hist_df = get_historical_weather(
            forecast_day["date"].day,
            forecast_day["date"].month,
            location_data=location_data,
        )

        rows = []
        if not hist_df.empty:
            for _, row in hist_df.iterrows():
                rows.append({
                    "year": int(row["year"]),
                    "temperature_2m_min": float(row["temperature_2m_min"]),
                    "temperature_2m_max": float(row["temperature_2m_max"]),
                    "rain_sum": float(clamp_non_negative(row["rain_sum"])),
                    "wind_speed_10m_mean": float(row["wind_speed_10m_mean"]),
                    "relative_humidity_2m_mean": float(row["relative_humidity_2m_mean"]),
                    "dew_point_2m_mean": float(row["dew_point_2m_mean"]),
                    "sunshine_duration": float(row["sunshine_duration"]),
                })
        historical_data_list.append(rows)

    context = {
        "historical_data": historical_data,
        "forecast_data": forecast_data,
        "historical_forecast_1": historical_data_list[0] if len(historical_data_list) > 0 else [],
        "historical_forecast_2": historical_data_list[1] if len(historical_data_list) > 1 else [],
        "month_names": MONTH_NAMES,
        "start_date": start_date_obj.strftime("%Y-%m-%d"),
        "end_date": end_date_obj.strftime("%Y-%m-%d"),
        "days_to_predict": TARGET_HORIZON,
        "selected_location": location_data,
        "today_date": dt.date.today().strftime("%Y-%m-%d"),
        "model_window_days": MODEL_INPUT_STEPS,
        "forecast_strategy": "Direct multi-step (2-day output in one pass)",
    }
    return convert_to_native_types(context)


@app.route("/search_locations")
def search_locations_route():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify([])
    return jsonify(convert_to_native_types(search_locations(query)))


@app.route("/")
def index():
    try:
        context = build_dashboard_context(DEFAULT_LOCATION)
        return render_template("index.html", **context)
    except Exception as e:
        print(f"[ERROR] index route failed: {e}")
        return render_template(
            "index.html",
            selected_location=DEFAULT_LOCATION,
            today_date=dt.date.today().strftime("%Y-%m-%d"),
            model_window_days=MODEL_INPUT_STEPS,
            days_to_predict=TARGET_HORIZON,
            forecast_strategy="Direct multi-step (2-day output in one pass)",
            historical_data=[],
            forecast_data=[],
            historical_forecast_1=[],
            historical_forecast_2=[],
            month_names=MONTH_NAMES,
            start_date=dt.date.today().strftime("%Y-%m-%d"),
            end_date=dt.date.today().strftime("%Y-%m-%d"),
        )


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    try:
        if request.method == "POST":
            location_data = {
                "name": request.form.get("location_name", DEFAULT_LOCATION["name"]),
                "lat": safe_float(request.form.get("location_lat"), DEFAULT_LOCATION["lat"]),
                "lon": safe_float(request.form.get("location_lon"), DEFAULT_LOCATION["lon"]),
                "elevation": request.form.get("location_elevation", DEFAULT_LOCATION["elevation"]),
                "admin1": request.form.get("location_admin1", DEFAULT_LOCATION["admin1"]) or DEFAULT_LOCATION["admin1"],
                "admin2": request.form.get("location_admin2", DEFAULT_LOCATION["admin2"]) or DEFAULT_LOCATION["admin2"],
                "admin3": request.form.get("location_admin3", DEFAULT_LOCATION["admin3"]) or DEFAULT_LOCATION["admin3"],
            }
            start_date = request.form.get("start_date")
            end_date = request.form.get("end_date")
        else:
            location_data = DEFAULT_LOCATION
            start_date = request.args.get("start_date")
            end_date = request.args.get("end_date")

        context = build_dashboard_context(location_data=location_data, start_date=start_date, end_date=end_date)
        return render_template("index.html", **context)
    except Exception as e:
        print(f"[ERROR] forecast route failed: {e}")
        return f"Error: {str(e)}", 500


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        if ask_thesis is None:
            return jsonify({
                "answer": "The chatbot module is not available. Please install the required dependencies.",
                "sources": [],
            }), 503

        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()

        if not question:
            return jsonify({"answer": "Please ask a question.", "sources": []}), 400

        if len(question) > 1000:
            return jsonify({
                "answer": "Your question is too long. Please keep it under 1000 characters.",
                "sources": [],
            }), 400

        result = ask_thesis(question)
        return jsonify(result)
    except Exception:
        app.logger.exception("Unhandled error in /api/chat")
        return jsonify({
            "answer": "Server error while processing your message. Please check terminal logs.",
            "sources": [],
        }), 500


@app.route("/api/chatbot-status")
def chatbot_status_route():
    if chatbot_status is None:
        return jsonify({
            "initialized": False,
            "api_key_set": False,
            "error": "Chatbot module not installed",
        })
    return jsonify(chatbot_status())


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
