"""
Weather Intelligence Platform — Flask Application
"""

import os
import logging
import time
import traceback
import functools
import json
from datetime import datetime
from collections import defaultdict

from flask import Flask, jsonify, request, render_template
from flask_caching import Cache

from scraper import weather_scraper, image_scraper, InputSanitizer
from ml_engine import ml_service
from theme_engine import theme_engine

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

DEFAULT_THEME = {
    "classification": "NORMAL",
    "theme_name": "Serene Skies",
    "emoji": "🌤️",
    "body_class": "theme-normal",
    "gradient_css": "linear-gradient(120deg, #0F2027, #203A43, #2C5364)",
    "color_primary": "#56AB91",
    "color_secondary": "#2AB87B",
    "color_accent": "#7FDBCA",
    "color_text": "#E8F5E9",
    "color_text_muted": "rgba(200,230,220,0.75)",
    "color_card_bg": "rgba(40,120,90,0.15)",
    "color_card_border": "rgba(80,180,140,0.22)",
    "glow_primary": "rgba(86,171,145,0.42)",
    "glow_secondary": "rgba(42,184,123,0.22)",
    "css_variables": {
        "--color-primary": "#56AB91",
        "--color-secondary": "#2AB87B",
        "--color-accent": "#7FDBCA",
        "--color-text": "#E8F5E9",
        "--color-text-muted": "rgba(200,230,220,0.75)",
        "--color-card-bg": "rgba(40,120,90,0.15)",
        "--color-card-border": "rgba(80,180,140,0.22)",
        "--glow-primary": "rgba(86,171,145,0.42)",
        "--glow-secondary": "rgba(42,184,123,0.22)",
        "--gradient-css": "linear-gradient(120deg, #0F2027, #203A43, #2C5364)",
    },
    "particles": {
        "type": "float", "count": 15, "speed_min": 5.0, "speed_max": 12.0,
        "size_min": 4.0, "size_max": 14.0, "opacity_min": 0.08, "opacity_max": 0.22,
        "color": "rgba(150,230,200,0.5)", "blur": 6.0,
    },
    "icons": ["🌤️", "🌿", "🍃"],
}

def make_default_ml(temperature=20.0):
    return {
        "classification": {
            "classification": "NORMAL",
            "confidence": 0.6,
            "class_probabilities": {"COLD": 0.2, "NORMAL": 0.6, "HOT": 0.2},
        },
        "clothing": {
            "recommendation": "T-Shirt",
            "confidence": 0.6,
            "reasoning": f"At {temperature}C, light clothing is appropriate.",
            "class_probabilities": {
                "Summer Wear": 0.1, "T-Shirt": 0.6,
                "Hoodie": 0.2, "Light Jacket": 0.07, "Heavy Jacket": 0.03,
            },
        },
        "pattern": {
            "pattern": "Mild & Pleasant", "confidence": 0.6,
            "characteristics": "Comfortable conditions.",
            "cluster_id": 3, "all_cluster_distances": {},
        },
        "forecast": {
            "hourly_temperatures": [round(temperature + i * 0.1, 1) for i in range(24)],
            "hourly_labels": [f"{h:02d}:00" for h in range(24)],
            "trend": "stable", "temperature_delta": 0.0,
            "min_forecast": temperature, "max_forecast": temperature, "confidence": 0.5,
        },
        "composite_confidence": 0.6,
        "training_metrics": {},
        "analysis_timestamp": datetime.now().isoformat(),
    }

def _safe_serialize(obj):
    try:
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
    except ImportError:
        pass
    if obj is None or isinstance(obj, (bool, str, int)): return obj
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'): return None
        return obj
    if isinstance(obj, dict): return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_safe_serialize(i) for i in obj]
    try: return _safe_serialize(obj.__dict__)
    except AttributeError: return str(obj)

def _safe_theme(theme_dict):
    result = {}
    for k, v in theme_dict.items():
        if isinstance(v, dict): result[k] = _safe_theme(v)
        elif isinstance(v, list): result[k] = [str(i) if not isinstance(i, (str, int, float, bool)) else i for i in v]
        elif isinstance(v, (str, int, float, bool)) or v is None: result[k] = v
        else: result[k] = str(v)
    return result


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "wi-dev-key"),
        DEBUG=False,
        JSON_SORT_KEYS=False,
        CACHE_TYPE="SimpleCache",
        CACHE_DEFAULT_TIMEOUT=1800,
        MAX_CONTENT_LENGTH=1 * 1024 * 1024,
    )

    cache = Cache(app)
    rate_store = defaultdict(list)

    def is_rate_limited(ip):
        now = time.time()
        rate_store[ip] = [t for t in rate_store[ip] if now - t < 60]
        if len(rate_store[ip]) >= 60: return True
        rate_store[ip].append(now)
        return False

    def rate_limit(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            if ip and is_rate_limited(ip):
                return jsonify({"status": "error", "error": "Rate limit exceeded."}), 429
            return f(*args, **kwargs)
        return wrapped

    def err(msg, code=400):
        return jsonify({"status": "error", "error": msg,
                        "timestamp": datetime.utcnow().isoformat() + "Z"}), code

    @app.after_request
    def add_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/weather", methods=["POST", "OPTIONS"])
    @rate_limit
    def weather_post():
        if request.method == "OPTIONS":
            return "", 204
        try:
            if not request.is_json:
                return err("Content-Type must be application/json")
            body = request.get_json(silent=True)
            if not body:
                return err("Empty or invalid JSON body.")
            city = str(body.get("city", "")).strip()
            if not city:
                return err("'city' field is required.")
            try:
                city = InputSanitizer.sanitize_city(city)
            except ValueError as e:
                return err(str(e), 400)

            logger.info(f"━━━ Request: city='{city}' ━━━")

            cache_key = f"weather_v3:{city.lower()}"
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"Cache HIT for {city}")
                return jsonify(cached), 200

            # Step 1: Weather
            logger.info("Step 1: Fetching weather...")
            weather = weather_scraper.fetch(city)
            logger.info(f"  temp={weather['temperature']}C hum={weather['humidity']}% wind={weather['wind_speed']}km/h src={weather['source']}")

            # Step 2: ML
            logger.info("Step 2: Running ML analysis...")
            ml_results = make_default_ml(float(weather["temperature"]))
            try:
                ml_results = ml_service.run_full_analysis(
                    temperature=float(weather["temperature"]),
                    humidity=float(weather["humidity"]),
                    wind_speed=float(weather["wind_speed"]),
                    feels_like=float(weather["feels_like"]),
                    current_hour=datetime.now().hour,
                )
                logger.info(f"  class={ml_results['classification']['classification']} clothing={ml_results['clothing']['recommendation']}")
            except Exception as e:
                logger.error(f"  ML FAILED (fallback used): {e}")
                logger.debug(traceback.format_exc())

            # Step 3: Classification + Pattern
            classification = "NORMAL"
            pattern = "Mild & Pleasant"
            try:
                classification = ml_results["classification"]["classification"]
                pattern = ml_results["pattern"]["pattern"]
            except (KeyError, TypeError):
                pass

            # Step 4: Theme
            logger.info(f"Step 3: Building theme ({classification}/{pattern})...")
            theme = DEFAULT_THEME.copy()
            try:
                raw_theme = theme_engine.get_theme(classification, pattern)
                theme = _safe_theme(raw_theme)
                logger.info(f"  theme='{theme.get('theme_name')}'")
            except Exception as e:
                logger.error(f"  THEME FAILED (default used): {e}")
                logger.debug(traceback.format_exc())

            # Step 5: Images
            logger.info("Step 4: Fetching images...")
            images = []
            try:
                images = image_scraper.fetch_images(city, classification, n_images=5)
                logger.info(f"  {len(images)} images fetched")
            except Exception as e:
                logger.error(f"  IMAGES FAILED (skipped): {e}")

            # Step 6: Build + verify response
            response_data = {
                "status": "success",
                "city": str(weather.get("city", city)),
                "country": str(weather.get("country", "")),
                "weather": _safe_serialize(weather),
                "ml": _safe_serialize(ml_results),
                "theme": theme,
                "images": images,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "cached": False,
            }

            # Test JSON serialization
            try:
                json.dumps(response_data)
            except (TypeError, ValueError) as e:
                logger.error(f"  JSON SERIALIZATION FAILED: {e}")
                logger.debug(traceback.format_exc())
                response_data["theme"] = DEFAULT_THEME
                # Try again
                try:
                    json.dumps(response_data)
                except Exception:
                    response_data["ml"] = make_default_ml(float(weather["temperature"]))

            cache.set(cache_key, response_data, timeout=1800)
            logger.info(f"Response ready for '{city}' ✓")
            return jsonify(response_data), 200

        except Exception as e:
            logger.error(f"UNHANDLED ERROR: {e}")
            logger.error(traceback.format_exc())
            return err(f"Server error: {str(e)}", 500)

    @app.route("/api/weather/<string:city>", methods=["GET"])
    @rate_limit
    def weather_get(city):
        try:
            city = InputSanitizer.sanitize_city(city)
        except ValueError as e:
            return err(str(e), 400)
        cache_key = f"wq:{city.lower()}"
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached), 200
        try:
            weather = weather_scraper.fetch(city)
        except Exception as e:
            return err(f"Failed to fetch weather: {str(e)}", 503)
        ml = make_default_ml(float(weather.get("temperature", 20)))
        try:
            ml = ml_service.run_full_analysis(
                temperature=float(weather["temperature"]),
                humidity=float(weather["humidity"]),
                wind_speed=float(weather["wind_speed"]),
                feels_like=float(weather["feels_like"]),
            )
        except Exception as e:
            logger.error(f"ML error: {e}")
        resp = {
            "status": "success",
            "city": weather["city"],
            "weather": _safe_serialize(weather),
            "classification": _safe_serialize(ml.get("classification", {})),
            "clothing": _safe_serialize(ml.get("clothing", {})),
            "forecast": _safe_serialize(ml.get("forecast", {})),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        cache.set(cache_key, resp, timeout=1800)
        return jsonify(resp), 200

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "healthy",
            "ml_ready": ml_service._initialized,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }), 200

    @app.route("/api/ml/metrics")
    def ml_metrics():
        if not ml_service._initialized:
            return err("ML not initialized.", 503)
        m = ml_service.training_metrics
        return jsonify({
            "status": "success",
            "metrics": {
                "forecaster": {"type": "LinearRegression", "r2_score": m.get("forecaster", {}).get("r2_score", 0)},
                "advisor": {"type": "LogisticRegression", "accuracy": m.get("advisor", {}).get("accuracy", 0), "classes": m.get("advisor", {}).get("classes", [])},
                "classifier": {"type": "LogisticRegression (Multi-class)", "accuracy": m.get("classifier", {}).get("accuracy", 0)},
                "cluster_engine": {"type": "KMeans", "n_clusters": m.get("cluster_engine", {}).get("n_clusters", 4)},
            }
        }), 200

    @app.route("/api/theme/<string:cls>")
    def get_theme(cls):
        cls = cls.upper()
        if cls not in ("HOT", "COLD", "NORMAL"):
            return err("Use HOT, COLD, or NORMAL.", 400)
        try:
            theme = _safe_theme(theme_engine.get_theme(cls, request.args.get("pattern")))
        except Exception as e:
            logger.error(f"Theme error: {e}")
            theme = DEFAULT_THEME
        return jsonify({"status": "success", "theme": theme}), 200

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return err("Endpoint not found.", 404)
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"500: {e}\n{traceback.format_exc()}")
        return err(f"Internal error: {str(e)}", 500)

    return app


app = create_app()


def initialize_ml():
    logger.info("=" * 50)
    logger.info("  Weather Intelligence — Starting Up")
    logger.info("=" * 50)
    try:
        metrics = ml_service.initialize()
        logger.info("✓ ML models trained and ready")
        logger.info(f"  R²={metrics.get('forecaster',{}).get('r2_score',0):.3f} | "
                    f"Acc={metrics.get('classifier',{}).get('accuracy',0):.3f}")
    except Exception as e:
        logger.error(f"ML init failed: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    initialize_ml()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"✓ Open http://localhost:{port}")
    logger.info("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
else:
    with app.app_context():
        initialize_ml()