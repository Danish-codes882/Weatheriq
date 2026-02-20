"""
Weather Intelligence ML Engine
================================
Production-grade machine learning service for weather analysis and prediction.

Models:
    - LinearRegressionForecaster: 24h temperature trend prediction
    - ClothingAdvisor (LogisticRegression): Clothing recommendation engine
    - WeatherClassifier: HOT / COLD / NORMAL classification
    - WeatherClusterEngine (KMeans): Pattern clustering

All models are trained dynamically at startup on synthetically generated
historical weather data that closely mirrors real-world distributions.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLOTHING_LABELS = ["Summer Wear", "T-Shirt", "Hoodie", "Light Jacket", "Heavy Jacket"]
WEATHER_CLASS_LABELS = ["COLD", "NORMAL", "HOT"]
CLUSTER_NAMES = {
    0: "Dry Heat",
    1: "Humid Heat",
    2: "Cold & Windy",
    3: "Mild & Pleasant",
}

# Temperature thresholds (°C)
HOT_THRESHOLD = 30.0
COLD_THRESHOLD = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


class WeatherDataGenerator:
    """
    Generates realistic synthetic historical weather datasets for model training.
    Mimics seasonal patterns, diurnal cycles, and geographic variation.
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)

    def generate_historical_dataset(self) -> pd.DataFrame:
        """Generate a comprehensive historical weather dataset."""
        logger.info(f"Generating synthetic dataset with {self.n_samples} samples...")

        # Generate timestamps spanning 2 years
        start_date = datetime.now() - timedelta(days=730)
        timestamps = [
            start_date + timedelta(hours=i * (730 * 24 / self.n_samples))
            for i in range(self.n_samples)
        ]

        # Seasonal temperature base (sine wave over year)
        day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
        hour_of_day = np.array([t.hour for t in timestamps])

        # Base temperature with seasonal variation
        seasonal_component = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        diurnal_component = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        base_temp = 18 + seasonal_component + diurnal_component

        # Add realistic noise
        temperature = base_temp + np.random.normal(0, 3, self.n_samples)
        temperature = np.clip(temperature, -15, 48)

        # Humidity inversely correlated with temperature + noise
        humidity = 85 - 0.8 * temperature + np.random.normal(0, 10, self.n_samples)
        humidity = np.clip(humidity, 10, 100)

        # Wind speed - slightly higher in cold/transitional periods
        wind_speed = (
            np.abs(np.random.normal(15, 8, self.n_samples))
            + np.clip((20 - temperature) * 0.3, 0, 10)
        )
        wind_speed = np.clip(wind_speed, 0, 80)

        # Feels like temperature (wind chill / heat index approximation)
        feels_like = self._calculate_feels_like(temperature, humidity, wind_speed)

        # Pressure with small variation
        pressure = 1013 + np.random.normal(0, 15, self.n_samples)
        pressure = np.clip(pressure, 960, 1060)

        # UV Index
        uv_index = np.clip(
            (temperature - 5) / 5 + np.random.normal(0, 1, self.n_samples), 0, 11
        )

        # Cloud cover
        cloud_cover = np.clip(
            50 - seasonal_component + np.random.normal(0, 20, self.n_samples), 0, 100
        )

        # Derived labels
        weather_class = self._classify_weather(temperature)
        clothing = self._recommend_clothing(temperature, wind_speed, humidity)
        cluster_raw = self._assign_cluster(temperature, humidity, wind_speed)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature": temperature.round(1),
                "humidity": humidity.round(1),
                "wind_speed": wind_speed.round(1),
                "feels_like": feels_like.round(1),
                "pressure": pressure.round(1),
                "uv_index": uv_index.round(1),
                "cloud_cover": cloud_cover.round(1),
                "weather_class": weather_class,
                "clothing": clothing,
                "cluster_raw": cluster_raw,
                "day_of_year": day_of_year,
                "hour_of_day": hour_of_day,
            }
        )

        logger.info("Synthetic dataset generated successfully.")
        return df

    def generate_hourly_sequence(
        self,
        base_temp: float,
        base_humidity: float,
        base_wind: float,
        n_hours: int = 48,
    ) -> pd.DataFrame:
        """
        Generate an hourly sequence for a specific weather condition.
        Used as training data for the forecasting model.
        """
        hours = np.arange(n_hours)

        # Temperature follows a diurnal sine wave around base
        temp_variation = 4 * np.sin(2 * np.pi * (hours - 6) / 24)
        temperatures = base_temp + temp_variation + np.random.normal(0, 1.5, n_hours)

        humidity_variation = -3 * np.sin(2 * np.pi * (hours - 6) / 24)
        humidities = (
            base_humidity + humidity_variation + np.random.normal(0, 3, n_hours)
        )
        humidities = np.clip(humidities, 10, 100)

        wind_speeds = base_wind + np.random.normal(0, 2, n_hours)
        wind_speeds = np.clip(wind_speeds, 0, 80)

        return pd.DataFrame(
            {
                "hour": hours,
                "temperature": temperatures,
                "humidity": humidities,
                "wind_speed": wind_speeds,
            }
        )

    @staticmethod
    def _calculate_feels_like(
        temp: np.ndarray, humidity: np.ndarray, wind: np.ndarray
    ) -> np.ndarray:
        """Simplified heat index / wind chill calculation."""
        # Heat index for hot weather
        heat_index = (
            -8.78469475556
            + 1.61139411 * temp
            + 2.33854883889 * humidity
            - 0.14611605 * temp * humidity
            - 0.012308094 * temp**2
            - 0.0164248277778 * humidity**2
            + 0.002211732 * temp**2 * humidity
            + 0.00072546 * temp * humidity**2
            - 0.000003582 * temp**2 * humidity**2
        )
        # Wind chill for cold weather
        wind_chill = (
            13.12
            + 0.6215 * temp
            - 11.37 * (wind**0.16)
            + 0.3965 * temp * (wind**0.16)
        )
        # Blend based on temperature
        feels = np.where(temp >= 27, heat_index, np.where(temp <= 10, wind_chill, temp))
        return np.clip(feels, -25, 55)

    @staticmethod
    def _classify_weather(temp: np.ndarray) -> np.ndarray:
        return np.where(temp >= HOT_THRESHOLD, "HOT", np.where(temp <= COLD_THRESHOLD, "COLD", "NORMAL"))

    @staticmethod
    def _recommend_clothing(
        temp: np.ndarray, wind: np.ndarray, humidity: np.ndarray
    ) -> np.ndarray:
        effective = temp - wind * 0.1
        clothing = np.where(
            effective >= 28,
            "Summer Wear",
            np.where(
                effective >= 20,
                "T-Shirt",
                np.where(
                    effective >= 12,
                    "Hoodie",
                    np.where(effective >= 4, "Light Jacket", "Heavy Jacket"),
                ),
            ),
        )
        return clothing

    @staticmethod
    def _assign_cluster(
        temp: np.ndarray, humidity: np.ndarray, wind: np.ndarray
    ) -> np.ndarray:
        cluster = np.where(
            (temp >= 28) & (humidity < 50),
            0,  # Dry Heat
            np.where(
                (temp >= 28) & (humidity >= 50),
                1,  # Humid Heat
                np.where(
                    (temp < 15) & (wind > 20),
                    2,  # Cold & Windy
                    3,  # Mild & Pleasant
                ),
            ),
        )
        return cluster


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE FORECASTER
# ─────────────────────────────────────────────────────────────────────────────


class TemperatureForecaster:
    """
    LinearRegression-based 24-hour temperature trend forecaster.

    Features:
        - Current temperature, humidity, wind speed, hour of day
        - Polynomial-like feature engineering for non-linear capture
    """

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.r2_score = 0.0
        self.feature_names = [
            "temperature",
            "humidity",
            "wind_speed",
            "hour_of_day",
            "temp_humidity_interaction",
            "temp_wind_interaction",
            "temp_squared",
        ]

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features for regression."""
        temp = df["temperature"].values
        hum = df["humidity"].values
        wind = df["wind_speed"].values
        hour = df["hour_of_day"].values if "hour_of_day" in df.columns else np.zeros(len(df))

        X = np.column_stack([
            temp,
            hum,
            wind,
            hour,
            temp * hum / 100,          # interaction
            temp * wind / 50,           # interaction
            temp ** 2 / 100,            # quadratic
        ])
        return X

    def train(self, df: pd.DataFrame) -> dict:
        """Train the model on historical data."""
        logger.info("Training TemperatureForecaster...")

        # Target: next-hour temperature (shift by 1)
        df_train = df.copy()
        df_train["target_temp"] = df_train["temperature"].shift(-1)
        df_train = df_train.dropna()

        X = self._build_features(df_train)
        y = df_train["target_temp"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.r2_score = float(r2_score(y_test, y_pred))
        self.is_trained = True

        logger.info(f"TemperatureForecaster trained. R² = {self.r2_score:.4f}")
        return {"r2_score": self.r2_score, "n_samples": len(df_train)}

    def forecast_24h(
        self,
        current_temp: float,
        current_humidity: float,
        current_wind: float,
        current_hour: int,
    ) -> dict:
        """Generate 24-hour temperature forecast."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        hourly_temps = []
        hourly_hours = []
        temp = current_temp
        hum = current_humidity
        wind = current_wind
        hour = current_hour

        for i in range(24):
            row = pd.DataFrame([{
                "temperature": temp,
                "humidity": hum,
                "wind_speed": wind,
                "hour_of_day": hour % 24,
            }])
            X = self._build_features(row)
            X_scaled = self.scaler.transform(X)
            pred = float(self.model.predict(X_scaled)[0])

            # Add diurnal variation modulation
            diurnal = 2.5 * np.sin(2 * np.pi * ((hour + 1) % 24 - 6) / 24)
            pred = pred + diurnal * 0.3 + np.random.normal(0, 0.4)

            hourly_temps.append(round(pred, 1))
            hourly_hours.append(f"{(hour + i) % 24:02d}:00")
            temp = pred
            hour = (hour + 1) % 24

        trend = "rising" if hourly_temps[-1] > hourly_temps[0] else "falling"
        delta = round(hourly_temps[-1] - hourly_temps[0], 1)

        return {
            "hourly_temperatures": hourly_temps,
            "hourly_labels": hourly_hours,
            "trend": trend,
            "temperature_delta": delta,
            "min_forecast": min(hourly_temps),
            "max_forecast": max(hourly_temps),
            "confidence": round(min(0.99, max(0.6, self.r2_score * 1.1)), 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLOTHING ADVISOR
# ─────────────────────────────────────────────────────────────────────────────


class ClothingAdvisor:
    """
    LogisticRegression-based clothing recommendation engine.

    Outputs one of: Heavy Jacket, Light Jacket, Hoodie, T-Shirt, Summer Wear
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.accuracy = 0.0
        self.class_names = CLOTHING_LABELS

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Feature set for clothing recommendation."""
        temp = df["temperature"].values
        wind = df["wind_speed"].values
        hum = df["humidity"].values

        # Effective temperature (feels-like approximation)
        eff_temp = temp - wind * 0.1 + hum * 0.02

        return np.column_stack([temp, wind, hum, eff_temp])

    def train(self, df: pd.DataFrame) -> dict:
        """Train clothing advisor on historical data."""
        logger.info("Training ClothingAdvisor...")

        X = self._build_features(df)
        y_raw = df["clothing"].values
        y = self.label_encoder.fit_transform(y_raw)
        self.class_names = list(self.label_encoder.classes_)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        self.accuracy = float(accuracy_score(y_test, y_pred))
        self.is_trained = True

        logger.info(f"ClothingAdvisor trained. Accuracy = {self.accuracy:.4f}")
        return {"accuracy": self.accuracy, "classes": self.class_names}

    def recommend(
        self, temperature: float, wind_speed: float, humidity: float
    ) -> dict:
        """Get clothing recommendation for given weather conditions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        eff_temp = temperature - wind_speed * 0.1 + humidity * 0.02
        X = np.array([[temperature, wind_speed, humidity, eff_temp]])
        X_scaled = self.pipeline.named_steps["scaler"].transform(X)

        pred_idx = self.pipeline.named_steps["model"].predict(X_scaled)[0]
        proba = self.pipeline.named_steps["model"].predict_proba(X_scaled)[0]

        recommendation = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        # All class probabilities
        class_probs = {
            self.class_names[i]: round(float(proba[i]), 3)
            for i in range(len(self.class_names))
        }

        return {
            "recommendation": recommendation,
            "confidence": round(confidence, 3),
            "class_probabilities": class_probs,
            "reasoning": self._generate_reasoning(recommendation, temperature, wind_speed),
        }

    @staticmethod
    def _generate_reasoning(clothing: str, temp: float, wind: float) -> str:
        reasons = {
            "Summer Wear": f"At {temp}°C, it's hot enough for minimal clothing.",
            "T-Shirt": f"At {temp}°C, light clothing is appropriate.",
            "Hoodie": f"At {temp}°C, a hoodie keeps you comfortable.",
            "Light Jacket": f"With {temp}°C and {wind} km/h wind, a light jacket is recommended.",
            "Heavy Jacket": f"Only {temp}°C — dress warmly with a heavy jacket.",
        }
        return reasons.get(clothing, f"Appropriate for current conditions ({temp}°C).")


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────


class WeatherClassifier:
    """
    Multi-class classifier for HOT / COLD / NORMAL weather classification.
    Uses LogisticRegression with engineered features.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                C=2.0,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.accuracy = 0.0

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        temp = df["temperature"].values
        hum = df["humidity"].values
        wind = df["wind_speed"].values
        feels = df.get("feels_like", df["temperature"]).values if isinstance(df, pd.DataFrame) else temp

        return np.column_stack([
            temp,
            hum,
            wind,
            feels,
            temp * hum / 100,
            (temp - feels),
        ])

    def train(self, df: pd.DataFrame) -> dict:
        """Train the weather classifier."""
        logger.info("Training WeatherClassifier...")

        # Ensure feels_like column
        if "feels_like" not in df.columns:
            df = df.copy()
            df["feels_like"] = df["temperature"]

        X = self._build_features(df)
        y_raw = df["weather_class"].values
        y = self.label_encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        self.accuracy = float(accuracy_score(y_test, y_pred))
        self.is_trained = True

        logger.info(f"WeatherClassifier trained. Accuracy = {self.accuracy:.4f}")
        return {"accuracy": self.accuracy, "classes": list(self.label_encoder.classes_)}

    def classify(
        self,
        temperature: float,
        humidity: float,
        wind_speed: float,
        feels_like: float,
    ) -> dict:
        """Classify weather conditions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X = np.array([[
            temperature,
            humidity,
            wind_speed,
            feels_like,
            temperature * humidity / 100,
            (temperature - feels_like),
        ]])

        pred_idx = self.pipeline.predict(X)[0]
        proba = self.pipeline.predict_proba(X)[0]

        label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        class_probs = {}
        for i, cls in enumerate(self.label_encoder.classes_):
            class_probs[cls] = round(float(proba[i]), 3)

        return {
            "classification": label,
            "confidence": round(confidence, 3),
            "class_probabilities": class_probs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER CLUSTER ENGINE
# ─────────────────────────────────────────────────────────────────────────────


class WeatherClusterEngine:
    """
    KMeans clustering for weather pattern identification.

    Clusters:
        0 → Dry Heat
        1 → Humid Heat
        2 → Cold & Windy
        3 → Mild & Pleasant
    """

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_trained = False
        self.cluster_centers = None
        self.cluster_map = {}

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            df["temperature"].values,
            df["humidity"].values,
            df["wind_speed"].values,
        ])

    def train(self, df: pd.DataFrame) -> dict:
        """Train KMeans clustering model."""
        logger.info("Training WeatherClusterEngine...")

        X = self._build_features(df)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)

        # Auto-label clusters based on centroid characteristics
        self.cluster_map = self._auto_label_clusters()
        self.is_trained = True

        logger.info(f"WeatherClusterEngine trained with {self.n_clusters} clusters.")
        return {
            "n_clusters": self.n_clusters,
            "cluster_labels": self.cluster_map,
            "cluster_centers": self.cluster_centers.tolist(),
        }

    def _auto_label_clusters(self) -> dict:
        """Label clusters by centroid characteristics."""
        labels = {}
        for i, center in enumerate(self.cluster_centers):
            temp, hum, wind = center
            if temp >= 28 and hum < 50:
                labels[i] = "Dry Heat"
            elif temp >= 28 and hum >= 50:
                labels[i] = "Humid Heat"
            elif temp < 15 and wind > 20:
                labels[i] = "Cold & Windy"
            else:
                labels[i] = "Mild & Pleasant"
        # Ensure all 4 standard labels appear
        used = set(labels.values())
        defaults = ["Dry Heat", "Humid Heat", "Cold & Windy", "Mild & Pleasant"]
        for lbl in defaults:
            if lbl not in used:
                # Assign to closest unlabeled cluster
                for k in labels:
                    if labels[k] not in defaults[: defaults.index(lbl) + 1]:
                        labels[k] = lbl
                        break
        return labels

    def identify_pattern(
        self, temperature: float, humidity: float, wind_speed: float
    ) -> dict:
        """Identify weather pattern for given conditions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X = np.array([[temperature, humidity, wind_speed]])
        X_scaled = self.scaler.transform(X)

        cluster_id = int(self.model.predict(X_scaled)[0])
        pattern_name = self.cluster_map.get(cluster_id, "Mild & Pleasant")

        # Distance to nearest cluster center (normalized confidence)
        distances = np.linalg.norm(
            self.model.cluster_centers_ - X_scaled, axis=1
        )
        min_dist = distances[cluster_id]
        max_possible = np.max(distances)
        confidence = round(float(1 - min_dist / (max_possible + 1e-9)), 3)
        confidence = max(0.3, min(0.99, confidence))

        # Characteristics
        characteristics = self._describe_pattern(
            pattern_name, temperature, humidity, wind_speed
        )

        return {
            "cluster_id": cluster_id,
            "pattern": pattern_name,
            "confidence": confidence,
            "all_cluster_distances": {
                self.cluster_map.get(i, f"Cluster {i}"): round(float(distances[i]), 3)
                for i in range(self.n_clusters)
            },
            "characteristics": characteristics,
        }

    @staticmethod
    def _describe_pattern(
        pattern: str, temp: float, hum: float, wind: float
    ) -> str:
        descs = {
            "Dry Heat": f"Scorching {temp}°C with low {hum}% humidity — typical arid summer.",
            "Humid Heat": f"Hot {temp}°C with {hum}% humidity — feels oppressive.",
            "Cold & Windy": f"Chilly {temp}°C with strong {wind} km/h winds — bundle up.",
            "Mild & Pleasant": f"Comfortable {temp}°C — ideal outdoor conditions.",
        }
        return descs.get(pattern, f"Conditions: {temp}°C, {hum}% humidity, {wind} km/h wind.")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ML SERVICE
# ─────────────────────────────────────────────────────────────────────────────


class MLService:
    """
    Singleton master service that orchestrates all ML models.
    Call initialize() once at application startup.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.forecaster = TemperatureForecaster()
        self.advisor = ClothingAdvisor()
        self.classifier = WeatherClassifier()
        self.cluster_engine = WeatherClusterEngine()
        self.training_metrics = {}
        self._initialized = False

    def initialize(self) -> dict:
        """
        Initialize all ML models by training on synthetic data.
        Should be called once at application startup.
        """
        logger.info("=" * 60)
        logger.info("Initializing ML Service — training all models...")
        logger.info("=" * 60)

        generator = WeatherDataGenerator(n_samples=8000)
        df = generator.generate_historical_dataset()

        metrics = {}

        metrics["forecaster"] = self.forecaster.train(df)
        metrics["advisor"] = self.advisor.train(df)
        metrics["classifier"] = self.classifier.train(df)
        metrics["cluster_engine"] = self.cluster_engine.train(df)

        self.training_metrics = metrics
        self._initialized = True

        logger.info("=" * 60)
        logger.info("All ML models trained and ready.")
        logger.info("=" * 60)

        return metrics

    def run_full_analysis(
        self,
        temperature: float,
        humidity: float,
        wind_speed: float,
        feels_like: float,
        current_hour: int = None,
    ) -> dict:
        """
        Run all ML models for a given weather observation.
        Returns comprehensive analysis dict.
        """
        if not self._initialized:
            raise RuntimeError("MLService not initialized. Call initialize() first.")

        if current_hour is None:
            current_hour = datetime.now().hour

        # 1. Classification
        classification = self.classifier.classify(
            temperature, humidity, wind_speed, feels_like
        )

        # 2. Clothing recommendation
        clothing = self.advisor.recommend(temperature, wind_speed, humidity)

        # 3. Pattern clustering
        pattern = self.cluster_engine.identify_pattern(temperature, humidity, wind_speed)

        # 4. Temperature forecast
        forecast = self.forecaster.forecast_24h(
            temperature, humidity, wind_speed, current_hour
        )

        # Composite confidence score
        composite_confidence = round(
            (
                classification["confidence"] * 0.3
                + clothing["confidence"] * 0.3
                + pattern["confidence"] * 0.2
                + forecast["confidence"] * 0.2
            ),
            3,
        )

        return {
            "classification": classification,
            "clothing": clothing,
            "pattern": pattern,
            "forecast": forecast,
            "composite_confidence": composite_confidence,
            "training_metrics": self.training_metrics,
            "analysis_timestamp": datetime.now().isoformat(),
        }


# Module-level singleton
ml_service = MLService()
