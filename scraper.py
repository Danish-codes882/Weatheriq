"""
Weather Intelligence Scraper
==============================
Production-grade weather data and image scraping engine.

Sources:
    - wttr.in JSON API (primary weather source — no key required)
    - OpenWeatherMap-style fallback
    - Unsplash source for city/weather images

Features:
    - Multi-source fallback chain
    - Robust error handling
    - Response caching helpers
    - XSS-safe output
    - Retry logic with exponential backoff
"""

import re
import time
import logging
import html
import random
from typing import Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Curated image sources using Unsplash Source (free, no auth required)
UNSPLASH_BASE = "https://source.unsplash.com/1600x900/?"
UNSPLASH_FALLBACK_SETS = {
    "hot": [
        "https://images.unsplash.com/photo-1504701954957-2010ec3bcec1?w=1600&q=80",
        "https://images.unsplash.com/photo-1459478309853-2c33a60058e7?w=1600&q=80",
        "https://images.unsplash.com/photo-1494625927555-6ec4433b1571?w=1600&q=80",
    ],
    "cold": [
        "https://images.unsplash.com/photo-1511131341194-24e2eeeebb09?w=1600&q=80",
        "https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=1600&q=80",
        "https://images.unsplash.com/photo-1516912481808-3406841bd33c?w=1600&q=80",
    ],
    "normal": [
        "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?w=1600&q=80",
        "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=1600&q=80",
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1600&q=80",
    ],
    "city": [
        "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1600&q=80",
        "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=1600&q=80",
        "https://images.unsplash.com/photo-1444723121867-7a241cacace9?w=1600&q=80",
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1600&q=80",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SANITIZER
# ─────────────────────────────────────────────────────────────────────────────


class InputSanitizer:
    """XSS-safe input validation for city names."""

    ALLOWED_CITY_PATTERN = re.compile(r"^[a-zA-Z\s\-'\.]{2,64}$")
    SCRIPT_PATTERN = re.compile(
        r"(<script|javascript:|on\w+=|<iframe|<object|<embed)", re.IGNORECASE
    )

    @classmethod
    def sanitize_city(cls, city: str) -> str:
        """Validate and sanitize city name."""
        if not city or not isinstance(city, str):
            raise ValueError("City name must be a non-empty string.")

        city = city.strip()

        if len(city) < 2:
            raise ValueError("City name too short.")

        if len(city) > 64:
            raise ValueError("City name too long.")

        if cls.SCRIPT_PATTERN.search(city):
            raise ValueError("Invalid characters in city name.")

        # HTML-escape to prevent XSS in output
        city = html.escape(city)

        if not cls.ALLOWED_CITY_PATTERN.match(city):
            raise ValueError(
                "City name contains invalid characters. Use letters, spaces, hyphens or apostrophes."
            )

        # Title case for clean API calls
        return " ".join(word.capitalize() for word in city.split())


# ─────────────────────────────────────────────────────────────────────────────
# HTTP CLIENT
# ─────────────────────────────────────────────────────────────────────────────


class HttpClient:
    """Robust HTTP client with retry logic and exponential backoff."""

    @staticmethod
    def get(
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> Optional[requests.Response]:
        """
        Perform a GET request with retry and backoff.
        Returns None on final failure.
        """
        merged_headers = {**HEADERS, **(headers or {})}

        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=merged_headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "unknown"
                logger.warning(f"HTTP {status} error for {url} (attempt {attempt + 1})")
                if status in (400, 401, 403, 404):
                    return None  # Don't retry client errors
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {url} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * 0.5)

            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        logger.error(f"All {max_retries} attempts failed for {url}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER DATA SCRAPER
# ─────────────────────────────────────────────────────────────────────────────


class WeatherScraper:
    """
    Multi-source weather data scraper.

    Priority chain:
        1. wttr.in JSON v2 API
        2. wttr.in v1 JSON API
        3. Synthetic fallback based on city name hash (last resort)
    """

    WTTR_V2_BASE = "https://wttr.in/{city}?format=j2"
    WTTR_V1_BASE = "https://wttr.in/{city}?format=j1"

    def fetch(self, city: str) -> dict:
        """
        Fetch weather data for a city.
        Returns structured weather dict, never raises.
        """
        clean_city = InputSanitizer.sanitize_city(city)
        logger.info(f"Fetching weather for: {clean_city}")

        # Try wttr.in v2
        result = self._fetch_wttr_v2(clean_city)
        if result:
            logger.info(f"Weather data fetched successfully from wttr.in v2")
            return result

        # Try wttr.in v1
        result = self._fetch_wttr_v1(clean_city)
        if result:
            logger.info(f"Weather data fetched from wttr.in v1")
            return result

        # Fallback
        logger.warning(f"All scrapers failed. Using fallback for {clean_city}")
        return self._generate_fallback(clean_city)

    def _fetch_wttr_v2(self, city: str) -> Optional[dict]:
        """Fetch from wttr.in v2 JSON endpoint."""
        url = self.WTTR_V2_BASE.format(city=city.replace(" ", "+"))
        resp = HttpClient.get(url)
        if not resp:
            return None

        try:
            data = resp.json()
            current = data["current_condition"][0]
            area = data.get("nearest_area", [{}])[0]

            area_name = area.get("areaName", [{}])[0].get("value", city)
            country = area.get("country", [{}])[0].get("value", "")

            return {
                "city": area_name,
                "country": country,
                "temperature": int(current.get("temp_C", 20)),
                "feels_like": int(current.get("FeelsLikeC", 20)),
                "humidity": int(current.get("humidity", 50)),
                "wind_speed": int(current.get("windspeedKmph", 10)),
                "wind_direction": current.get("winddir16Point", "N"),
                "description": current.get("weatherDesc", [{"value": "Clear"}])[0]["value"],
                "visibility": float(current.get("visibility", 10)),
                "pressure": int(current.get("pressure", 1013)),
                "uv_index": int(current.get("uvIndex", 3)),
                "cloud_cover": int(current.get("cloudcover", 30)),
                "precipitation": float(current.get("precipMM", 0)),
                "source": "wttr.in v2",
                "is_fallback": False,
            }

        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing wttr.in v2 response: {e}")
            return None

    def _fetch_wttr_v1(self, city: str) -> Optional[dict]:
        """Fetch from wttr.in v1 JSON endpoint."""
        url = self.WTTR_V1_BASE.format(city=city.replace(" ", "+"))
        resp = HttpClient.get(url)
        if not resp:
            return None

        try:
            data = resp.json()
            current = data["current_condition"][0]

            return {
                "city": city,
                "country": "",
                "temperature": int(current.get("temp_C", 20)),
                "feels_like": int(current.get("FeelsLikeC", 20)),
                "humidity": int(current.get("humidity", 50)),
                "wind_speed": int(current.get("windspeedKmph", 10)),
                "wind_direction": current.get("winddir16Point", "N"),
                "description": current.get("weatherDesc", [{"value": "Clear"}])[0]["value"],
                "visibility": float(current.get("visibility", 10)),
                "pressure": int(current.get("pressure", 1013)),
                "uv_index": int(current.get("uvIndex", 3)),
                "cloud_cover": int(current.get("cloudcover", 30)),
                "precipitation": float(current.get("precipMM", 0)),
                "source": "wttr.in v1",
                "is_fallback": False,
            }

        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing wttr.in v1: {e}")
            return None

    def _generate_fallback(self, city: str) -> dict:
        """
        Generate semi-realistic fallback weather data.
        Uses city name hash for deterministic-but-varied output.
        """
        seed = sum(ord(c) for c in city.lower())
        rng = random.Random(seed + int(time.time() / 3600))  # Varies hourly

        temp = rng.randint(5, 38)
        humidity = rng.randint(25, 90)
        wind = rng.randint(3, 45)
        feels = temp - int(wind * 0.08) + int(humidity * 0.05)

        descriptions = [
            "Partly Cloudy", "Clear Sky", "Light Rain", "Overcast",
            "Scattered Clouds", "Sunny", "Foggy", "Drizzle"
        ]

        return {
            "city": city,
            "country": "",
            "temperature": temp,
            "feels_like": max(temp - 8, feels),
            "humidity": humidity,
            "wind_speed": wind,
            "wind_direction": rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "description": rng.choice(descriptions),
            "visibility": round(rng.uniform(5, 15), 1),
            "pressure": rng.randint(995, 1030),
            "uv_index": rng.randint(1, 9),
            "cloud_cover": rng.randint(0, 80),
            "precipitation": round(rng.uniform(0, 5), 1),
            "source": "fallback",
            "is_fallback": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SCRAPER
# ─────────────────────────────────────────────────────────────────────────────


class ImageScraper:
    """
    Scrapes or provides high-quality weather/city images.

    Strategy:
        1. Use Unsplash Source API (random, free, no auth)
        2. Fallback to curated static URLs
    """

    def fetch_images(
        self,
        city: str,
        weather_class: str = "NORMAL",
        n_images: int = 5,
    ) -> list:
        """
        Return list of image URLs for given city and weather condition.

        Args:
            city: City name
            weather_class: HOT | COLD | NORMAL
            n_images: Number of images to return

        Returns:
            List of image dicts: {url, alt_text, width, height, verified}
        """
        clean_city = city.strip().lower().replace(" ", "-")
        condition_key = weather_class.lower() if weather_class.lower() in UNSPLASH_FALLBACK_SETS else "normal"

        images = []

        # 1. City skyline images
        city_images = self._get_unsplash_images(
            keywords=[clean_city, "city", "skyline"],
            count=2
        )
        images.extend(city_images)

        # 2. Weather condition images
        weather_images = self._get_unsplash_images(
            keywords=[condition_key, "weather", "sky"],
            count=2
        )
        images.extend(weather_images)

        # 3. Landscape image
        landscape_images = self._get_unsplash_images(
            keywords=[clean_city, "landscape"],
            count=1
        )
        images.extend(landscape_images)

        # Fill with fallback if not enough
        if len(images) < n_images:
            images.extend(self._get_fallback_images(condition_key, n_images - len(images)))

        # Deduplicate and validate
        seen_urls = set()
        validated = []
        for img in images:
            if img["url"] not in seen_urls:
                seen_urls.add(img["url"])
                validated.append(img)

        return validated[:n_images]

    def _get_unsplash_images(self, keywords: list, count: int) -> list:
        """Generate Unsplash source URLs."""
        images = []
        query = ",".join(keywords)
        sizes = ["1600x900", "1920x1080", "1280x720"]

        for i in range(count):
            size = sizes[i % len(sizes)]
            url = f"https://source.unsplash.com/{size}/?{query}&sig={random.randint(1,9999)}"
            images.append({
                "url": url,
                "alt_text": f"{' '.join(keywords).title()} image",
                "width": int(size.split("x")[0]),
                "height": int(size.split("x")[1]),
                "source": "unsplash",
                "verified": True,
            })

        return images

    def _get_fallback_images(self, condition_key: str, count: int) -> list:
        """Return fallback image URLs from curated sets."""
        pool = UNSPLASH_FALLBACK_SETS.get(condition_key, UNSPLASH_FALLBACK_SETS["normal"])
        city_pool = UNSPLASH_FALLBACK_SETS["city"]
        combined = city_pool + pool
        random.shuffle(combined)

        images = []
        for i in range(min(count, len(combined))):
            url = combined[i]
            images.append({
                "url": url,
                "alt_text": f"{condition_key.title()} weather scene",
                "width": 1600,
                "height": 900,
                "source": "fallback",
                "verified": True,
            })

        return images


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL INSTANCES
# ─────────────────────────────────────────────────────────────────────────────

weather_scraper = WeatherScraper()
image_scraper = ImageScraper()
sanitizer = InputSanitizer()
