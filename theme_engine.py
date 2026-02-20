
"""
Weather Intelligence Theme Engine
====================================
Maps weather classification to dynamic UI theme configurations.

Each theme defines:
    - Color palette (CSS variables)
    - Animation style
    - Particle configuration
    - Icon set
    - Gradient definition
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# THEME DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParticleConfig:
    """Configuration for CSS particle/animation effects."""
    type: str                     # "heatwave" | "snow" | "float" | "rain"
    count: int = 40               # Number of particles
    speed_min: float = 1.0        # Animation speed minimum (seconds)
    speed_max: float = 4.0        # Animation speed maximum (seconds)
    size_min: float = 2.0         # Particle size minimum (px)
    size_max: float = 8.0         # Particle size maximum (px)
    opacity_min: float = 0.3
    opacity_max: float = 0.9
    color: str = "rgba(255,255,255,0.6)"
    blur: float = 0.0


@dataclass
class ThemeConfig:
    """Complete theme configuration for a weather classification."""
    # Identity
    classification: str           # HOT | COLD | NORMAL
    theme_name: str               # Human-readable label
    emoji: str

    # Gradient
    gradient_start: str           # CSS color
    gradient_mid: str
    gradient_end: str
    gradient_angle: int = 135

    # CSS Variables (hex/rgb values)
    color_primary: str = "#ffffff"
    color_secondary: str = "#eeeeee"
    color_accent: str = "#f0a500"
    color_text: str = "#ffffff"
    color_text_muted: str = "rgba(255,255,255,0.7)"
    color_card_bg: str = "rgba(255,255,255,0.08)"
    color_card_border: str = "rgba(255,255,255,0.15)"
    color_glass: str = "rgba(255,255,255,0.06)"
    color_shadow: str = "rgba(0,0,0,0.3)"

    # Glow colors
    glow_primary: str = "rgba(255,165,0,0.4)"
    glow_secondary: str = "rgba(255,100,0,0.2)"

    # Particles
    particles: Optional[ParticleConfig] = None

    # Transition timing
    transition_duration: str = "0.8s"

    # Additional CSS class applied to body
    body_class: str = ""

    # Weather icons (emoji set)
    icons: List[str] = field(default_factory=lambda: ["☀️", "🌤️", "⛅"])

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# THEME DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

HOT_THEME = ThemeConfig(
    classification="HOT",
    theme_name="Blazing Heat",
    emoji="🔥",
    gradient_start="#FF4500",
    gradient_mid="#FF6B35",
    gradient_end="#FF8C00",
    gradient_angle=135,
    color_primary="#FF4500",
    color_secondary="#FF6B35",
    color_accent="#FFD700",
    color_text="#FFFFFF",
    color_text_muted="rgba(255,240,200,0.8)",
    color_card_bg="rgba(255,80,0,0.12)",
    color_card_border="rgba(255,150,50,0.25)",
    color_glass="rgba(255,60,0,0.08)",
    color_shadow="rgba(200,50,0,0.35)",
    glow_primary="rgba(255,100,0,0.5)",
    glow_secondary="rgba(255,69,0,0.3)",
    particles=ParticleConfig(
        type="heatwave",
        count=25,
        speed_min=2.0,
        speed_max=6.0,
        size_min=3.0,
        size_max=12.0,
        opacity_min=0.1,
        opacity_max=0.4,
        color="rgba(255,140,0,0.5)",
        blur=4.0,
    ),
    body_class="theme-hot",
    icons=["🌞", "🔥", "☀️", "🏜️", "♨️"],
)

COLD_THEME = ThemeConfig(
    classification="COLD",
    theme_name="Arctic Chill",
    emoji="❄️",
    gradient_start="#0A1628",
    gradient_mid="#1A2B5E",
    gradient_end="#0D3B6E",
    gradient_angle=160,
    color_primary="#4FC3F7",
    color_secondary="#81D4FA",
    color_accent="#B3E5FC",
    color_text="#E3F2FD",
    color_text_muted="rgba(179,229,252,0.75)",
    color_card_bg="rgba(30,80,150,0.18)",
    color_card_border="rgba(100,180,255,0.2)",
    color_glass="rgba(20,60,120,0.1)",
    color_shadow="rgba(5,20,60,0.4)",
    glow_primary="rgba(79,195,247,0.45)",
    glow_secondary="rgba(30,120,220,0.25)",
    particles=ParticleConfig(
        type="snow",
        count=60,
        speed_min=3.0,
        speed_max=8.0,
        size_min=2.0,
        size_max=7.0,
        opacity_min=0.4,
        opacity_max=0.95,
        color="rgba(220,240,255,0.85)",
        blur=1.0,
    ),
    body_class="theme-cold",
    icons=["❄️", "🌨️", "⛄", "🧊", "🌬️"],
)

NORMAL_THEME = ThemeConfig(
    classification="NORMAL",
    theme_name="Serene Skies",
    emoji="🌤️",
    gradient_start="#0F2027",
    gradient_mid="#203A43",
    gradient_end="#2C5364",
    gradient_angle=120,
    color_primary="#56AB91",
    color_secondary="#2AB87B",
    color_accent="#7FDBCA",
    color_text="#E8F5E9",
    color_text_muted="rgba(200,230,220,0.75)",
    color_card_bg="rgba(40,120,90,0.15)",
    color_card_border="rgba(80,180,140,0.22)",
    color_glass="rgba(30,100,70,0.08)",
    color_shadow="rgba(10,40,30,0.35)",
    glow_primary="rgba(86,171,145,0.42)",
    glow_secondary="rgba(42,184,123,0.22)",
    particles=ParticleConfig(
        type="float",
        count=20,
        speed_min=4.0,
        speed_max=10.0,
        size_min=4.0,
        size_max=14.0,
        opacity_min=0.08,
        opacity_max=0.22,
        color="rgba(150,230,200,0.5)",
        blur=6.0,
    ),
    body_class="theme-normal",
    icons=["🌤️", "🌿", "🍃", "☁️", "🌾"],
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

THEME_REGISTRY = {
    "HOT": HOT_THEME,
    "COLD": COLD_THEME,
    "NORMAL": NORMAL_THEME,
}

# Pattern-specific sub-themes (overlay adjustments)
PATTERN_OVERRIDES = {
    "Dry Heat": {
        "gradient_mid": "#D4520A",
        "color_accent": "#FFB800",
    },
    "Humid Heat": {
        "gradient_mid": "#8B2500",
        "color_accent": "#FFD700",
    },
    "Cold & Windy": {
        "gradient_mid": "#0A1040",
        "color_accent": "#90E0FF",
    },
    "Mild & Pleasant": {
        "gradient_mid": "#1A4040",
        "color_accent": "#80FFD0",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# THEME ENGINE
# ─────────────────────────────────────────────────────────────────────────────


class ThemeEngine:
    """
    Resolves the UI theme configuration for a given weather classification
    and optionally blends in pattern-specific overrides.
    """

    @staticmethod
    def get_theme(
        classification: str,
        pattern: str = None,
    ) -> dict:
        """
        Get complete theme configuration dict.

        Args:
            classification: HOT | COLD | NORMAL
            pattern: Optional weather pattern name for fine-tuning

        Returns:
            Complete theme config as dict (JSON-serializable)
        """
        classification = classification.upper().strip()

        if classification not in THEME_REGISTRY:
            classification = "NORMAL"

        theme = THEME_REGISTRY[classification]
        theme_dict = theme.to_dict()

        # Apply pattern override if provided
        if pattern and pattern in PATTERN_OVERRIDES:
            override = PATTERN_OVERRIDES[pattern]
            theme_dict.update(override)

        # Build CSS variables dict for frontend injection
        theme_dict["css_variables"] = ThemeEngine._build_css_variables(theme_dict)

        # Build gradient string
        theme_dict["gradient_css"] = (
            f"linear-gradient({theme_dict['gradient_angle']}deg, "
            f"{theme_dict['gradient_start']}, "
            f"{theme_dict['gradient_mid']}, "
            f"{theme_dict['gradient_end']})"
        )

        return theme_dict

    @staticmethod
    def _build_css_variables(theme: dict) -> dict:
        """Build CSS custom property map."""
        return {
            "--color-primary": theme["color_primary"],
            "--color-secondary": theme["color_secondary"],
            "--color-accent": theme["color_accent"],
            "--color-text": theme["color_text"],
            "--color-text-muted": theme["color_text_muted"],
            "--color-card-bg": theme["color_card_bg"],
            "--color-card-border": theme["color_card_border"],
            "--color-glass": theme["color_glass"],
            "--color-shadow": theme["color_shadow"],
            "--glow-primary": theme["glow_primary"],
            "--glow-secondary": theme["glow_secondary"],
            "--gradient-css": (
                f"linear-gradient({theme['gradient_angle']}deg, "
                f"{theme['gradient_start']}, "
                f"{theme['gradient_mid']}, "
                f"{theme['gradient_end']})"
            ),
        }

    @staticmethod
    def get_animation_config(particle_config: dict) -> dict:
        """
        Build animation parameters for the frontend particle system.
        Returns JSON-safe config.
        """
        if not particle_config:
            return {"type": "float", "count": 20}

        return {
            "type": particle_config.get("type", "float"),
            "count": particle_config.get("count", 30),
            "speed_min": particle_config.get("speed_min", 2),
            "speed_max": particle_config.get("speed_max", 6),
            "size_min": particle_config.get("size_min", 3),
            "size_max": particle_config.get("size_max", 10),
            "opacity_min": particle_config.get("opacity_min", 0.2),
            "opacity_max": particle_config.get("opacity_max", 0.8),
            "color": particle_config.get("color", "rgba(255,255,255,0.6)"),
            "blur": particle_config.get("blur", 0),
        }


# Module-level instance
theme_engine = ThemeEngine()
