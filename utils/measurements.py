import re


def format_length(context, value):
    """Format a Blender world-space length for compact live feedback."""
    value = abs(float(value))
    try:
        scene = context.scene
        units = scene.unit_settings
        if getattr(units, "system", "NONE") != "NONE":
            import bpy
            return bpy.utils.units.to_string(
                units.system,
                "LENGTH",
                value,
                precision=4,
                split_unit=False,
            )
    except Exception:
        pass
    return f"{value:.3f} m"


def parse_length(text):
    """Parse SketchUp-like length input. Plain values are Blender units."""
    raw = (text or "").strip().lower().replace(" ", "")
    if not raw:
        return None
    raw = raw.replace(",", ".")
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))(mm|cm|m|km|ft|in|")?', raw)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or ""
    factors = {
        "": 1.0,
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "km": 1000.0,
        "ft": 0.3048,
        "in": 0.0254,
        '"': 0.0254,
    }
    value = number * factors[unit]
    return value if value > 1e-9 else None
