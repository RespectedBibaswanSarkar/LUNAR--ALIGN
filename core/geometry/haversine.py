import math

MOON_RADIUS_KM = 1737.4


def haversine_distance(lon1, lat1, lon2, lat2, radius_km=MOON_RADIUS_KM):
    """Return great-circle distance in kilometers for two geographic points on the Moon (or specified body)."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
    return radius_km * c
