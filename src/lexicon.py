import re

BOTTLENECK_PATTERNS = [
    r"\b(?:embargo|blockade|ban|boycott)\b",
    r"\b(?:pipeline|refiner(?:y|ies)|terminal|tanker|cargo)\b",
    r"\b(?:chokepoint|hormuz|suez|red sea|strait)\b",
    r"\b(?:disrupt(?:ion|ed)?|halt(?:ed)?|shut(?:down)?|outage|shortage)\b",
    r"\b(?:strike|sabotage|seiz(?:e|ed|ure)|drone attack|missile attack)\b"
]

DIPLOMATIC_PATTERNS = [
    r"\b(?:ceasefire|truce|talks|diplomacy|diplomatic)\b",
    r"\b(?:negotiat(?:e|ion|ing)|peace treaty|envoy|summit|meeting|dialogue)\b"
]

def classify_title(title: str) -> str:
    if not isinstance(title, str):
        return "unrelated"
    t = title.lower()
    if any(re.search(pat, t) for pat in BOTTLENECK_PATTERNS):
        return "bottleneck"
    elif any(re.search(pat, t) for pat in DIPLOMATIC_PATTERNS):
        return "diplomatic"
    else:
        return "unrelated"
