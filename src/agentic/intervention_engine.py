# src/agentic/intervention_engine.py
from typing import Dict, Any, List

# Intervention library
INTERVENTION_MAP = {
    "late_booking": [
        {"type": "reassurance", "text": "Priority express check-in to save your time."},
        {"type": "perk", "text": "Complimentary welcome drink upon arrival."},
        {"type": "logistics", "text": "Clear directions & parking info to make arrival smooth."}
    ],
    "value_sensitive": [
        {"type": "offer", "text": "Exclusive 10% discount on breakfast or late check-out for this stay."},
        {"type": "highlight", "text": "Free Wi-Fi and complimentary shuttle service to nearby attractions."}
    ],
    "low_price_sensitivity": [
        {"type": "upsell", "text": "Would you like a room upgrade or spa package? We can reserve it now."}
    ],
    "short_stay": [
        {"type": "convenience", "text": "We prepared a quick arrival guide and express services for short stays."},
        {"type": "timing", "text": "If you need early check-in, reply and we’ll prioritize your room."}
    ],
    "long_stay": [
        {"type": "experience", "text": "We can arrange a local day-tour or laundry package for longer stays."}
    ],
    "high_expectations": [
        {"type": "reassurance", "text": "Our team is ready to handle any special requests you have — tell us what matters most."},
        {"type": "quality", "text": "We’ll prioritize a quiet, high-floor room with thoughtful amenities."}
    ],
    "premium_cluster": [
        {"type": "premium", "text": "We recommend our signature spa and chef's tasting menu for an elevated stay."},
        {"type": "welcome", "text": "Complimentary fruit basket on arrival (subject to availability)."}
    ],
    "value_cluster": [
        {"type": "local_events", "text": "Here are free or low-cost local events happening during your stay."}
    ],
    "frequent_traveler": [
        {"type": "loyalty", "text": "We can apply your loyalty benefits and ensure your preferences are remembered."}
    ],
    "loyal_guest": [
        {"type": "thankyou", "text": "As a valued guest, enjoy a special courtesy on us."}
    ],
    # default fallback implicitly handled in map_intents_to_interventions
}

def map_intents_to_interventions(intents: List[tuple], hotel_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Convert ranked intents -> concrete interventions, enriched by hotel_context if provided.
    intents: list of (intent_label, score)
    """
    hotel_context = hotel_context or {}
    interventions = []
    seen = set()
    for intent, score in intents:
        if intent in INTERVENTION_MAP:
            for item in INTERVENTION_MAP[intent]:
                enriched = item.copy()
                # enrich with context if available
                if item["type"] == "local_events" and hotel_context.get("local_events"):
                    enriched["text"] = f"Local events: {hotel_context['local_events'][:3]}"
                if item["type"] == "offer" and hotel_context.get("offers"):
                    enriched["text"] = f"{hotel_context['offers'][0]} — {item['text']}"
                key = (enriched["type"], enriched["text"])
                if key not in seen:
                    interventions.append({"intent": intent, "score": float(score), **enriched})
                    seen.add(key)
    if not interventions:
        interventions.append({"intent": "default", "score": 0.1, "type": "info", "text": "We look forward to making your stay comfortable. Reply if you need anything."})
    return interventions
