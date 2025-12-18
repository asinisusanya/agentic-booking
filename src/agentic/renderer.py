# src/agentic/renderer.py
from typing import List, Dict, Any
from .intent_engine import interpret_shap
from .intervention_engine import map_intents_to_interventions

# Map cluster → tone
CLUSTER_TO_TONE = {0: "premium", 1: "friendly"}


# -------------------------------------------------------
# Helper: Natural language joiner
# ["A", "B", "C"] → "A, B, and C"
# -------------------------------------------------------
def natural_join(items: List[str]) -> str:
    items = [i.strip() for i in items if i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# -------------------------------------------------------
# Helper: Group interventions into arrival/experience/perks
# -------------------------------------------------------
def _group_interventions(interventions: List[Dict[str, Any]]):
    arrival_types = {"reassurance", "logistics", "convenience", "timing"}
    experience_types = {"premium", "upsell", "experience", "local_events", "quality", "highlight"}
    perks_types = {"offer", "perk", "thankyou", "value", "loyalty"}

    arrival = []
    experience = []
    perks = []

    for it in interventions:
        t = it.get("type", "").lower()
        txt = it.get("text", "")
        if t in arrival_types:
            arrival.append(txt)
        elif t in experience_types:
            experience.append(txt)
        elif t in perks_types:
            perks.append(txt)
        else:
            experience.append(txt)

    # Deduplicate
    def uniq(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "arrival": uniq(arrival),
        "experience": uniq(experience),
        "perks": uniq(perks),
    }


# -------------------------------------------------------
# Luxury-grade email renderer (v3)
# -------------------------------------------------------
def render_email_v3(
    guest_name: str,
    hotel_name: str,
    tone: str,
    interventions: List[Dict[str, Any]],
    booking_info: Dict[str, Any]
) -> Dict[str, str]:

    groups = _group_interventions(interventions)
    is_friendly = (tone == "friendly")

    # Soft emojis for friendly tone only
    EMO = {
        "arrival": "🛬 " if is_friendly else "",
        "suggest": "✨ " if is_friendly else "",
        "perk": "💝 " if is_friendly else "",
    }

    # -----------------------------------------
    # Tone templates
    # -----------------------------------------
    if tone == "premium":
        greeting = f"Dear {guest_name},"
        intro = (
            f"We are delighted to welcome you to {hotel_name}. "
            "To ensure a seamless and comfortable arrival, our team has prepared a few thoughtful arrangements for your stay."
        )
        closing = (
            "Warm regards,\n"
            f"The {hotel_name} Team"
        )
    else:
        greeting = f"Hi {guest_name},"
        intro = (
            f"Thanks for choosing {hotel_name}! We’re excited to welcome you and have prepared a few thoughtful details "
            "to make your arrival and stay as easy and enjoyable as possible."
        )
        closing = (
            "Warm regards,\n"
            f"The {hotel_name} Team"
        )

    # -----------------------------------------
    # Build paragraphs with luxury tone
    # -----------------------------------------

    # ARRIVAL
    arrival_text = ""
    if groups["arrival"]:
        combined = natural_join(groups["arrival"])
        arrival_text = (
            f"{combined}. "
            "Our team will be happy to assist with anything else you may need upon arrival."
        )

    # EXPERIENCE
    experience_text = ""
    if groups["experience"]:
        combined = natural_join(groups["experience"])
        experience_text = (
            f"{combined}. "
            "Should you wish to personalize your stay further, we would be glad to arrange it for you."
        )

    # PERKS
    perks_text = ""
    if groups["perks"]:
        combined = natural_join(groups["perks"])
        perks_text = (
            f"{combined}. "
            "These touches are offered to ensure your stay feels warm, memorable, and effortlessly comfortable."
        )

    # -----------------------------------------
    # Build final email text
    # -----------------------------------------
    plain_lines = [greeting, "", intro, ""]

    if arrival_text:
        plain_lines.append(f"{EMO['arrival']}Smooth Arrival")
        plain_lines.append(arrival_text)
        plain_lines.append("")

    if experience_text:
        plain_lines.append(f"{EMO['suggest']}Personalized Suggestions")
        plain_lines.append(experience_text)
        plain_lines.append("")

    if perks_text:
        plain_lines.append(f"{EMO['perk']}A Thoughtful Courtesy")
        plain_lines.append(perks_text)
        plain_lines.append("")

    plain_lines.append(
        "If you would like us to arrange anything before your arrival, simply reply to this message — "
        "we would be pleased to assist."
    )
    plain_lines.append("")
    plain_lines.append(closing)

    plain_email = "\n".join(plain_lines)

    # -----------------------------------------
    # HTML version
    # -----------------------------------------
    html_parts = [
        f"<p>{greeting}</p>",
        f"<p>{intro}</p>",
    ]

    if arrival_text:
        html_parts.append(f"<h4>{EMO['arrival']}Smooth Arrival</h4>")
        html_parts.append(f"<p>{arrival_text}</p>")

    if experience_text:
        html_parts.append(f"<h4>{EMO['suggest']}Personalized Suggestions</h4>")
        html_parts.append(f"<p>{experience_text}</p>")

    if perks_text:
        html_parts.append(f"<h4>{EMO['perk']}A Thoughtful Courtesy</h4>")
        html_parts.append(f"<p>{perks_text}</p>")

    html_parts.append(
        "<p>If you would like us to arrange anything before your arrival, simply reply to this message — "
        "we would be pleased to assist.</p>"
    )
    html_parts.append(f"<p>{closing.replace(chr(10), '<br>')}</p>")

    html_email = "\n".join(html_parts)

    subject = f"{hotel_name} — A warm welcome and thoughtful arrangements for your stay"

    return {
        "subject": subject,
        "plain": plain_email,
        "html": html_email
    }


# -------------------------------------------------------
# Wrapper for email_agent
# -------------------------------------------------------
def generate_agentic_email_v3(
    booking: Dict[str, Any],
    shap_contribs: Dict[str, float],
    model_prediction: float,
    hotel_context: Dict[str, Any],
    guest_name: str,
    hotel_name: str
):
    # Orchestrates the new renderer
    from .intent_engine import interpret_shap
    from .intervention_engine import map_intents_to_interventions

    intents = interpret_shap(shap_contribs, booking)
    interventions = map_intents_to_interventions(intents, hotel_context)
    tone = CLUSTER_TO_TONE.get(booking.get("cluster"), "friendly")

    email = render_email_v3(
        guest_name=guest_name,
        hotel_name=hotel_name,
        tone=tone,
        interventions=interventions,
        booking_info=booking
    )

    # Rationale (same as before, but prettier output)
    rationale = {
        "intents_ranked": intents,
        "top_interventions": interventions[:5],
        "shap_contribs": shap_contribs,
        "prediction_for_logs": model_prediction
    }

    return {
        "email": email,
        "rationale": rationale,
        "tone": tone,
        "booking_id": booking.get("booking_id")
    }
