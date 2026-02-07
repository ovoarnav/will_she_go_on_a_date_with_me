import base64
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from markov import update_belief, outcome_probs

# ---------- config ----------
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.2

client = OpenAI()

SCHEMA: Dict[str, Any] = {
    "name": "dm_feature_extract",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "interest_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "features": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "plan_specificity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "confirmation_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reschedule_intent": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "soft_decline_intent": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "hedge_density": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "enthusiasm_markers": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "question_engagement": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "sentiment": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                },
                "required": [
                    "plan_specificity",
                    "confirmation_strength",
                    "reschedule_intent",
                    "soft_decline_intent",
                    "hedge_density",
                    "enthusiasm_markers",
                    "question_engagement",
                    "sentiment",
                ],
            },
            "quality_flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "LOW_READABILITY",
                        "MISSING_CONTEXT",
                        "UNCLEAR_SPEAKER_LABELS",
                        "MULTIPLE_CONVERSATIONS",
                    ],
                },
            },
            "notes": {"type": "string", "maxLength": 400},
        },
        "required": ["interest_score", "features", "quality_flags", "notes"],
    },
}

SYSTEM_PROMPT = """You are an analyzer for two-person chat screenshots.
Goal: extract high-level conversation features that predict if a planned hangout/date happens.

Safety/Privacy rules:
- Do NOT include names, handles, phone numbers, addresses, venues, or other identifying details.
- Do NOT quote messages verbatim; only paraphrase generally.
- If the screenshots are unreadable or speakers are unclear, add the appropriate quality_flags.
Return ONLY JSON that matches the provided schema."""

USER_PROMPT = """Analyze these chat screenshots.
Assume there are 2 speakers: ME and THEM (the person I invited).
Extract the numeric features in the schema.
Compute interest_score in [0,1] with emphasis on logistics/confirmation over sentiment.
Keep notes short and non-identifying."""


def img_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_from_screenshots(image_paths: List[str]) -> Dict[str, Any]:
    content = [{"type": "input_text", "text": USER_PROMPT}]
    for p in image_paths:
        url = img_to_data_url(Path(p))
        content.append({"type": "input_image", "image_url": url})

    resp = client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA["name"],
                "schema": SCHEMA["schema"],
                "strict": True,
            }
        },
    )
    return json.loads(resp.output_text)


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    # default state
    return {
        "belief": [0.50, 0.35, 0.15],  # cold, warm, hot
        "updates": 0
    }


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def update_strength_from_quality(quality_flags: List[str]) -> float:
    """
    If screenshots are messy, dampen the update so one bad read doesn't swing belief.
    """
    if not quality_flags:
        return 1.0
    # progressively dampen
    if "LOW_READABILITY" in quality_flags or "UNCLEAR_SPEAKER_LABELS" in quality_flags:
        return 0.35
    if "MISSING_CONTEXT" in quality_flags:
        return 0.6
    return 0.5


def mix(old_b, new_b, alpha: float):
    return [
        (1 - alpha) * old_b[i] + alpha * new_b[i]
        for i in range(3)
    ]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="Paths to chat screenshots (png/jpg)")
    ap.add_argument("--state", default="state.json", help="State file path")
    ap.add_argument("--reset", action="store_true", help="Reset belief state")
    args = ap.parse_args()

    state_path = Path(args.state)

    if args.reset and state_path.exists():
        state_path.unlink()

    state = load_state(state_path)
    b0 = state["belief"]

    # 1) Extract
    result = extract_from_screenshots(args.images)
    print(json.dumps(result, indent=2))

    interest = float(result["interest_score"])
    features = result["features"]
    b1, emission, emission_score = update_belief(b0, interest, features)

    # 2) Damp update if quality is low
    alpha = update_strength_from_quality(result["quality_flags"])
    b_updated = mix(b0, b1, alpha)

    # 3) Save updated belief
    state["belief"] = b_updated
    state["updates"] = int(state.get("updates", 0)) + 1
    save_state(state_path, state)

    probs = outcome_probs(b_updated)

    print("\n--- EMISSION SCORE (true signal) ---")
    print(f"{emission_score:.3f}")

    print("\n--- TEXT TEMPERATURE (interest_score) ---")
    print(f"{interest:.3f}")

    print("\n--- EMISSION (cold,warm,hot) ---")
    print([round(x, 3) for x in emission])

    print("\n--- PREV BELIEF (cold,warm,hot) ---")
    print([round(x, 3) for x in b0])

    print("\n--- UPDATED BELIEF (cold,warm,hot) ---")
    print([round(x, 3) for x in b_updated])
    print(f"(alpha={alpha:.2f}, updates={state['updates']})")

    print("\n--- OUTCOME PROBS ---")
    print({k: round(v, 3) for k, v in probs.items()})


if __name__ == "__main__":
    main()
