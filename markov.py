import math
from typing import Dict, List, Tuple


# ============================================================
# Helpers
# ============================================================

def normalize(v: List[float]) -> List[float]:
    s = sum(v)
    if s <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [x / s for x in v]


def matvec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# TRANSITION MATRIX (state persistence)
# ============================================================

# [COLD, WARM, HOT]
A = [
    [0.72, 0.25, 0.03],  # cold tends to stay cold early
    [0.18, 0.58, 0.24],  # warm flexible
    [0.06, 0.24, 0.70],  # hot sticky
]

# ============================================================
# PERSONAL BASE RATES (edit with your real data)
# ============================================================

# number of date you've been on n =

#BASE_SHOW =  / n
#BASE_RES =  / n
#BASE_NO =  / n


# ============================================================
# Build personalized outcome matrix
# ============================================================

def build_outcome_matrix(
        base_show: float,
        base_res: float,
        base_no: float,
        hot_boost: float = 0.12,
        cold_drop: float = 0.50,
) -> Dict[str, List[float]]:
    # WARM = your real average
    warm_show = base_show
    warm_res = base_res
    warm_no = base_no

    # HOT = stronger show probability
    hot_show = clamp(base_show + hot_boost, 0.05, 0.97)
    hot_res = clamp(base_res * 0.7, 0.01, 0.40)
    hot_no = clamp(1.0 - hot_show - hot_res, 0.01, 0.90)

    # COLD = weaker show
    cold_show = clamp(base_show - cold_drop, 0.01, 0.80)
    cold_res = clamp(base_res * 1.6, 0.02, 0.60)
    cold_no = clamp(1.0 - cold_show - cold_res, 0.05, 0.98)

    cold_row = normalize([cold_show, cold_res, cold_no])
    warm_row = normalize([warm_show, warm_res, warm_no])
    hot_row = normalize([hot_show, hot_res, hot_no])

    return {
        "show": [cold_row[0], warm_row[0], hot_row[0]],
        "resched": [cold_row[1], warm_row[1], hot_row[1]],
        "no": [cold_row[2], warm_row[2], hot_row[2]],
    }


OUTCOME = build_outcome_matrix(BASE_SHOW, BASE_RES, BASE_NO)


# ============================================================
# FEATURE → EMISSION SCORE
# (this is the big improvement)
# ============================================================

def compute_emission_score(interest_score: float, features: Dict) -> float:
    """
    Combines interest_score + extracted features into a stable signal.
    Prevents early chats from being treated as rejection.
    """

    logi = 0.6 * features.get("plan_specificity", 0) + \
           0.4 * features.get("confirmation_strength", 0)

    social = 0.6 * features.get("question_engagement", 0) + \
             0.4 * features.get("enthusiasm_markers", 0)

    penalty = 0.5 * features.get("soft_decline_intent", 0) + \
              0.3 * features.get("hedge_density", 0)

    # combine
    score = (
            0.55 * interest_score +
            0.25 * logi +
            0.20 * social -
            0.45 * penalty
    )

    return clamp(score, 0.0, 1.0)


# ============================================================
# EMISSION FROM SCORE → STATE PROB
# ============================================================

def emission_from_score(score: float) -> List[float]:
    """
    Soft Gaussian mapping score → [cold,warm,hot]
    """

    centers = [0.22, 0.52, 0.80]  # cold warm hot
    var = [0.06, 0.07, 0.06]  # softer = less jumpy

    raw = [math.exp(-((score - c) ** 2) / v) for c, v in zip(centers, var)]

    # epsilon smoothing (critical)
    eps = 1e-3
    raw = [r + eps for r in raw]

    return normalize(raw)


# ============================================================
# BELIEF UPDATE
# ============================================================

def update_belief(
        b: List[float],
        interest_score: float,
        features: Dict
) -> Tuple[List[float], List[float], float]:
    """
    Returns:
    new_belief
    emission_vector
    emission_score
    """

    emission_score = compute_emission_score(interest_score, features)
    e = emission_from_score(emission_score)

    pred = matvec(A, b)
    post = [e[i] * pred[i] for i in range(3)]
    new_b = normalize(post)

    return new_b, e, emission_score


# ============================================================
# OUTCOME PROBABILITIES
# ============================================================

def outcome_probs(b: List[float]) -> Dict[str, float]:
    return {
        "show": sum(b[i] * OUTCOME["show"][i] for i in range(3)),
        "resched": sum(b[i] * OUTCOME["resched"][i] for i in range(3)),
        "no": sum(b[i] * OUTCOME["no"][i] for i in range(3)),
    }


def debug_outcome_matrix():
    return OUTCOME
