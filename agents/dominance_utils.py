from collections import Counter
from typing import List, Dict, Optional


def select_dominant_meeting(chunks: List[Dict]) -> Optional[str]:
    """
    Select dominant meeting_id (UUID string) using similarity-weighted dominance.

    Rules:
    - Works ONLY with meeting_id (UUID string)
    - Weights dominance by similarity score if available
    - Requires >= 50% weighted dominance
    - Returns None if dominance unclear
    """

    if not isinstance(chunks, list) or not chunks:
        return None

    weighted_counts = {}
    total_weight = 0.0

    for ch in chunks:
        if not isinstance(ch, dict):
            continue

        meeting_id = ch.get("meeting_id")
        if not isinstance(meeting_id, str):
            continue

        # Use similarity if available, else default weight = 1.0
        weight = ch.get("similarity", 1.0)

        # Ensure valid numeric weight
        try:
            weight = float(weight)
        except Exception:
            weight = 1.0

        if weight < 0:
            weight = 0.0

        weighted_counts[meeting_id] = weighted_counts.get(meeting_id, 0.0) + weight
        total_weight += weight

    if not weighted_counts or total_weight == 0:
        return None

    dominant_meeting = max(weighted_counts, key=weighted_counts.get)
    dominance_ratio = weighted_counts[dominant_meeting] / total_weight

    # Require stronger dominance threshold
    if dominance_ratio >= 0.50:
        return dominant_meeting

    return None
