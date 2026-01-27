from collections import Counter
from typing import List, Dict, Optional


def select_dominant_meeting(chunks: List[Dict]) -> Optional[int]:
    if not isinstance(chunks, list) or not chunks:
        return None

    meeting_ids = [
        ch.get("meeting_index")
        for ch in chunks
        if isinstance(ch, dict) and isinstance(ch.get("meeting_index"), int)
    ]



    if not meeting_ids:
        return None

    counts = Counter(meeting_ids)
    total = sum(counts.values())

    if total == 0:
        return None

    dominant_meeting, dominant_count = counts.most_common(1)[0]

    dominance_ratio = dominant_count / total

    if dominance_ratio >= 0.40:
        return dominant_meeting

    return None
