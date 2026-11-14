from ntsb_agent import desk
from ntsb_agent.settings import MAX_HITS, PREVIEW

def hit_order(hit):
    # best score first; if two scores tie, the earlier letter first
    score = hit[0]
    position_in_case = hit[1]
    return (-score, position_in_case)


def search_history(query: str) -> str:
    """
    Search all letters in this case for words, and get short previews of where they appear.
    If nothing comes back, try other words with the same meaning (for example rule, regulation, rulemaking).

    Args:
        query: A few keywords, for example "final rule inspection".
    """
    # 1. the search words: lower-case, 3+ letters
    words = []
    for word in query.lower().split():
        word = word.strip(".,;:!?\"'()")
        if len(word) >= 3:
            words.append(word)
    if len(words) == 0:
        return "Give at least one keyword with 3 or more letters."

    # 2. score every letter = how many times the words appear in it
    hits = []
    position_in_case = 0
    for letter in desk.current.case["history"]:
        text = letter["text"]
        text_lower = text.lower()

        score = 0
        first = -1
        for word in words:
            count = text_lower.count(word)
            score = score + count
            if count > 0:
                where = text_lower.find(word)
                if first == -1 or where < first:
                    first = where

        if score > 0:
            # preview: ~150 chars around the first hit
            start = first - 50
            if start < 0:
                start = 0
            preview = text[start:start + PREVIEW]
            preview = " ".join(preview.split())          # squash newlines and double spaces
            hits.append((score, position_in_case, letter, preview))

        position_in_case = position_in_case + 1

    # 3. nothing found
    if len(hits) == 0:
        return f"No letter mentions: {', '.join(words)}. Try other words with the same meaning."

    # 4. best first (if two scores tie, the earlier letter comes first)
    hits.sort(key=hit_order)

    # 5. top 20, one line each
    lines = [f"{len(hits)} letters match (showing up to {MAX_HITS}, best first):"]
    shown = 0
    for score, position_in_case, letter, preview in hits:
        if shown == MAX_HITS:
            break
        lines.append(f"{letter['letter_id']} (Day {letter['day']}, {letter['from']}, {score} hits): ...{preview}...")
        shown = shown + 1

    return "\n".join(lines)


desk.TOOLS["search_history"] = search_history
