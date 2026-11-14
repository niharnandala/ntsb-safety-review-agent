from ntsb_agent import desk
from ntsb_agent.grades import last_ntsb_grade, letter_line

def get_recommendation() -> str:
    """
    Open the case: what NTSB recommended, who it was sent to, which letter you must review, NTSB's latest
    letter, the last grade NTSB gave in this case, and a list of all letters. Call this first.
    """
    c = desk.current.case
    lines = []

    lines.append(f"CASE: {c['case_id']} | mode: {c['mode']} | priority: {c['priority']} | most wanted: {c['most_wanted']}")

    if c["recipient"]:
        lines.append(f"SENT TO: {c['recipient']}")
    else:
        lines.append("SENT TO: (name not in the data)")

    lines.append(f"RECOMMENDATION: {c['recommendation']}")
    lines.append(f"LETTER TO REVIEW: {c['to_evaluate']} (the newest letter, from the organisation)")

    # NTSB's latest letter: newest letter from NTSB
    latest_ntsb = None
    for letter in reversed(c["history"]):
        if letter["from"] == "NTSB":
            latest_ntsb = letter["letter_id"]
            break

    if latest_ntsb is not None:
        lines.append(f"NTSB'S LATEST LETTER: {latest_ntsb} (read it to see what NTSB last asked for)")
    else:
        lines.append("NTSB'S LATEST LETTER: none yet (the only request is the recommendation itself)")

    # last grade NTSB gave to OUR case
    grade, from_letter = last_ntsb_grade(c["history"], c["case_id"])
    if grade is not None:
        lines.append(f"LAST GRADE FROM NTSB IN THIS CASE: {grade} (in {from_letter})")
    else:
        lines.append("LAST GRADE FROM NTSB IN THIS CASE: none found in the letters")

    # letter list (no text)
    history = c["history"]
    lines.append(f"LETTERS ({len(history)}):")
    if len(history) <= 15:
        for letter in history:
            lines.append("  " + letter_line(letter))
    else:
        # long cases: first 3 + last 12, so the summary stays small
        for letter in history[:3]:
            lines.append("  " + letter_line(letter))
        lines.append(f"  ... {len(history) - 15} older letters not listed, use search_history ...")
        for letter in history[-12:]:
            lines.append("  " + letter_line(letter))

    return "\n".join(lines)


desk.TOOLS["get_recommendation"] = get_recommendation
