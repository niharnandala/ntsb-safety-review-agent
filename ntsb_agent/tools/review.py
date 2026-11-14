"""days_late (helper), the draft -> rules -> approval tools, and the human-only record_decision."""
from ntsb_agent import desk
from ntsb_agent.settings import GRADES, LETTER_LIMIT, LOW_CONFIDENCE, REPLY_WINDOW
from ntsb_agent.grades import grades_for_case
from ntsb_agent.tools.quotes import verify_quote

def days_late(history, today, case_id):
    # 1. position of NTSB's latest letter
    ntsb_position = -1
    position = 0
    for letter in history:
        if letter["from"] == "NTSB":
            ntsb_position = position
        position = position + 1
    if ntsb_position == -1:
        return 0                                   # NTSB never wrote -> nothing to be late for

    ntsb_letter = history[ntsb_position]

    # 2. if that letter CLOSED our case, there's nothing left to answer
    grades = grades_for_case(ntsb_letter["text"], case_id)
    if len(grades) > 0 and grades[-1].startswith("Closed"):
        return 0

    # 3. first org letter after it (or today, if they haven't answered yet)
    end_day = today
    for letter in history[ntsb_position + 1:]:
        if letter["from"] == "ORG":
            end_day = letter["day"]
            break

    # 4. days past the 90-day window
    late = end_day - ntsb_letter["day"] - REPLY_WINDOW
    if late < 0:
        late = 0
    return late


def submit_review(grade: str, confidence: float, reasons: str, quote_letter_id: str, quote: str, letter: str) -> str:
    """
    Hand in your review draft. Nothing is sent: it is saved for the rule check and human approval.

    Args:
        grade: One of: Open – Acceptable, Open – Acceptable Alternate, Open – Unacceptable, Closed – Acceptable, Closed – Acceptable Alternate, Closed – Unacceptable, Closed – Other.
        confidence: How sure you are about the grade, from 0.0 to 1.0.
        reasons: Two to four sentences on why this grade fits.
        quote_letter_id: The letter your evidence quote comes from, like "L-0003".
        quote: Exact words from that letter (at least 4 words) that support the grade.
        letter: The draft NTSB letter to the organisation, ending with the classification.
    """
    # the AI may type "-" instead of "–": make it match our 7 names
    clean_grade = grade.strip().replace(" - ", " – ")

    desk.current.draft = {
        "grade": clean_grade,
        "confidence": confidence,
        "reasons": reasons,
        "quote_letter_id": quote_letter_id,
        "quote": quote,
        "letter": letter,
    }
    desk.current.rules_passed = False                  # a new draft must be checked again
    desk.current.log.append({"step": len(desk.current.log), "time": desk.now(), "event": "draft saved", "draft": desk.current.draft})
    return "Draft saved. Next: call check_rules."


def check_rules() -> str:
    """
    Run the rule check on your saved draft. Fix anything marked FAIL and submit again before asking for approval.
    """
    d = desk.current.draft
    if d is None:
        return "No draft yet. Call submit_review first."

    results = []          # list of (status, message)

    # rule 1: the grade must be one of our 7
    if d["grade"] in GRADES:
        results.append(("PASS", f"grade is one of the 7 grades ({d['grade']})"))
    else:
        results.append(("FAIL", f"'{d['grade']}' is not one of the 7 grades"))

    # rule 2: the quote must really be in the letter
    check = verify_quote(d["quote_letter_id"], d["quote"])
    if check.startswith("VERIFIED"):
        results.append(("PASS", "evidence quote: " + check))
    else:
        results.append(("FAIL", "evidence quote: " + check))

    # rule 3: confidence is a number from 0 to 1; below 0.6 a human should look closer
    try:
        confidence = float(d["confidence"])
    except (TypeError, ValueError):
        confidence = -1
    if confidence < 0 or confidence > 1:
        results.append(("FAIL", "confidence must be a number from 0 to 1"))
    elif confidence < LOW_CONFIDENCE:
        results.append(("WARN", f"low confidence ({confidence}): a human should look closely"))
    else:
        results.append(("PASS", f"confidence {confidence}"))

    # rule 4: the letter names the case and isn't too long
    if desk.current.case["case_id"] in d["letter"]:
        results.append(("PASS", "letter names the case ID"))
    else:
        results.append(("FAIL", f"letter must name the case ID ({desk.current.case['case_id']})"))
    if len(d["letter"]) <= LETTER_LIMIT:
        results.append(("PASS", f"letter length {len(d['letter'])} chars (limit {LETTER_LIMIT})"))
    else:
        results.append(("FAIL", f"letter is {len(d['letter'])} chars, limit is {LETTER_LIMIT}"))

    # rule 5: late reply -> warn, so the letter can mention it
    history = desk.current.case["history"]
    today = history[-1]["day"]
    late = days_late(history, today, desk.current.case["case_id"])
    if late > 0:
        results.append(("WARN", f"reply came {late} days after the 90-day window"))
    else:
        results.append(("PASS", "reply came within 90 days"))

    # rule 6: who must approve
    if d["grade"].startswith("Closed"):
        results.append(("INFO", "approval needed from: analyst + supervisor"))
    else:
        results.append(("INFO", "approval needed from: analyst"))

    # passed = no FAIL anywhere
    passed = True
    for status, message in results:
        if status == "FAIL":
            passed = False
    desk.current.rules_passed = passed
    desk.current.log.append({"step": len(desk.current.log), "time": desk.now(), "event": "rules checked",
                        "results": results, "passed": passed})

    lines = []
    for status, message in results:
        lines.append(f"{status}: {message}")
    if passed:
        lines.append("READY for approval.")
    else:
        lines.append("NOT READY: fix the FAIL items and submit again.")
    return "\n".join(lines)


def request_approval() -> str:
    """
    Send your checked draft to the human reviewers. Only works after check_rules says READY.
    """
    if desk.current.draft is None:
        return "Refused: no draft. Call submit_review first."
    if not desk.current.rules_passed:
        return "Refused: the draft has not passed check_rules."

    if desk.current.draft["grade"].startswith("Closed"):
        level = "analyst + supervisor"
    else:
        level = "analyst"
    desk.current.approval = {"level": level, "status": "pending", "decisions": []}
    desk.current.log.append({"step": len(desk.current.log), "time": desk.now(), "event": "approval requested", "level": level})
    return f"Sent for approval to: {level}. Status: pending. Your part is done."


def read_attachment(name: str) -> str:
    """
    Try to open an attachment mentioned in a letter.

    Args:
        name: What the letter calls the attachment, for example "enclosed report".
    """
    desk.current.log.append({"step": len(desk.current.log), "time": desk.now(), "event": "attachment flagged", "name": name})
    return (f"Attachment '{name}' is not available in this system. Flagged for the analyst to check by hand. "
            "Do not guess what it says.")


desk.TOOLS["submit_review"] = submit_review
desk.TOOLS["check_rules"] = check_rules
desk.TOOLS["request_approval"] = request_approval
desk.TOOLS["read_attachment"] = read_attachment


def record_decision(role, approved, note=""):
    # the human's click. NOT a tool: it's not in TOOLS, so the AI can't call it
    desk.current.approval["decisions"].append({"role": role, "approved": approved, "note": note, "time": desk.now()})

    if desk.current.approval["level"] == "analyst + supervisor":
        needed = ["analyst", "supervisor"]
    else:
        needed = ["analyst"]

    said_yes = []
    for decision in desk.current.approval["decisions"]:
        if decision["approved"]:
            said_yes.append(decision["role"])

    if not approved:
        desk.current.approval["status"] = "rejected"
    else:
        everyone = True
        for role_needed in needed:
            if role_needed not in said_yes:
                everyone = False
        if everyone:
            desk.current.approval["status"] = "approved"

    desk.current.log.append({"step": len(desk.current.log), "time": desk.now(), "event": "human decision",
                        "role": role, "approved": approved, "note": note, "status": desk.current.approval["status"]})
    return desk.current.approval["status"]
