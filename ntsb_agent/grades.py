"""Find NTSB grades in letter text and turn them into our 7 grade names."""
import re

# old style: the grade sits in quotes -> "Open--Acceptable Response."
QUOTED_GRADE = re.compile(r'["“]\s*((?:Open|Closed)\s*[-–—]+\s*[A-Za-z /]+?)\s*[."”,]', re.I)

# new style (mostly 2010+): no quotes -> OPEN—ACCEPTABLE RESPONSE
# without quotes we can't tell where the grade ends, so only NTSB's real grade words count
PLAIN_GRADE = re.compile(
    r"\b((?:OPEN|CLOSED)\s*[-–—]+\s*"
    r"(?:ACCEPTABLE\s+ALTERNATE\s+(?:RESPONSE|ACTION)"
    r"|UNACCEPTABLE\s+(?:RESPONSE|ACTION)(?:\s*/\s*NO\s+RESPONSE\s+RECEIVED)?"
    r"|ACCEPTABLE\s+(?:RESPONSE|ACTION)"
    r"|EXCEEDS\s+RECOMMENDED\s+ACTION|AWAIT\s+RESPONSE|NO\s+LONGER\s+APPLICABLE"
    r"|SUPERSEDED|SUPERCEDED|RECONSIDERED))",
    re.I,
)

def find_grades(text):
    found = []
    quoted_spans = []

    # 1. grades in quotes
    for m in QUOTED_GRADE.finditer(text):
        found.append((m.start(), m.group(1)))
        quoted_spans.append((m.start(), m.end()))

    # 2. grades without quotes (skip the ones we already found inside quotes)
    for m in PLAIN_GRADE.finditer(text):
        inside_quotes = False
        for start, end in quoted_spans:
            if start <= m.start() < end:
                inside_quotes = True
        if not inside_quotes:
            found.append((m.start(), m.group(1)))

    found.sort()          # reading order, by position
    return found

def to_class(grade):
    g = grade.lower()
    if g.startswith("open"):
        side = "Open"
    else:
        side = "Closed"

    if "await" in g:
        return None                          # "await response" isn't a grade on a reply
    if "supersed" in g or "superced" in g:
        return "Closed – Other"
    if "unacceptable" in g:                  # check before "acceptable" (it contains it)
        return side + " – Unacceptable"
    if "alternate" in g:
        return side + " – Acceptable Alternate"
    if "acceptable" in g or "exceed" in g:
        return side + " – Acceptable"
    if side == "Closed":
        return "Closed – Other"              # no longer applicable, reconsidered
    return None


def grades_for_case(text, case_id):
    found = find_grades(text)
    grades = []
    previous_end = 0

    # 1. keep a grade only if our case ID is in the text just before it
    for position, grade_words in found:
        before = text[previous_end:position]
        if case_id in before:
            clean = to_class(grade_words)
            if clean is not None:
                grades.append(clean)
        previous_end = position + len(grade_words)

    # 2. a letter about only our case often has no ID right before the grade
    if len(grades) == 0 and "[OTHER-REC]" not in text:
        for position, grade_words in found:
            clean = to_class(grade_words)
            if clean is not None:
                grades.append(clean)

    return grades


def last_ntsb_grade(history, case_id):
    # newest NTSB letter that grades our case -> its last grade
    for letter in reversed(history):
        if letter["from"] != "NTSB":
            continue
        grades = grades_for_case(letter["text"], case_id)
        if len(grades) > 0:
            return grades[-1], letter["letter_id"]
    return None, None


def letter_line(letter):
    line = f"{letter['letter_id']} · Day {letter['day']} · from {letter['from']}"
    if letter["days_since_previous"] is not None:
        line = line + f", {letter['days_since_previous']} days after the previous letter"
    line = line + f" · {len(letter['text'])} chars"
    return line
