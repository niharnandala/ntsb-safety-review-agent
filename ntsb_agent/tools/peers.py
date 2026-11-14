"""get_peers: how NTSB handled the other organisations that got the same recommendation."""
from datetime import date
from ntsb_agent import desk
from ntsb_agent.grades import grades_for_case
from ntsb_agent.tools.precedents import short_start, short_end

by_srid = {}


def group_by_recommendation(all_episodes):
    # all episodes of all piles, grouped by recommendation (real ID)
    by_srid.clear()
    for e in all_episodes:
        srid = e["_srid"]
        if srid not in by_srid:
            by_srid[srid] = []
        by_srid[srid].append(e)
    return len(by_srid)


MAX_PEERS = 5

def days_between(a, b):
    # "2006-04-20", "2006-07-06" -> 77
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


def peer_letters(past):
    # a peer's letters: its history + NTSB's answer to it (the grade letter), all before our date
    letters = []
    for letter in past["history"]:
        letters.append(letter)

    reply = past["history"][-1]
    gap = days_between(past["_reply_date"], past["_answer_date"])
    answer_letter = {
        "letter_id": f"L-{len(past['history']) + 1:04d}",
        "day": reply["day"] + gap,
        "days_since_previous": gap,
        "from": "NTSB",
        "text": past["answer"]["ntsb_letter"],
    }
    letters.append(answer_letter)
    return letters


def get_peers() -> str:
    """
    See how NTSB handled the other organisations that got this same recommendation, before this reply:
    their letters and the grade NTSB gave at each step. Read a peer letter with read_letter, for example "HI-L-0002".
    """
    # 1. latest episode of every OTHER organisation, answered before our reply date
    latest = {}
    unknown = 0
    for past in by_srid[desk.current._srid]:
        if past["recipient"] == desk.current.case["recipient"]:
            continue                              # skip our own organisation
        if past["_answer_date"] >= desk.current._cutoff:
            continue                              # decided after our reply -> the future

        code = past["recipient"]
        if code == "":
            unknown = unknown + 1
            code = f"ORG{unknown}"                # name not in the data

        if code not in latest:
            latest[code] = past
        elif past["_answer_date"] > latest[code]["_answer_date"]:
            latest[code] = past

    if len(latest) == 0:
        return "No other organisation got a grade for this recommendation before this reply."

    # 2. count the grades of all peers
    counts = {}
    for code in latest:
        label = latest[code]["answer"]["label"]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    count_texts = []
    for label in counts:
        count_texts.append(f"{counts[label]} × {label}")

    # 3. the most recent peers first, keep 5
    order = []
    for code in latest:
        order.append((latest[code]["_answer_date"], code))
    order.sort(reverse=True)

    desk.current.peers = {}
    lines = [f"{len(latest)} other organisations got this recommendation. Latest grades before this reply: " + ", ".join(count_texts)]
    lines.append(f"The {min(MAX_PEERS, len(order))} most recent (read any letter with read_letter, e.g. \"{order[0][1]}-L-0001\"):")

    shown = 0
    for answered, code in order:
        if shown == MAX_PEERS:
            break
        past = latest[code]
        desk.current.peers[code] = past
        letters = peer_letters(past)
        lines.append(f"{code} · {len(letters)} letters")
        for letter in letters:
            line = f"   {code}-{letter['letter_id']} · Day {letter['day']} · {letter['from']}"
            if letter["from"] == "NTSB":
                grades = grades_for_case(letter["text"], desk.current.case["case_id"])
                if len(grades) > 0:
                    line = line + f" → {grades[-1]}"
            lines.append(line)

        # previews, so the AI can decide which letter is worth opening
        last_reply = past["history"][-1]
        lines.append(f"   their reply: {short_start(last_reply['text'], 120)}")
        lines.append(f"   NTSB said: {short_end(past['answer']['ntsb_letter'], 150)}")
        shown = shown + 1

    return "\n".join(lines)


desk.TOOLS["get_peers"] = get_peers
