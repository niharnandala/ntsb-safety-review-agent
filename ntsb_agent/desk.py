"""The desk (one open case, answer removed), run_tool (the one door) and the audit log."""
import json
from datetime import datetime, timezone

class Desk:
    def __init__(self, episode):
        # copy the visible part only
        self.case = {}
        for key, value in episode.items():
            if key == "answer" or key.startswith("_"):
                continue
            self.case[key] = value

        # same for every letter: drop the hidden real date
        clean_history = []
        for letter in episode["history"]:
            clean_letter = {}
            for key, value in letter.items():
                if key.startswith("_"):
                    continue
                clean_letter[key] = value
            clean_history.append(clean_letter)
        self.case["history"] = clean_history

        # hidden notes for the tools only
        self._srid = episode["_srid"]
        self._cutoff = episode["_reply_date"]

        # filled later
        self.log = []
        self.draft = None
        self.rules_passed = False
        self.approval = None
        self.peers = {}                 # filled by get_peers: "HI" -> Hawaii's episode

    def letter(self, letter_id):
        wanted = letter_id.strip().upper()          # forgive " l-0003 "
        for letter in self.case["history"]:
            if letter["letter_id"] == wanted:
                return letter
        return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


current = None

def open_case(episode):
    global current
    current = Desk(episode)
    current.log.append({
        "step": 0,
        "time": now(),
        "event": "case opened",
        "case_id": current.case["case_id"],
        "episode_id": current.case["episode_id"],
    })
    return current


TOOLS = {}

def run_tool(name, params):
    if name not in TOOLS:
        result = f"ERROR: there is no tool called {name}."
    else:
        try:
            result = TOOLS[name](**params)          # {"letter_id": "L-0003"} -> letter_id="L-0003"
        except TypeError as err:
            result = f"ERROR: wrong parameters for {name}: {err}"

    current.log.append({
        "step": len(current.log),
        "time": now(),
        "tool": name,
        "params": params,
        "result_chars": len(result),
        "result_preview": result[:300],
    })
    return result


def save_log(open_desk, logs_dir):
    path = f"{logs_dir}/{open_desk.case['episode_id']}.jsonl"
    with open(path, "w") as f:
        for entry in open_desk.log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path
