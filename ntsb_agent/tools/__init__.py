"""Importing this package registers every tool in desk.TOOLS (in the order the agent usually uses them)."""
from ntsb_agent.tools import recommendation, letters, search, precedents, peers, quotes, review
from ntsb_agent.tools.precedents import build_index
from ntsb_agent.tools.peers import group_by_recommendation
from ntsb_agent.tools.review import record_decision, days_late
from ntsb_agent import desk

# put the menu in the order the agent usually uses the tools
MENU_ORDER = ["get_recommendation", "read_letter", "search_history", "find_precedents", "get_peers",
              "verify_quote", "submit_review", "check_rules", "request_approval", "read_attachment"]
ordered = {}
for name in MENU_ORDER:
    ordered[name] = desk.TOOLS[name]
desk.TOOLS.clear()
for name in ordered:
    desk.TOOLS[name] = ordered[name]


def setup(train, val, test):
    # build the precedent index (train only) and the peers lookup (all piles)
    indexed = build_index(train)
    grouped = group_by_recommendation(train + val + test)
    print("precedent index:", indexed, "training episodes |", grouped, "recommendations grouped")
