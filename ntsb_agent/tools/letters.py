from ntsb_agent import desk
from ntsb_agent.settings import PAGE_SIZE
from ntsb_agent.tools.peers import peer_letters

def find_letter(letter_id):
    # our own letter: "L-0003"; a peer's letter: "HI-L-0002" (only after get_peers)
    letter_id = letter_id.strip().upper()

    if letter_id.startswith("L-"):
        return desk.current.letter(letter_id)

    if "-L-" in letter_id:
        code, number = letter_id.split("-L-", 1)
        if code not in desk.current.peers:
            return None
        for letter in peer_letters(desk.current.peers[code]):
            if letter["letter_id"] == "L-" + number:
                return letter
    return None


def read_letter(letter_id: str, page: int = 1) -> str:
    """
    Read one letter, one page at a time. Your own case's letters look like "L-0003".
    Letters of other organisations (from get_peers) look like "HI-L-0002".

    Args:
        letter_id: The letter to read, like "L-0003" or "HI-L-0002".
        page: Which page to read, starting at 1. Long letters have more than one page.
    """
    h = find_letter(letter_id)
    if h is None:
        return (f"No letter called {letter_id}. Use get_recommendation for this case's letters, "
                "or get_peers first for other organisations' letters.")
    text = h["text"]

    # 1. how many pages (4,000 chars each, rounded up)
    pages = len(text) // PAGE_SIZE
    if len(text) % PAGE_SIZE != 0:
        pages = pages + 1
    if pages == 0:
        pages = 1

    # 2. page out of range
    if page < 1 or page > pages:
        return f"{letter_id} has {pages} page(s). Ask for page 1 to {pages}."

    # 3. this page's slice
    start = (page - 1) * PAGE_SIZE
    end = page * PAGE_SIZE
    chunk = text[start:end]

    # 4. header + slice
    header = f"{letter_id.strip().upper()} · Day {h['day']} · from {h['from']} · {len(text)} chars | page {page} of {pages}"
    return header + "\n\n" + chunk


desk.TOOLS["read_letter"] = read_letter
