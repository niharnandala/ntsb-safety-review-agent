"""find_precedents: BM25 search over the training cases (the RAG tool)."""
import re
from rank_bm25 import BM25Okapi
from ntsb_agent import desk

STOP_WORDS = set("the and for that this with from are was were have has had will would been our your their its "
                 "which also into such than they them these those not but all any can may".split())

def tokens(text):
    # text -> list of search words (lower-case, 3+ letters, no common filler words)
    words = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) >= 3 and word not in STOP_WORDS:
            words.append(word)
    return words

bm25 = None
index_episodes = []


def build_index(train):
    # one search document per training case: the recommendation + the reply NTSB graded
    global bm25, index_episodes
    search_docs = []
    for e in train:
        reply = e["history"][-1]["text"]
        search_docs.append(tokens(e["recommendation"] + " " + reply))
    bm25 = BM25Okapi(search_docs)
    index_episodes = train
    return len(search_docs)


MAIL_HEADER = re.compile(r"Letter Mail Controlled.*?MC#\s*\d+", re.I | re.S)

def short_start(text, n):
    # first n chars, after removing the "Letter Mail Controlled ... MC# 123" header
    text = MAIL_HEADER.sub("", text)
    text = " ".join(text.split())
    if text == "":
        return "(no text besides the mail header)"
    if len(text) <= n:
        return text
    return text[:n] + "..."

def short_end(text, n):
    # last n chars: NTSB's reason + grade are usually at the end
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return "..." + text[-n:]


def find_precedents(query: str, k: int = 5) -> str:
    """
    Find similar past cases and the grade NTSB gave them. Only cases NTSB decided before this reply are used.

    Args:
        query: What to search for, for example the main points of the reply you are reviewing.
        k: How many past cases to return, from 1 to 8.
    """
    # 1. keep k between 1 and 8
    k = int(k)
    if k < 1:
        k = 1
    if k > 8:
        k = 8

    # 2. score every training case against the query
    scores = bm25.get_scores(tokens(query))

    # 3. order the cases from best score to worst
    pairs = []
    for i in range(len(scores)):
        pairs.append((scores[i], i))
    pairs.sort(reverse=True)

    # 4. walk down the list, skip bad ones, stop at k
    picked = []
    for score, i in pairs:
        if score <= 0:
            break                                         # the rest don't match at all
        past = index_episodes[i]
        if past["_srid"] == desk.current._srid:
            continue                                      # same recommendation -> would copy itself
        if past["_answer_date"] >= desk.current._cutoff:
            continue                                      # decided after our reply -> the future
        picked.append(past)
        if len(picked) == k:
            break

    if len(picked) == 0:
        return "No earlier similar cases found."

    # 5. one block per past case
    lines = [f"{len(picked)} similar past cases (NTSB decided all of them before this reply):"]
    number = 1
    for past in picked:
        lines.append(f"{number}. {past['episode_id']} ({past['mode']}) | NTSB GRADE: {past['answer']['label']}")
        lines.append(f"   recommendation: {short_start(past['recommendation'], 160)}")
        lines.append(f"   reply: {short_start(past['history'][-1]['text'], 220)}")
        lines.append(f"   NTSB said: {short_end(past['answer']['ntsb_letter'], 220)}")
        number = number + 1

    return "\n".join(lines)


desk.TOOLS["find_precedents"] = find_precedents
