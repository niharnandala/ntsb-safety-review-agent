from ntsb_agent import desk
from ntsb_agent.tools.letters import find_letter

def verify_quote(letter_id: str, quote: str) -> str:
    """
    Check that a quote really appears, word for word, in one letter. Use it before you rely on a quote.

    Args:
        letter_id: The letter the quote comes from, like "L-0003" or "HI-L-0002".
        quote: The exact words being quoted from that letter.
    """
    # 1. find the letter
    h = find_letter(letter_id)
    if h is None:
        return f"NOT FOUND: there is no letter {letter_id}."

    # 2. clean both the same way: lower-case, one space between words
    clean_quote = " ".join(quote.lower().split())
    clean_text = " ".join(h["text"].lower().split())

    # 3. a tiny quote proves nothing
    if len(clean_quote.split()) < 4:
        return "NOT FOUND: the quote is too short to count as evidence (use at least 4 words)."

    # 4. is it really there?
    if clean_quote in clean_text:
        return f"VERIFIED: the quote is in {letter_id}."
    return f"NOT FOUND: {letter_id} does not contain these exact words."


desk.TOOLS["verify_quote"] = verify_quote
