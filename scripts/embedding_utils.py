def last_n_tokens(text, n=30):   #takes text , split it in words, returns last 30 words
    if not text:
        return ""

    tokens= text.split()
    return " ".join(tokens[-n:])


def build_embedding_text(chunk, prev_chunk=None):  #for overlapping last 30 words

    # First chunk of meeting
    if prev_chunk is None:
        return chunk["text"]

    # Subsequent chunk in same meeting
    overlap= last_n_tokens(prev_chunk["text"], 30)
    return overlap + " " + chunk["text"]
