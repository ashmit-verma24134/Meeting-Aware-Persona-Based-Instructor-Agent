class Decision:
    """
    Centralized decision vocabulary.
    Acts like an enum without runtime overhead.
    """

    CHAT_ONLY = "CHAT_ONLY"
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    NO_EVIDENCE = "NO_EVIDENCE"
    NO_DOMINANT_EVIDENCE = "NO_DOMINANT_EVIDENCE"
    IGNORE = "IGNORE"
