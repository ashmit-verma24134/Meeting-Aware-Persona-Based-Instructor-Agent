# memory/session_memory.py
# Session-level conversational memory
# Stores recent question-answer pairs per session
# NOTE: This memory is NOT a knowledge base and is NOT embedded.

class SessionMemory:
    def __init__(self):
        # Dictionary to store sessions
        # Format:
        # {
        #   session_id: [
        #       {"question": "...", "answer": "..."},
        #       ...
        #   ]
        # }
        self.sessions = {}

    def add_turn(self, session_id, question, answer):
        """
        Store a question-answer turn for a session.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "question": question,
            "answer": answer
        })

    def get_recent_context(self, session_id, k=2):
        """
        Retrieve the last k question-answer pairs for a session.
        Returns an empty list if session does not exist.
        """
        if session_id not in self.sessions:
            return []

        return self.sessions[session_id][-k:]

    def clear_session(self, session_id):
        """
        Clear memory for a specific session.
        Useful when starting a new conversation.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def has_session(self, session_id):
        """
        Check if a session exists in memory.
        """
        return session_id in self.sessions
