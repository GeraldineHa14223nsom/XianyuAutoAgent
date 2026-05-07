import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationHistory:
    """
    Manages per-item conversation history for XianyuAutoAgent.
    Stores chat context keyed by (item_id, user_id) to support
    multi-turn dialogue for the LLM agents.
    """

    def __init__(self, max_messages: int = 50, ttl_seconds: int = 7200):
        # Increased max_messages from 20->50 and ttl from 1hr->2hr to better
        # handle longer negotiation sessions without losing context mid-conversation
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._histories: Dict[str, List[Message]] = defaultdict(list)
        self._last_access: Dict[str, float] = {}

    def _make_key(self, item_id: str, user_id: str) -> str:
        return f"{item_id}:{user_id}"

    def add_message(self, item_id: str, user_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        key = self._make_key(item_id, user_id)
        self._last_access[key] = time.time()
        self._histories[key].append(Message(role=role, content=content))
        # Trim to max_messages, keeping system context
        if len(self._histories[key]) > self.max_messages:
            self._histories[key] = self._histories[key][-self.max_messages:]

    def get_messages(self, item_id: str, user_id: str) -> List[Dict[str, str]]:
        """Return conversation history as list of dicts for LLM consumption."""
        key = self._make_key(item_id, user_id)
        self._last_access[key] = time.time()
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._histories[key]
        ]

    def clear(self, item_id: str, user_id: str) -> None:
        """Clear history for a specific conversation."""
        key = self._make_key(item_id, user_id)
        self._histories.pop(key, None)
        self._last_access.pop(key, None)

    def evict_expired(self) -> int:
        """Remove conversations that haven't been accessed within TTL. Returns count removed."""
        now = time.time()
        expired = [
            key for key, last in self._last_access.items()
            if now - last > self.ttl_seconds
        ]
        for key in expired:
            self._histories.pop(key, None)
            self._last_access.pop(key, None)
        return len(expired)

    def session_count(self) -> int:
        return len(self._histories)
