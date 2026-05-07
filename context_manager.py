import logging
import threading
from typing import List, Dict, Optional
from conversation_history import ConversationHistory

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Thread-safe wrapper around ConversationHistory.
    Provides high-level helpers used by XianyuAgent to build
    LLM prompt context from ongoing conversations.

    Personal note: increased default max_messages from 20 to 50 so longer
    negotiations don't lose early context (e.g. the buyer's original offer).
    Also bumped ttl_seconds to 7200 (2 hours) since some Xianyu chats can
    go quiet for a while before the buyer comes back.
    """

    def __init__(self, max_messages: int = 50, ttl_seconds: int = 7200):
        self._history = ConversationHistory(
            max_messages=max_messages,
            ttl_seconds=ttl_seconds,
        )
        self._lock = threading.Lock()

    def record_user_message(self, item_id: str, user_id: str, content: str) -> None:
        with self._lock:
            self._history.add_message(item_id, user_id, "user", content)
        logger.debug("[ContextManager] user msg recorded item=%s user=%s", item_id, user_id)

    def record_assistant_message(self, item_id: str, user_id: str, content: str) -> None:
        with self._lock:
            self._history.add_message(item_id, user_id, "assistant", content)
        logger.debug("[ContextManager] assistant msg recorded item=%s user=%s", item_id, user_id)

    def build_context(
        self,
        item_id: str,
        user_id: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a full messages list ready for the OpenAI-compatible API.
        Optionally prepends a system prompt.
        """
        with self._lock:
            history = self._history.get_messages(item_id, user_id)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        return messages

    def reset_conversation(self, item_id: str, user_id: str) -> None:
        with self._lock:
            self._history.clear(item_id, user_id)
        logger.info("[ContextManager] conversation reset item=%s user=%s", item_id, user_id)

    def cleanup_expired(self) -> int:
        """Evict stale sessions; intended to be called periodically."""
        with self._lock:
            removed = self._history.evict_expired()
        if removed:
            logger.info("[ContextManager] evicted %d expired sessions", removed)
        return removed

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return self._history.session_count()
