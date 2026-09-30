"""Conversation memory for agent runs.

The executor is stateless with respect to history: it asks a
:class:`Memory` for the messages to send on every turn and writes each new
message back.  That keeps "how much context do we keep?" a policy decision
rather than a hard-coded loop detail.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional

from .types import Message, Role


class Memory(ABC):
    """Minimal message store interface used by the agent loop."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append *message* to the store."""

    @abstractmethod
    def messages(self) -> List[Message]:
        """Return the messages to send to the model, oldest first."""

    def extend(self, messages: Iterable[Message]) -> None:
        """Append several messages in order."""
        for message in messages:
            self.add(message)

    def clear(self) -> None:
        """Drop all stored messages."""
        raise NotImplementedError("clear() is not implemented by this memory")


class ConversationMemory(Memory):
    """Bounded, system-message aware conversation buffer.

    System messages are pinned (never evicted) and always returned first.
    The remaining messages form a FIFO buffer capped at ``max_messages``.
    ``on_evict`` is invoked with every evicted message, which is the intended
    hook for rolling summarisation.

    Args:
        messages: Optional seed messages (may include system messages).
        max_messages: Maximum number of non-system messages retained.
        on_evict: Callback invoked once per evicted message.
    """

    def __init__(
        self,
        messages: Optional[Iterable[Message]] = None,
        *,
        max_messages: int = 40,
        on_evict: Optional[Callable[[Message], None]] = None,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages
        self.on_evict = on_evict
        self._system: List[Message] = []
        self._history: List[Message] = []
        #: Messages dropped from the buffer, oldest first.
        self.evicted: List[Message] = []
        if messages:
            self.extend(messages)

    def add(self, message: Message) -> None:
        """Store *message*, evicting the oldest history entry if needed."""
        if not isinstance(message, Message):
            raise TypeError("ConversationMemory only stores Message instances")
        if message.role is Role.SYSTEM:
            self._system.append(message)
            return

        self._history.append(message)
        while len(self._history) > self.max_messages:
            dropped = self._history.pop(0)
            self.evicted.append(dropped)
            if self.on_evict is not None:
                self.on_evict(dropped)

    def messages(self) -> List[Message]:
        """Pinned system messages followed by the rolling history."""
        return [*self._system, *self._history]

    def clear(self) -> None:
        """Drop all messages, including pinned system messages."""
        self._system.clear()
        self._history.clear()

    @property
    def history(self) -> List[Message]:
        """Read-only view of the non-system messages."""
        return list(self._history)

    def __len__(self) -> int:
        return len(self._system) + len(self._history)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(system={len(self._system)}, "
            f"history={len(self._history)}, max_messages={self.max_messages})"
        )
