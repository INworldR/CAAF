#!/usr/bin/env python3
"""
Conversation Manager for CAAF - JSONL-based conversation logging and management.

This module provides comprehensive conversation storage, retrieval, and search
capabilities using JSONL (JSON Lines) format for efficient streaming and
append-only operations.

Features:
- JSONL-based conversation storage
- Streaming read/write operations
- Full-text search capabilities
- Conversation metadata management
- Efficient indexing and querying
- Backup and recovery support
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import jsonlines

from ..core.exceptions import (
    ConversationLoadError,
    ConversationSaveError,
    PersistenceError,
    SearchIndexError,
)


class ConversationManager:
    """
    Manages conversation persistence using JSONL format.

    This class provides efficient storage and retrieval of conversation data
    using JSON Lines format, which allows for streaming operations and
    append-only writes.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        conversations_subdir: str = "conversations",
        auto_create_dirs: bool = True,
    ) -> None:
        """
        Initialize the conversation manager.

        Args:
            data_dir: Base directory for data storage
            conversations_subdir: Subdirectory name for conversations
            auto_create_dirs: Whether to automatically create directories

        Raises:
            PersistenceError: If directory setup fails
        """
        try:
            self.data_dir = Path(data_dir)
            self.conversations_dir = self.data_dir / conversations_subdir
            self.logger = logging.getLogger("caaf.persistence.conversations")

            if auto_create_dirs:
                self._ensure_directories()

            self.logger.info(
                f"ConversationManager initialized with data_dir: {self.data_dir}"
            )

        except Exception as e:
            raise PersistenceError(
                f"Failed to initialize ConversationManager: {e}",
                details={
                    "data_dir": str(data_dir),
                    "conversations_subdir": conversations_subdir,
                },
                original_error=e,
            ) from e

    def _ensure_directories(self) -> None:
        """
        Ensure required directories exist.

        Raises:
            PersistenceError: If directory creation fails
        """
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.conversations_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directories exist: {self.conversations_dir}")
        except Exception as e:
            raise PersistenceError(
                f"Failed to create directories: {e}",
                details={
                    "data_dir": str(self.data_dir),
                    "conversations_dir": str(self.conversations_dir),
                },
                original_error=e,
            ) from e

    def _get_conversation_file_path(
        self, agent_id: str, conversation_id: str | UUID
    ) -> Path:
        """
        Get the file path for a conversation.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier

        Returns:
            Path to the conversation file
        """
        if isinstance(conversation_id, UUID):
            conversation_id = str(conversation_id)

        # Sanitize agent_id for filesystem
        safe_agent_id = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_id)

        # Use agent-specific subdirectories for organization
        agent_dir = self.conversations_dir / safe_agent_id
        agent_dir.mkdir(exist_ok=True)

        return agent_dir / f"{conversation_id}.jsonl"

    def save_conversation(
        self,
        agent_id: str,
        conversation_id: str | UUID,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        append: bool = False,
    ) -> None:
        """
        Save a conversation to JSONL file.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            messages: List of message dictionaries
            metadata: Optional conversation metadata
            append: Whether to append to existing file or overwrite

        Raises:
            ConversationSaveError: If saving fails
        """
        try:
            file_path = self._get_conversation_file_path(agent_id, conversation_id)

            # Prepare conversation data
            conversation_data = {
                "conversation_id": str(conversation_id),
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
                "message_count": len(messages),
            }

            mode = "a" if append else "w"

            with jsonlines.open(file_path, mode=mode) as writer:
                # Write conversation metadata (if not appending or file is new)
                if not append or not file_path.exists():
                    writer.write(  # type: ignore[union-attr]
                        {"type": "conversation_metadata", "data": conversation_data}
                    )

                # Write messages
                for message in messages:
                    message_entry = {
                        "type": "message",
                        "data": message,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    writer.write(message_entry)  # type: ignore[union-attr]

            self.logger.info(
                f"Saved conversation {conversation_id} for agent {agent_id} "
                f"({len(messages)} messages, append={append})"
            )

        except Exception as e:
            raise ConversationSaveError(
                f"Failed to save conversation {conversation_id} for agent {agent_id}: {e}",
                conversation_id=str(conversation_id),
                storage_path=str(file_path) if "file_path" in locals() else None,
                original_error=e,
            ) from e

    def save_message(
        self,
        agent_id: str,
        conversation_id: str | UUID,
        message: dict[str, Any],
    ) -> None:
        """
        Save a single message to an existing conversation.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            message: Message dictionary

        Raises:
            ConversationSaveError: If saving fails
        """
        try:
            file_path = self._get_conversation_file_path(agent_id, conversation_id)

            if not file_path.exists():
                raise ConversationSaveError(
                    f"Conversation file does not exist: {conversation_id}",
                    conversation_id=str(conversation_id),
                    storage_path=str(file_path),
                )

            message_entry = {
                "type": "message",
                "data": message,
                "timestamp": datetime.utcnow().isoformat(),
            }

            with jsonlines.open(file_path, mode="a") as writer:
                writer.write(message_entry)

            self.logger.debug(f"Appended message to conversation {conversation_id}")

        except Exception as e:
            if isinstance(e, ConversationSaveError):
                raise
            raise ConversationSaveError(
                f"Failed to save message to conversation {conversation_id}: {e}",
                conversation_id=str(conversation_id),
                original_error=e,
            ) from e

    def load_conversation(
        self,
        agent_id: str,
        conversation_id: str | UUID,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Load a complete conversation from JSONL file.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            include_metadata: Whether to include conversation metadata

        Returns:
            Dictionary with conversation data and messages

        Raises:
            ConversationLoadError: If loading fails
        """
        try:
            file_path = self._get_conversation_file_path(agent_id, conversation_id)

            if not file_path.exists():
                raise ConversationLoadError(
                    f"Conversation file not found: {conversation_id}",
                    conversation_id=str(conversation_id),
                    storage_path=str(file_path),
                )

            conversation_data: dict[str, Any] = {
                "conversation_id": str(conversation_id),
                "agent_id": agent_id,
                "messages": [],
                "metadata": {},
            }

            with jsonlines.open(file_path, mode="r") as reader:
                for entry in reader:
                    if (
                        entry.get("type") == "conversation_metadata"
                        and include_metadata
                    ):
                        conversation_data["metadata"] = entry.get("data", {}).get(
                            "metadata", {}
                        )
                    elif entry.get("type") == "message":
                        messages_list = conversation_data["messages"]
                        if isinstance(messages_list, list):
                            messages_list.append(entry.get("data", {}))

            self.logger.debug(
                f"Loaded conversation {conversation_id} with {len(conversation_data['messages'])} messages"
            )

            return conversation_data

        except Exception as e:
            if isinstance(e, ConversationLoadError):
                raise
            raise ConversationLoadError(
                f"Failed to load conversation {conversation_id}: {e}",
                conversation_id=str(conversation_id),
                storage_path=str(file_path) if "file_path" in locals() else None,
                original_error=e,
            ) from e

    def load_conversation_messages(
        self,
        agent_id: str,
        conversation_id: str | UUID,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Load messages from a conversation with pagination.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of message dictionaries

        Raises:
            ConversationLoadError: If loading fails
        """
        try:
            file_path = self._get_conversation_file_path(agent_id, conversation_id)

            if not file_path.exists():
                raise ConversationLoadError(
                    f"Conversation file not found: {conversation_id}",
                    conversation_id=str(conversation_id),
                    storage_path=str(file_path),
                )

            messages: list[dict[str, Any]] = []
            message_count = 0

            with jsonlines.open(file_path, mode="r") as reader:
                for entry in reader:
                    if entry.get("type") == "message":
                        if message_count >= offset:
                            messages.append(entry.get("data", {}))
                            if limit and len(messages) >= limit:
                                break
                        message_count += 1

            return messages

        except Exception as e:
            if isinstance(e, ConversationLoadError):
                raise
            raise ConversationLoadError(
                f"Failed to load messages from conversation {conversation_id}: {e}",
                conversation_id=str(conversation_id),
                original_error=e,
            ) from e

    def list_conversations(
        self,
        agent_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all conversations for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of conversations to return

        Returns:
            List of conversation metadata dictionaries

        Raises:
            ConversationLoadError: If listing fails
        """
        try:
            safe_agent_id = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_id)
            agent_dir = self.conversations_dir / safe_agent_id

            if not agent_dir.exists():
                return []

            conversations = []

            # Get all .jsonl files in agent directory
            jsonl_files = list(agent_dir.glob("*.jsonl"))

            # Sort by modification time (newest first)
            jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            if limit:
                jsonl_files = jsonl_files[:limit]

            for file_path in jsonl_files:
                try:
                    conversation_id = file_path.stem

                    # Read first entry to get metadata
                    with jsonlines.open(file_path, mode="r") as reader:
                        for entry in reader:
                            if entry.get("type") == "conversation_metadata":
                                metadata = entry.get("data", {})
                                conversations.append(
                                    {
                                        "conversation_id": conversation_id,
                                        "agent_id": agent_id,
                                        "file_path": str(file_path),
                                        "last_modified": datetime.fromtimestamp(
                                            file_path.stat().st_mtime
                                        ).isoformat(),
                                        **metadata,
                                    }
                                )
                                break
                        else:
                            # No metadata found, create basic entry
                            conversations.append(
                                {
                                    "conversation_id": conversation_id,
                                    "agent_id": agent_id,
                                    "file_path": str(file_path),
                                    "last_modified": datetime.fromtimestamp(
                                        file_path.stat().st_mtime
                                    ).isoformat(),
                                    "metadata": {},
                                }
                            )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to read conversation file {file_path}: {e}"
                    )
                    continue

            return conversations

        except Exception as e:
            raise ConversationLoadError(
                f"Failed to list conversations for agent {agent_id}: {e}",
                original_error=e,
            ) from e

    def search_conversations(
        self,
        agent_id: str,
        query: str,
        limit: int = 50,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search conversations by content.

        Args:
            agent_id: Agent identifier
            query: Search query string
            limit: Maximum number of results
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matching conversations with context

        Raises:
            SearchIndexError: If search fails
        """
        try:
            search_flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(re.escape(query), search_flags)

            conversations = self.list_conversations(agent_id)
            results = []

            for conversation_info in conversations:
                conversation_id = conversation_info["conversation_id"]

                try:
                    # Load conversation messages
                    messages = self.load_conversation_messages(
                        agent_id, conversation_id
                    )

                    # Search through messages
                    matching_messages = []
                    for i, message in enumerate(messages):
                        content = message.get("content", "")
                        if isinstance(content, str) and pattern.search(content):
                            matching_messages.append(
                                {
                                    "message_index": i,
                                    "message": message,
                                    "match_context": self._get_match_context(
                                        content, query, case_sensitive
                                    ),
                                }
                            )

                    if matching_messages:
                        results.append(
                            {
                                **conversation_info,
                                "matches": matching_messages,
                                "match_count": len(matching_messages),
                            }
                        )

                        if len(results) >= limit:
                            break

                except Exception as e:
                    self.logger.warning(
                        f"Error searching conversation {conversation_id}: {e}"
                    )
                    continue

            self.logger.info(
                f"Search for '{query}' returned {len(results)} conversations"
            )
            return results

        except Exception as e:
            raise SearchIndexError(
                f"Failed to search conversations for agent {agent_id}: {e}",
                query=query,
                operation="search",
                original_error=e,
            ) from e

    def _get_match_context(
        self,
        content: str,
        query: str,
        case_sensitive: bool,
        context_chars: int = 100,
    ) -> str:
        """
        Get context around a search match.

        Args:
            content: Full content string
            query: Search query
            case_sensitive: Whether search is case sensitive
            context_chars: Number of characters to include around match

        Returns:
            Context string with highlighted match
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)

        match = pattern.search(content)
        if not match:
            return content[: context_chars * 2]

        start_pos = max(0, match.start() - context_chars)
        end_pos = min(len(content), match.end() + context_chars)

        context = content[start_pos:end_pos]

        # Add ellipsis if truncated
        if start_pos > 0:
            context = "..." + context
        if end_pos < len(content):
            context = context + "..."

        return context

    def delete_conversation(
        self,
        agent_id: str,
        conversation_id: str | UUID,
        backup: bool = True,
    ) -> None:
        """
        Delete a conversation file.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            backup: Whether to create a backup before deletion

        Raises:
            ConversationSaveError: If deletion fails
        """
        try:
            file_path = self._get_conversation_file_path(agent_id, conversation_id)

            if not file_path.exists():
                self.logger.warning(
                    f"Conversation file not found for deletion: {conversation_id}"
                )
                return

            if backup:
                backup_path = file_path.with_suffix(
                    f".backup.{int(datetime.utcnow().timestamp())}.jsonl"
                )
                file_path.rename(backup_path)
                self.logger.info(f"Backed up conversation to {backup_path}")
            else:
                file_path.unlink()
                self.logger.info(f"Deleted conversation {conversation_id}")

        except Exception as e:
            raise ConversationSaveError(
                f"Failed to delete conversation {conversation_id}: {e}",
                conversation_id=str(conversation_id),
                original_error=e,
            ) from e

    def get_conversation_stats(self, agent_id: str) -> dict[str, Any]:
        """
        Get statistics about conversations for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with conversation statistics

        Raises:
            PersistenceError: If stats calculation fails
        """
        try:
            safe_agent_id = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_id)
            agent_dir = self.conversations_dir / safe_agent_id

            if not agent_dir.exists():
                return {
                    "total_conversations": 0,
                    "total_messages": 0,
                    "total_size_bytes": 0,
                    "oldest_conversation": None,
                    "newest_conversation": None,
                }

            jsonl_files = list(agent_dir.glob("*.jsonl"))
            total_conversations = len(jsonl_files)
            total_messages = 0
            total_size = 0

            oldest_time = None
            newest_time = None

            for file_path in jsonl_files:
                try:
                    file_stat = file_path.stat()
                    total_size += file_stat.st_size

                    mtime = file_stat.st_mtime
                    if oldest_time is None or mtime < oldest_time:
                        oldest_time = mtime
                    if newest_time is None or mtime > newest_time:
                        newest_time = mtime

                    # Count messages in file
                    with jsonlines.open(file_path, mode="r") as reader:
                        for entry in reader:
                            if entry.get("type") == "message":
                                total_messages += 1

                except Exception as e:
                    self.logger.warning(f"Error processing file {file_path}: {e}")
                    continue

            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "total_size_bytes": total_size,
                "oldest_conversation": (
                    datetime.fromtimestamp(oldest_time).isoformat()
                    if oldest_time
                    else None
                ),
                "newest_conversation": (
                    datetime.fromtimestamp(newest_time).isoformat()
                    if newest_time
                    else None
                ),
            }

        except Exception as e:
            raise PersistenceError(
                f"Failed to calculate conversation stats for agent {agent_id}: {e}",
                original_error=e,
            ) from e

    def cleanup_old_conversations(
        self,
        agent_id: str,
        days_old: int = 30,
        backup: bool = True,
    ) -> int:
        """
        Clean up old conversation files.

        Args:
            agent_id: Agent identifier
            days_old: Delete files older than this many days
            backup: Whether to backup before deletion

        Returns:
            Number of files cleaned up

        Raises:
            PersistenceError: If cleanup fails
        """
        try:
            safe_agent_id = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_id)
            agent_dir = self.conversations_dir / safe_agent_id

            if not agent_dir.exists():
                return 0

            cutoff_time = datetime.utcnow().timestamp() - (days_old * 24 * 60 * 60)
            cleaned_count = 0

            for file_path in agent_dir.glob("*.jsonl"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        if backup:
                            backup_dir = agent_dir / "backup"
                            backup_dir.mkdir(exist_ok=True)
                            backup_path = (
                                backup_dir
                                / f"{file_path.stem}.backup.{int(cutoff_time)}.jsonl"
                            )
                            file_path.rename(backup_path)
                        else:
                            file_path.unlink()

                        cleaned_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup file {file_path}: {e}")
                        continue

            self.logger.info(
                f"Cleaned up {cleaned_count} old conversations for agent {agent_id}"
            )
            return cleaned_count

        except Exception as e:
            raise PersistenceError(
                f"Failed to cleanup old conversations for agent {agent_id}: {e}",
                original_error=e,
            ) from e

    def stream_conversation_messages(
        self,
        agent_id: str,
        conversation_id: str | UUID,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream messages from a conversation file.

        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier

        Yields:
            Message dictionaries

        Raises:
            ConversationLoadError: If streaming fails
        """
        try:
            file_path = self._get_conversation_file_path(agent_id, conversation_id)

            if not file_path.exists():
                raise ConversationLoadError(
                    f"Conversation file not found: {conversation_id}",
                    conversation_id=str(conversation_id),
                    storage_path=str(file_path),
                )

            with jsonlines.open(file_path, mode="r") as reader:
                for entry in reader:
                    if entry.get("type") == "message":
                        yield entry.get("data", {})

        except Exception as e:
            if isinstance(e, ConversationLoadError):
                raise
            raise ConversationLoadError(
                f"Failed to stream conversation {conversation_id}: {e}",
                conversation_id=str(conversation_id),
                original_error=e,
            ) from e


# Export main class
__all__ = ["ConversationManager"]
