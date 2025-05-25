#!/usr/bin/env python3
"""
Agent Memory Module - Pattern learning and knowledge retention for CAAF agents.

This module provides intelligent memory management for agents, including pattern
recognition, learning mechanisms, memory consolidation, and context-based retrieval.
The AgentMemory class enables agents to learn from conversations and improve over time.

Features:
- Pattern recognition from conversation history
- Learning mechanisms that improve over time
- Memory consolidation and summarization
- Context-based knowledge retrieval
- Memory decay and importance scoring
- Efficient storage and indexing
- Integration with ConversationManager
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID, uuid4

from ..core.exceptions import AgentMemoryError
from .conversation_manager import ConversationManager


class MemoryPattern(NamedTuple):
    """Represents a learned pattern from conversations."""

    pattern_id: str
    pattern_type: str  # 'topic', 'response', 'behavior', 'preference'
    content: str
    frequency: int
    confidence: float
    last_seen: datetime
    created_at: datetime
    importance: float
    context_tags: list[str]


class KnowledgeItem(NamedTuple):
    """Represents a piece of learned knowledge."""

    knowledge_id: str
    content: str
    source_type: str  # 'conversation', 'pattern', 'summary'
    source_id: str
    confidence: float
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    context_tags: list[str]
    related_items: list[str]


class MemoryContext(NamedTuple):
    """Context for memory retrieval and storage."""

    topic: str | None = None
    user_intent: str | None = None
    conversation_type: str | None = None
    keywords: list[str] | None = None
    timeframe: str | None = None
    importance_threshold: float = 0.1


class AgentMemory:
    """
    Intelligent memory system for CAAF agents.

    This class provides pattern learning, knowledge retention, and context-based
    retrieval capabilities. It learns from conversation history and improves
    agent responses over time.
    """

    def __init__(
        self,
        agent_id: str,
        memory_dir: str | Path = "./data/memory",
        conversation_manager: ConversationManager | None = None,
        max_patterns: int = 1000,
        max_knowledge_items: int = 5000,
        decay_factor: float = 0.95,
        consolidation_threshold: int = 10,
    ) -> None:
        """
        Initialize the agent memory system.

        Args:
            agent_id: Unique identifier for the agent
            memory_dir: Directory for memory storage
            conversation_manager: ConversationManager instance for data access
            max_patterns: Maximum number of patterns to store
            max_knowledge_items: Maximum number of knowledge items to store
            decay_factor: Factor for memory decay (0.0-1.0)
            consolidation_threshold: Number of related items needed for consolidation

        Raises:
            AgentMemoryError: If initialization fails
        """
        try:
            self.agent_id = agent_id
            self.memory_dir = Path(memory_dir)
            self.conversation_manager = conversation_manager or ConversationManager()
            self.max_patterns = max_patterns
            self.max_knowledge_items = max_knowledge_items
            self.decay_factor = decay_factor
            self.consolidation_threshold = consolidation_threshold

            self.logger = logging.getLogger(f"caaf.memory.{agent_id}")

            # Initialize storage
            self._ensure_memory_directories()

            # In-memory caches for efficiency
            self._patterns: dict[str, MemoryPattern] = {}
            self._knowledge: dict[str, KnowledgeItem] = {}
            self._topic_index: dict[str, set[str]] = defaultdict(set)
            self._keyword_index: dict[str, set[str]] = defaultdict(set)

            # Load existing memory
            self._load_memory()

            self.logger.info(f"AgentMemory initialized for agent {agent_id}")

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to initialize AgentMemory for agent {agent_id}: {e}",
                agent_id=agent_id,
                operation="initialization",
                original_error=e,
            ) from e

    def _ensure_memory_directories(self) -> None:
        """Ensure memory directories exist."""
        try:
            agent_memory_dir = self.memory_dir / self.agent_id
            agent_memory_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (agent_memory_dir / "patterns").mkdir(exist_ok=True)
            (agent_memory_dir / "knowledge").mkdir(exist_ok=True)
            (agent_memory_dir / "summaries").mkdir(exist_ok=True)

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to create memory directories: {e}",
                agent_id=self.agent_id,
                original_error=e,
            ) from e

    def _get_memory_file_path(self, memory_type: str) -> Path:
        """Get path to memory file."""
        return self.memory_dir / self.agent_id / f"{memory_type}.json"

    def _load_memory(self) -> None:
        """Load existing memory from storage."""
        try:
            # Load patterns
            patterns_file = self._get_memory_file_path("patterns")
            if patterns_file.exists():
                with open(patterns_file, encoding="utf-8") as f:
                    patterns_data = json.load(f)
                    for pattern_data in patterns_data:
                        pattern = MemoryPattern(
                            pattern_id=pattern_data["pattern_id"],
                            pattern_type=pattern_data["pattern_type"],
                            content=pattern_data["content"],
                            frequency=pattern_data["frequency"],
                            confidence=pattern_data["confidence"],
                            last_seen=datetime.fromisoformat(pattern_data["last_seen"]),
                            created_at=datetime.fromisoformat(
                                pattern_data["created_at"]
                            ),
                            importance=pattern_data["importance"],
                            context_tags=pattern_data["context_tags"],
                        )
                        self._patterns[pattern.pattern_id] = pattern

            # Load knowledge
            knowledge_file = self._get_memory_file_path("knowledge")
            if knowledge_file.exists():
                with open(knowledge_file, encoding="utf-8") as f:
                    knowledge_data = json.load(f)
                    for item_data in knowledge_data:
                        item = KnowledgeItem(
                            knowledge_id=item_data["knowledge_id"],
                            content=item_data["content"],
                            source_type=item_data["source_type"],
                            source_id=item_data["source_id"],
                            confidence=item_data["confidence"],
                            importance=item_data["importance"],
                            created_at=datetime.fromisoformat(item_data["created_at"]),
                            last_accessed=datetime.fromisoformat(
                                item_data["last_accessed"]
                            ),
                            access_count=item_data["access_count"],
                            context_tags=item_data["context_tags"],
                            related_items=item_data["related_items"],
                        )
                        self._knowledge[item.knowledge_id] = item

            # Rebuild indexes
            self._rebuild_indexes()

            self.logger.debug(
                f"Loaded {len(self._patterns)} patterns and {len(self._knowledge)} knowledge items"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load existing memory: {e}")
            # Continue with empty memory

    def _rebuild_indexes(self) -> None:
        """Rebuild search indexes from loaded data."""
        self._topic_index.clear()
        self._keyword_index.clear()

        # Index patterns
        for pattern in self._patterns.values():
            for tag in pattern.context_tags:
                self._topic_index[tag.lower()].add(pattern.pattern_id)

            # Index keywords from content
            keywords = self._extract_keywords(pattern.content)
            for keyword in keywords:
                self._keyword_index[keyword.lower()].add(pattern.pattern_id)

        # Index knowledge items
        for item in self._knowledge.values():
            for tag in item.context_tags:
                self._topic_index[tag.lower()].add(item.knowledge_id)

            keywords = self._extract_keywords(item.content)
            for keyword in keywords:
                self._keyword_index[keyword.lower()].add(item.knowledge_id)

    def _save_memory(self) -> None:
        """Save memory to storage."""
        try:
            # Save patterns
            patterns_data = []
            for pattern in self._patterns.values():
                patterns_data.append(
                    {
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type,
                        "content": pattern.content,
                        "frequency": pattern.frequency,
                        "confidence": pattern.confidence,
                        "last_seen": pattern.last_seen.isoformat(),
                        "created_at": pattern.created_at.isoformat(),
                        "importance": pattern.importance,
                        "context_tags": pattern.context_tags,
                    }
                )

            patterns_file = self._get_memory_file_path("patterns")
            with open(patterns_file, "w", encoding="utf-8") as f:
                json.dump(patterns_data, f, indent=2)

            # Save knowledge
            knowledge_data = []
            for item in self._knowledge.values():
                knowledge_data.append(
                    {
                        "knowledge_id": item.knowledge_id,
                        "content": item.content,
                        "source_type": item.source_type,
                        "source_id": item.source_id,
                        "confidence": item.confidence,
                        "importance": item.importance,
                        "created_at": item.created_at.isoformat(),
                        "last_accessed": item.last_accessed.isoformat(),
                        "access_count": item.access_count,
                        "context_tags": item.context_tags,
                        "related_items": item.related_items,
                    }
                )

            knowledge_file = self._get_memory_file_path("knowledge")
            with open(knowledge_file, "w", encoding="utf-8") as f:
                json.dump(knowledge_data, f, indent=2)

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to save memory: {e}",
                agent_id=self.agent_id,
                operation="save",
                original_error=e,
            ) from e

    def learn_from_conversations(
        self,
        conversation_ids: list[str | UUID] | None = None,
        max_conversations: int = 10,
    ) -> dict[str, Any]:
        """
        Learn patterns and knowledge from conversation history.

        Args:
            conversation_ids: Specific conversations to learn from (if None, uses recent ones)
            max_conversations: Maximum number of conversations to analyze

        Returns:
            Learning statistics

        Raises:
            AgentMemoryError: If learning fails
        """
        try:
            if conversation_ids is None:
                # Get recent conversations
                conversations = self.conversation_manager.list_conversations(
                    self.agent_id, limit=max_conversations
                )
                conversation_ids = [conv["conversation_id"] for conv in conversations]

            learned_patterns = 0
            learned_knowledge = 0

            for conv_id in conversation_ids:
                try:
                    conversation = self.conversation_manager.load_conversation(
                        self.agent_id, conv_id
                    )

                    # Extract patterns from conversation
                    patterns = self._extract_patterns_from_conversation(conversation)
                    for pattern in patterns:
                        if self._add_pattern(pattern):
                            learned_patterns += 1

                    # Extract knowledge from conversation
                    knowledge_items = self._extract_knowledge_from_conversation(
                        conversation
                    )
                    for item in knowledge_items:
                        if self._add_knowledge_item(item):
                            learned_knowledge += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to learn from conversation {conv_id}: {e}"
                    )
                    continue

            # Consolidate memory after learning
            consolidated = self._consolidate_memory()

            # Save updated memory
            self._save_memory()

            stats = {
                "conversations_processed": len(conversation_ids),
                "patterns_learned": learned_patterns,
                "knowledge_items_learned": learned_knowledge,
                "memory_consolidated": consolidated,
                "total_patterns": len(self._patterns),
                "total_knowledge": len(self._knowledge),
            }

            self.logger.info(f"Learning completed: {stats}")
            return stats

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to learn from conversations: {e}",
                agent_id=self.agent_id,
                operation="learn",
                original_error=e,
            ) from e

    def _extract_patterns_from_conversation(
        self, conversation: dict[str, Any]
    ) -> list[MemoryPattern]:
        """Extract patterns from a conversation."""
        patterns: list[MemoryPattern] = []
        messages = conversation.get("messages", [])

        if len(messages) < 2:
            return patterns

        # Topic patterns - what topics are discussed
        topics = self._extract_topics_from_messages(messages)
        for topic in topics:
            pattern = MemoryPattern(
                pattern_id=str(uuid4()),
                pattern_type="topic",
                content=topic,
                frequency=1,
                confidence=0.7,
                last_seen=datetime.utcnow(),
                created_at=datetime.utcnow(),
                importance=0.5,
                context_tags=[topic.lower()],
            )
            patterns.append(pattern)

        # Response patterns - how the agent typically responds
        response_patterns = self._extract_response_patterns(messages)
        patterns.extend(response_patterns)

        # Behavior patterns - interaction styles
        behavior_patterns = self._extract_behavior_patterns(messages)
        patterns.extend(behavior_patterns)

        return patterns

    def _extract_topics_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[str]:
        """Extract topics from conversation messages."""
        topics: set[str] = set()

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                # Simple topic extraction - can be enhanced with NLP
                words = re.findall(r"\b[A-Z][a-z]+\b", content)
                topics.update(words)

        return list(topics)[:5]  # Limit to top 5 topics

    def _extract_response_patterns(
        self, messages: list[dict[str, Any]]
    ) -> list[MemoryPattern]:
        """Extract response patterns from messages."""
        patterns: list[MemoryPattern] = []

        for i in range(len(messages) - 1):
            user_msg = messages[i]
            agent_msg = messages[i + 1]

            if user_msg.get("role") == "user" and agent_msg.get("role") == "assistant":

                user_content = user_msg.get("content", "")
                agent_content = agent_msg.get("content", "")

                if user_content and agent_content:
                    # Create response pattern
                    pattern_content = f"User: {user_content[:100]}... -> Agent: {agent_content[:100]}..."

                    pattern = MemoryPattern(
                        pattern_id=str(uuid4()),
                        pattern_type="response",
                        content=pattern_content,
                        frequency=1,
                        confidence=0.6,
                        last_seen=datetime.utcnow(),
                        created_at=datetime.utcnow(),
                        importance=0.4,
                        context_tags=self._extract_keywords(user_content)[:3],
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_behavior_patterns(
        self, messages: list[dict[str, Any]]
    ) -> list[MemoryPattern]:
        """Extract behavior patterns from messages."""
        patterns: list[MemoryPattern] = []

        agent_messages = [msg for msg in messages if msg.get("role") == "assistant"]

        if len(agent_messages) < 3:
            return patterns

        # Analyze response length patterns
        lengths = [len(msg.get("content", "")) for msg in agent_messages]
        avg_length = sum(lengths) / len(lengths)

        if avg_length > 500:
            behavior = "verbose_responses"
        elif avg_length < 100:
            behavior = "concise_responses"
        else:
            behavior = "balanced_responses"

        pattern = MemoryPattern(
            pattern_id=str(uuid4()),
            pattern_type="behavior",
            content=f"Response style: {behavior} (avg: {avg_length:.0f} chars)",
            frequency=1,
            confidence=0.8,
            last_seen=datetime.utcnow(),
            created_at=datetime.utcnow(),
            importance=0.6,
            context_tags=["response_style", behavior],
        )
        patterns.append(pattern)

        return patterns

    def _extract_knowledge_from_conversation(
        self, conversation: dict[str, Any]
    ) -> list[KnowledgeItem]:
        """Extract knowledge items from a conversation."""
        knowledge_items: list[KnowledgeItem] = []
        messages = conversation.get("messages", [])
        conversation_id = conversation.get("conversation_id", "unknown")

        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                if len(content) > 50:  # Only consider substantial responses

                    # Extract key facts or statements
                    facts = self._extract_facts_from_text(content)

                    for fact in facts:
                        item = KnowledgeItem(
                            knowledge_id=str(uuid4()),
                            content=fact,
                            source_type="conversation",
                            source_id=conversation_id,
                            confidence=0.7,
                            importance=0.5,
                            created_at=datetime.utcnow(),
                            last_accessed=datetime.utcnow(),
                            access_count=1,
                            context_tags=self._extract_keywords(fact)[:3],
                            related_items=[],
                        )
                        knowledge_items.append(item)

        return knowledge_items

    def _extract_facts_from_text(self, text: str) -> list[str]:
        """Extract factual statements from text."""
        # Simple fact extraction - can be enhanced with NLP
        sentences = re.split(r"[.!?]+", text)
        facts: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if (
                len(sentence) > 20
                and len(sentence) < 200
                and not sentence.startswith(("I think", "Maybe", "Perhaps"))
            ):
                facts.append(sentence)

        return facts[:3]  # Limit to top 3 facts per message

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Remove common words
        stop_words = {
            "the",
            "and",
            "but",
            "for",
            "are",
            "can",
            "you",
            "your",
            "that",
            "this",
            "with",
            "have",
            "will",
            "been",
            "from",
            "they",
            "know",
            "want",
            "use",
            "her",
            "him",
            "his",
            "she",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "too",
        }

        keywords = [word for word in words if word not in stop_words]

        # Return most common keywords
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(10)]

    def _add_pattern(self, new_pattern: MemoryPattern) -> bool:
        """Add a pattern to memory, merging with existing if similar."""
        # Check for similar existing patterns
        for existing_id, existing_pattern in self._patterns.items():
            if (
                existing_pattern.pattern_type == new_pattern.pattern_type
                and self._calculate_similarity(
                    existing_pattern.content, new_pattern.content
                )
                > 0.8
            ):

                # Merge patterns
                updated_pattern = MemoryPattern(
                    pattern_id=existing_id,
                    pattern_type=existing_pattern.pattern_type,
                    content=existing_pattern.content,  # Keep original content
                    frequency=existing_pattern.frequency + 1,
                    confidence=min(0.95, existing_pattern.confidence + 0.05),
                    last_seen=datetime.utcnow(),
                    created_at=existing_pattern.created_at,
                    importance=min(1.0, existing_pattern.importance + 0.1),
                    context_tags=list(
                        set(existing_pattern.context_tags + new_pattern.context_tags)
                    ),
                )
                self._patterns[existing_id] = updated_pattern
                self._update_indexes_for_pattern(updated_pattern)
                return False  # Merged, not new

        # Add new pattern if memory not full
        if len(self._patterns) < self.max_patterns:
            self._patterns[new_pattern.pattern_id] = new_pattern
            self._update_indexes_for_pattern(new_pattern)
            return True

        # Replace least important pattern if memory full
        least_important = min(self._patterns.values(), key=lambda p: p.importance)
        if new_pattern.importance > least_important.importance:
            del self._patterns[least_important.pattern_id]
            self._patterns[new_pattern.pattern_id] = new_pattern
            self._update_indexes_for_pattern(new_pattern)
            return True

        return False

    def _add_knowledge_item(self, new_item: KnowledgeItem) -> bool:
        """Add a knowledge item to memory."""
        # Check for similar existing items
        for existing_id, existing_item in self._knowledge.items():
            if (
                self._calculate_similarity(existing_item.content, new_item.content)
                > 0.8
            ):
                # Update existing item
                updated_item = KnowledgeItem(
                    knowledge_id=existing_id,
                    content=existing_item.content,
                    source_type=existing_item.source_type,
                    source_id=existing_item.source_id,
                    confidence=min(0.95, existing_item.confidence + 0.05),
                    importance=min(1.0, existing_item.importance + 0.1),
                    created_at=existing_item.created_at,
                    last_accessed=datetime.utcnow(),
                    access_count=existing_item.access_count + 1,
                    context_tags=list(
                        set(existing_item.context_tags + new_item.context_tags)
                    ),
                    related_items=existing_item.related_items,
                )
                self._knowledge[existing_id] = updated_item
                return False

        # Add new item if memory not full
        if len(self._knowledge) < self.max_knowledge_items:
            self._knowledge[new_item.knowledge_id] = new_item
            self._update_indexes_for_knowledge(new_item)
            return True

        # Replace least important item if memory full
        least_important = min(self._knowledge.values(), key=lambda k: k.importance)
        if new_item.importance > least_important.importance:
            del self._knowledge[least_important.knowledge_id]
            self._knowledge[new_item.knowledge_id] = new_item
            self._update_indexes_for_knowledge(new_item)
            return True

        return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _update_indexes_for_pattern(self, pattern: MemoryPattern) -> None:
        """Update search indexes for a pattern."""
        for tag in pattern.context_tags:
            self._topic_index[tag.lower()].add(pattern.pattern_id)

        keywords = self._extract_keywords(pattern.content)
        for keyword in keywords:
            self._keyword_index[keyword.lower()].add(pattern.pattern_id)

    def _update_indexes_for_knowledge(self, item: KnowledgeItem) -> None:
        """Update search indexes for a knowledge item."""
        for tag in item.context_tags:
            self._topic_index[tag.lower()].add(item.knowledge_id)

        keywords = self._extract_keywords(item.content)
        for keyword in keywords:
            self._keyword_index[keyword.lower()].add(item.knowledge_id)

    def retrieve_knowledge(
        self,
        context: MemoryContext,
        max_items: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant knowledge based on context.

        Args:
            context: Memory context for retrieval
            max_items: Maximum number of items to return

        Returns:
            List of relevant knowledge items with relevance scores

        Raises:
            AgentMemoryError: If retrieval fails
        """
        try:
            relevant_items: list[tuple[str, float]] = []

            # Search by topic
            if context.topic:
                topic_items = self._topic_index.get(context.topic.lower(), set())
                for item_id in topic_items:
                    if item_id in self._knowledge:
                        score = self._calculate_relevance_score(
                            self._knowledge[item_id], context
                        )
                        if score >= context.importance_threshold:
                            relevant_items.append((item_id, score))

            # Search by keywords
            if context.keywords:
                for keyword in context.keywords:
                    keyword_items = self._keyword_index.get(keyword.lower(), set())
                    for item_id in keyword_items:
                        if item_id in self._knowledge:
                            score = self._calculate_relevance_score(
                                self._knowledge[item_id], context
                            )
                            if score >= context.importance_threshold:
                                relevant_items.append((item_id, score))

            # Remove duplicates and sort by relevance
            unique_items: dict[str, float] = {}
            for item_id, score in relevant_items:
                if item_id not in unique_items or unique_items[item_id] < score:
                    unique_items[item_id] = score

            sorted_items = sorted(
                unique_items.items(), key=lambda x: x[1], reverse=True
            )

            # Prepare results
            results: list[dict[str, Any]] = []
            for item_id, relevance_score in sorted_items[:max_items]:
                item = self._knowledge[item_id]

                # Update access statistics
                updated_item = KnowledgeItem(
                    knowledge_id=item.knowledge_id,
                    content=item.content,
                    source_type=item.source_type,
                    source_id=item.source_id,
                    confidence=item.confidence,
                    importance=item.importance,
                    created_at=item.created_at,
                    last_accessed=datetime.utcnow(),
                    access_count=item.access_count + 1,
                    context_tags=item.context_tags,
                    related_items=item.related_items,
                )
                self._knowledge[item_id] = updated_item

                results.append(
                    {
                        "knowledge_id": item.knowledge_id,
                        "content": item.content,
                        "relevance_score": relevance_score,
                        "confidence": item.confidence,
                        "importance": item.importance,
                        "source_type": item.source_type,
                        "context_tags": item.context_tags,
                    }
                )

            return results

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to retrieve knowledge: {e}",
                agent_id=self.agent_id,
                operation="retrieve",
                original_error=e,
            ) from e

    def _calculate_relevance_score(
        self, item: KnowledgeItem, context: MemoryContext
    ) -> float:
        """Calculate relevance score for a knowledge item given context."""
        score = 0.0

        # Base importance score
        score += item.importance * 0.3

        # Confidence score
        score += item.confidence * 0.2

        # Context matching
        if context.topic:
            topic_match = any(
                context.topic.lower() in tag.lower() for tag in item.context_tags
            )
            if topic_match:
                score += 0.3

        if context.keywords:
            keyword_matches = sum(
                1
                for keyword in context.keywords
                if any(keyword.lower() in tag.lower() for tag in item.context_tags)
            )
            score += (keyword_matches / len(context.keywords)) * 0.2

        # Recency bonus (items accessed recently get slight boost)
        days_since_access = (datetime.utcnow() - item.last_accessed).days
        recency_bonus = max(0, 1 - (days_since_access / 30)) * 0.1
        score += recency_bonus

        # Access frequency bonus
        frequency_bonus = min(0.1, item.access_count / 100)
        score += frequency_bonus

        return min(1.0, score)

    def _consolidate_memory(self) -> dict[str, int]:
        """Consolidate memory by merging similar items and applying decay."""
        consolidated = {"patterns_merged": 0, "knowledge_merged": 0, "items_decayed": 0}

        # Apply memory decay
        current_time = datetime.utcnow()

        # Decay patterns
        patterns_to_remove: list[str] = []
        for pattern_id, pattern in self._patterns.items():
            days_since_seen = (current_time - pattern.last_seen).days
            decay_amount = (days_since_seen / 30) * (1 - self.decay_factor)

            new_importance = pattern.importance * (1 - decay_amount)
            if new_importance < 0.1:
                patterns_to_remove.append(pattern_id)
            else:
                updated_pattern = MemoryPattern(
                    pattern_id=pattern.pattern_id,
                    pattern_type=pattern.pattern_type,
                    content=pattern.content,
                    frequency=pattern.frequency,
                    confidence=pattern.confidence,
                    last_seen=pattern.last_seen,
                    created_at=pattern.created_at,
                    importance=new_importance,
                    context_tags=pattern.context_tags,
                )
                self._patterns[pattern_id] = updated_pattern

        for pattern_id in patterns_to_remove:
            del self._patterns[pattern_id]
            consolidated["items_decayed"] += 1

        # Decay knowledge items
        knowledge_to_remove: list[str] = []
        for item_id, item in self._knowledge.items():
            days_since_access = (current_time - item.last_accessed).days
            decay_amount = (days_since_access / 30) * (1 - self.decay_factor)

            new_importance = item.importance * (1 - decay_amount)
            if new_importance < 0.1:
                knowledge_to_remove.append(item_id)
            else:
                updated_item = KnowledgeItem(
                    knowledge_id=item.knowledge_id,
                    content=item.content,
                    source_type=item.source_type,
                    source_id=item.source_id,
                    confidence=item.confidence,
                    importance=new_importance,
                    created_at=item.created_at,
                    last_accessed=item.last_accessed,
                    access_count=item.access_count,
                    context_tags=item.context_tags,
                    related_items=item.related_items,
                )
                self._knowledge[item_id] = updated_item

        for item_id in knowledge_to_remove:
            del self._knowledge[item_id]
            consolidated["items_decayed"] += 1

        # Rebuild indexes after consolidation
        self._rebuild_indexes()

        return consolidated

    def get_memory_statistics(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_time = datetime.utcnow()

        # Pattern statistics
        pattern_types = Counter(p.pattern_type for p in self._patterns.values())
        avg_pattern_importance = (
            sum(p.importance for p in self._patterns.values()) / len(self._patterns)
            if self._patterns
            else 0
        )

        # Knowledge statistics
        knowledge_sources = Counter(k.source_type for k in self._knowledge.values())
        avg_knowledge_importance = (
            sum(k.importance for k in self._knowledge.values()) / len(self._knowledge)
            if self._knowledge
            else 0
        )

        # Recent activity
        recent_patterns = sum(
            1 for p in self._patterns.values() if (current_time - p.last_seen).days <= 7
        )
        recent_knowledge = sum(
            1
            for k in self._knowledge.values()
            if (current_time - k.last_accessed).days <= 7
        )

        return {
            "agent_id": self.agent_id,
            "total_patterns": len(self._patterns),
            "total_knowledge": len(self._knowledge),
            "pattern_types": dict(pattern_types),
            "knowledge_sources": dict(knowledge_sources),
            "avg_pattern_importance": avg_pattern_importance,
            "avg_knowledge_importance": avg_knowledge_importance,
            "recent_patterns": recent_patterns,
            "recent_knowledge": recent_knowledge,
            "memory_utilization": {
                "patterns": f"{len(self._patterns)}/{self.max_patterns}",
                "knowledge": f"{len(self._knowledge)}/{self.max_knowledge_items}",
            },
            "index_sizes": {
                "topics": len(self._topic_index),
                "keywords": len(self._keyword_index),
            },
        }

    def clear_memory(self, memory_type: str = "all") -> None:
        """
        Clear memory of specified type.

        Args:
            memory_type: Type of memory to clear ('patterns', 'knowledge', 'all')

        Raises:
            AgentMemoryError: If clearing fails
        """
        try:
            if memory_type in ("patterns", "all"):
                self._patterns.clear()

            if memory_type in ("knowledge", "all"):
                self._knowledge.clear()

            if memory_type == "all":
                self._topic_index.clear()
                self._keyword_index.clear()
            else:
                self._rebuild_indexes()

            self._save_memory()
            self.logger.info(f"Cleared {memory_type} memory for agent {self.agent_id}")

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to clear {memory_type} memory: {e}",
                agent_id=self.agent_id,
                operation="clear",
                original_error=e,
            ) from e

    def export_memory(self, output_path: str | Path) -> None:
        """Export memory to a file for backup or analysis."""
        try:
            export_data = {
                "agent_id": self.agent_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "statistics": self.get_memory_statistics(),
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "content": p.content,
                        "frequency": p.frequency,
                        "confidence": p.confidence,
                        "last_seen": p.last_seen.isoformat(),
                        "created_at": p.created_at.isoformat(),
                        "importance": p.importance,
                        "context_tags": p.context_tags,
                    }
                    for p in self._patterns.values()
                ],
                "knowledge": [
                    {
                        "knowledge_id": k.knowledge_id,
                        "content": k.content,
                        "source_type": k.source_type,
                        "source_id": k.source_id,
                        "confidence": k.confidence,
                        "importance": k.importance,
                        "created_at": k.created_at.isoformat(),
                        "last_accessed": k.last_accessed.isoformat(),
                        "access_count": k.access_count,
                        "context_tags": k.context_tags,
                        "related_items": k.related_items,
                    }
                    for k in self._knowledge.values()
                ],
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Memory exported to {output_path}")

        except Exception as e:
            raise AgentMemoryError(
                f"Failed to export memory: {e}",
                agent_id=self.agent_id,
                operation="export",
                original_error=e,
            ) from e


# Export main class
__all__ = ["AgentMemory", "MemoryPattern", "KnowledgeItem", "MemoryContext"]
