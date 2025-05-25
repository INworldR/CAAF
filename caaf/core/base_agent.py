#!/usr/bin/env python3
"""
Base Agent Module - Core abstraction for all CAAF agents.

This module provides the foundational BaseAgent abstract class that defines
the interface and common functionality for all agents in the CAAF framework.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# Custom exception classes for agent-related errors
class AgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails."""

    pass


class AgentCommunicationError(AgentError):
    """Raised when agent communication fails."""

    pass


class AgentMemoryError(AgentError):
    """Raised when agent memory operations fail."""

    pass


# Message and conversation data structures
class Message(BaseModel):
    """Represents a single message in a conversation."""

    id: UUID = Field(default_factory=uuid4)
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class ConversationHistory(BaseModel):
    """Represents a conversation history with metadata."""

    id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(..., description="ID of the agent owning this conversation")
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a new message to the conversation."""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get the most recent messages from the conversation."""
        return self.messages[-limit:] if self.messages else []

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class ModelInterface(Protocol):
    """Protocol defining the interface for AI models."""

    async def generate_response(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        """Generate a response from the model."""
        ...

    def is_available(self) -> bool:
        """Check if the model is available."""
        ...


class MemoryInterface(Protocol):
    """Protocol defining the interface for agent memory."""

    async def store_conversation(self, conversation: ConversationHistory) -> None:
        """Store a conversation in memory."""
        ...

    async def retrieve_conversations(
        self, agent_id: str, limit: Optional[int] = None
    ) -> List[ConversationHistory]:
        """Retrieve conversations for an agent."""
        ...

    async def search_conversations(
        self, agent_id: str, query: str
    ) -> List[ConversationHistory]:
        """Search conversations by content."""
        ...


class BaseAgent(ABC):
    """
    Abstract base class for all CAAF agents.

    This class provides the core interface and common functionality that all
    agents must implement. It handles conversation management, memory integration,
    and basic communication patterns.

    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name for the agent
        model: AI model interface for generating responses
        memory: Memory interface for conversation persistence
        logger: Logger instance for agent activities
        current_conversation: Current active conversation
        config: Agent configuration parameters
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        model: ModelInterface,
        memory: Optional[MemoryInterface] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            model: AI model interface for generating responses
            memory: Optional memory interface for conversation persistence
            config: Optional configuration parameters

        Raises:
            AgentInitializationError: If agent initialization fails
        """
        try:
            self.agent_id = agent_id
            self.name = name
            self.model = model
            self.memory = memory
            self.config = config or {}
            self.logger = logging.getLogger(f"caaf.agent.{agent_id}")
            self.current_conversation: Optional[ConversationHistory] = None
            self._initialize_agent()

        except Exception as e:
            raise AgentInitializationError(
                f"Failed to initialize agent {agent_id}: {e}"
            ) from e

    def _initialize_agent(self) -> None:
        """Initialize agent-specific components. Override in subclasses."""
        pass

    async def chat(
        self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """
        Process a chat message and generate a response.

        This method handles the complete conversation flow including:
        - Message validation and preprocessing
        - Context integration
        - Model interaction with error handling
        - Response post-processing
        - Memory storage

        Args:
            message: The input message from the user
            context: Optional context information for the conversation
            **kwargs: Additional parameters for response generation

        Returns:
            Generated response from the agent

        Raises:
            AgentCommunicationError: If communication with the model fails
            AgentMemoryError: If memory operations fail
        """
        try:
            # Validate input
            if not message or not message.strip():
                raise ValueError("Message cannot be empty")

            # Initialize conversation if needed
            if self.current_conversation is None:
                self.current_conversation = ConversationHistory(
                    agent_id=self.agent_id, metadata=context or {}
                )

            # Add user message to conversation
            user_message = self.current_conversation.add_message(
                role="user", content=message.strip(), metadata=context or {}
            )

            self.logger.info(f"Processing message from user: {message[:100]}...")

            # Prepare messages for model
            model_messages = self._prepare_model_messages()

            # Generate response with error handling and retries
            response = await self._generate_response_with_retry(
                model_messages, **kwargs
            )

            # Add assistant response to conversation
            assistant_message = self.current_conversation.add_message(
                role="assistant",
                content=response,
                metadata={"generation_params": kwargs},
            )

            # Store conversation in memory
            await self.add_memory(self.current_conversation)

            self.logger.info(f"Generated response: {response[:100]}...")

            return response

        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            if isinstance(e, (AgentCommunicationError, AgentMemoryError)):
                raise
            raise AgentCommunicationError(f"Failed to process chat message: {e}") from e

    async def _generate_response_with_retry(
        self, messages: List[Dict[str, str]], max_retries: int = 3, **kwargs: Any
    ) -> str:
        """
        Generate response with retry logic for robustness.

        Args:
            messages: Messages to send to the model
            max_retries: Maximum number of retry attempts
            **kwargs: Additional generation parameters

        Returns:
            Generated response from the model

        Raises:
            AgentCommunicationError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if not self.model.is_available():
                    raise AgentCommunicationError("Model is not available")

                response = await self.model.generate_response(messages, **kwargs)

                if not response or not response.strip():
                    raise AgentCommunicationError("Model returned empty response")

                return response.strip()

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All retry attempts failed: {e}")

        raise AgentCommunicationError(
            f"Failed to generate response after {max_retries + 1} attempts"
        ) from last_error

    def _prepare_model_messages(self) -> List[Dict[str, str]]:
        """
        Prepare messages for the model interface.

        Returns:
            List of messages formatted for the model
        """
        if not self.current_conversation or not self.current_conversation.messages:
            return []

        # Convert messages to model format
        model_messages = []
        for msg in self.current_conversation.messages:
            model_messages.append({"role": msg.role, "content": msg.content})

        return model_messages

    async def add_memory(
        self,
        conversation: ConversationHistory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a conversation in the agent's memory.

        Args:
            conversation: The conversation to store
            metadata: Optional metadata to associate with the storage

        Raises:
            AgentMemoryError: If memory storage fails
        """
        if self.memory is None:
            self.logger.warning("No memory interface configured, skipping storage")
            return

        try:
            # Add metadata if provided
            if metadata:
                conversation.metadata.update(metadata)

            await self.memory.store_conversation(conversation)
            self.logger.debug(f"Stored conversation {conversation.id} in memory")

        except Exception as e:
            self.logger.error(f"Failed to store conversation in memory: {e}")
            raise AgentMemoryError(f"Memory storage failed: {e}") from e

    async def get_conversation_history(
        self, limit: Optional[int] = None, search_query: Optional[str] = None
    ) -> List[ConversationHistory]:
        """
        Retrieve conversation history from memory.

        Args:
            limit: Maximum number of conversations to retrieve
            search_query: Optional search query to filter conversations

        Returns:
            List of conversation histories

        Raises:
            AgentMemoryError: If memory retrieval fails
        """
        if self.memory is None:
            self.logger.warning(
                "No memory interface configured, returning empty history"
            )
            return []

        try:
            if search_query:
                conversations = await self.memory.search_conversations(
                    self.agent_id, search_query
                )
            else:
                conversations = await self.memory.retrieve_conversations(
                    self.agent_id, limit
                )

            self.logger.debug(
                f"Retrieved {len(conversations)} conversations from memory"
            )
            return conversations

        except Exception as e:
            self.logger.error(f"Failed to retrieve conversation history: {e}")
            raise AgentMemoryError(f"Memory retrieval failed: {e}") from e

    def get_current_conversation(self) -> Optional[ConversationHistory]:
        """
        Get the current active conversation.

        Returns:
            Current conversation or None if no active conversation
        """
        return self.current_conversation

    def start_new_conversation(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationHistory:
        """
        Start a new conversation, storing the current one if it exists.

        Args:
            metadata: Optional metadata for the new conversation

        Returns:
            The new conversation instance
        """
        # Store current conversation if it exists and has messages
        if (
            self.current_conversation
            and self.current_conversation.messages
            and self.memory
        ):
            try:
                asyncio.create_task(self.add_memory(self.current_conversation))
            except Exception as e:
                self.logger.error(f"Failed to store previous conversation: {e}")

        # Create new conversation
        self.current_conversation = ConversationHistory(
            agent_id=self.agent_id, metadata=metadata or {}
        )

        self.logger.info(f"Started new conversation {self.current_conversation.id}")
        return self.current_conversation

    @abstractmethod
    async def process_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a message with agent-specific logic.

        This method must be implemented by subclasses to define
        agent-specific message processing behavior.

        Args:
            message: The input message to process
            context: Optional context for processing

        Returns:
            Processed response
        """
        pass

    @abstractmethod
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities and configuration of this agent.

        Returns:
            Dictionary describing agent capabilities
        """
        pass

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get basic information about this agent.

        Returns:
            Dictionary with agent information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.__class__.__name__,
            "model_available": self.model.is_available() if self.model else False,
            "memory_enabled": self.memory is not None,
            "current_conversation_id": (
                str(self.current_conversation.id) if self.current_conversation else None
            ),
            "config": self.config,
            "capabilities": self.get_agent_capabilities(),
        }

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, name={self.name})"
