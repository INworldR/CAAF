#!/usr/bin/env python3
"""
CAAF Exception Classes - Comprehensive error handling for the CAAF framework.

This module provides a hierarchical structure of exception classes that cover
all possible error scenarios in the CAAF framework. Each exception includes
helpful error messages, optional error codes, and contextual details to aid
in debugging and error handling.

Exception Hierarchy:
    CAAFError
    ├── ModelError
    │   ├── ModelConnectionError
    │   ├── ModelTimeoutError
    │   ├── ModelRateLimitError
    │   └── ModelConfigurationError
    ├── AgentError
    │   ├── AgentCommunicationError
    │   ├── AgentMemoryError
    │   └── AgentInitializationError
    ├── PersistenceError
    │   ├── ConversationSaveError
    │   ├── ConversationLoadError
    │   └── SearchIndexError
    └── ConfigurationError
"""

from __future__ import annotations

from typing import Any


class CAAFError(Exception):
    """
    Base exception class for all CAAF-related errors.

    This is the root of the CAAF exception hierarchy. All other CAAF exceptions
    inherit from this class, allowing for easy catching of any CAAF-related error.

    Attributes:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        details: Optional dictionary with additional error context
        original_error: Optional reference to the original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize a CAAF error.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
            original_error: Optional reference to the original exception
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return a string representation of the error."""
        error_parts = [self.message]

        if self.error_code:
            error_parts.append(f"Error Code: {self.error_code}")

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            error_parts.append(f"Details: {details_str}")

        if self.original_error:
            error_parts.append(f"Caused by: {self.original_error}")

        return " | ".join(error_parts)

    def __repr__(self) -> str:
        """Return a detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r}, "
            f"original_error={self.original_error!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
        }


# Model-related exceptions
class ModelError(CAAFError):
    """
    Base exception for all model-related errors.

    This exception and its subclasses handle errors that occur during
    interaction with AI models (Claude, Ollama, etc.).
    """

    pass


class ModelConnectionError(ModelError):
    """
    Raised when there are connection issues with AI models.

    This includes network connectivity problems, API endpoint failures,
    authentication issues, and service unavailability.

    Examples:
        - Network timeouts
        - API endpoint not responding
        - Invalid API credentials
        - Service temporarily unavailable
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a model connection error.

        Args:
            message: Human-readable error message
            model_name: Name of the model that failed to connect
            endpoint: API endpoint that failed
            status_code: HTTP status code if applicable
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if model_name:
            details["model_name"] = model_name
        if endpoint:
            details["endpoint"] = endpoint
        if status_code:
            details["status_code"] = status_code

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ModelTimeoutError(ModelError):
    """
    Raised when model operations exceed their timeout limits.

    This exception is thrown when requests to AI models take longer
    than the configured timeout period.

    Examples:
        - Long-running inference requests
        - Network latency issues
        - Model overload
    """

    def __init__(
        self,
        message: str,
        timeout_duration: float | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a model timeout error.

        Args:
            message: Human-readable error message
            timeout_duration: The timeout duration that was exceeded
            operation: The operation that timed out
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if timeout_duration:
            details["timeout_duration"] = timeout_duration
        if operation:
            details["operation"] = operation

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ModelRateLimitError(ModelError):
    """
    Raised when API rate limits are exceeded.

    This exception is thrown when the frequency of requests to AI models
    exceeds the allowed rate limits.

    Examples:
        - Too many requests per minute
        - Token usage limits exceeded
        - Concurrent request limits exceeded
    """

    def __init__(
        self,
        message: str,
        limit_type: str | None = None,
        limit_value: int | None = None,
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a model rate limit error.

        Args:
            message: Human-readable error message
            limit_type: Type of limit exceeded (requests, tokens, etc.)
            limit_value: The limit value that was exceeded
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if limit_type:
            details["limit_type"] = limit_type
        if limit_value:
            details["limit_value"] = limit_value
        if retry_after:
            details["retry_after"] = retry_after

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ModelConfigurationError(ModelError):
    """
    Raised when there are configuration issues with AI models.

    This exception is thrown when model configuration is invalid,
    incomplete, or incompatible.

    Examples:
        - Missing API keys
        - Invalid model parameters
        - Unsupported model versions
        - Configuration validation failures
    """

    def __init__(
        self,
        message: str,
        config_field: str | None = None,
        config_value: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a model configuration error.

        Args:
            message: Human-readable error message
            config_field: The configuration field that caused the error
            config_value: The invalid configuration value
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if config_field:
            details["config_field"] = config_field
        if config_value is not None:
            details["config_value"] = str(config_value)

        kwargs["details"] = details
        super().__init__(message, **kwargs)


# Agent-related exceptions
class AgentError(CAAFError):
    """
    Base exception for all agent-related errors.

    This exception and its subclasses handle errors that occur during
    agent operations, communication, and lifecycle management.
    """

    pass


class AgentCommunicationError(AgentError):
    """
    Raised when there are communication issues between agents.

    This includes message passing failures, protocol errors,
    and inter-agent coordination problems.

    Examples:
        - Failed message delivery
        - Protocol version mismatches
        - Message serialization errors
        - Agent discovery failures
    """

    def __init__(
        self,
        message: str,
        sender_agent: str | None = None,
        receiver_agent: str | None = None,
        message_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an agent communication error.

        Args:
            message: Human-readable error message
            sender_agent: ID of the sending agent
            receiver_agent: ID of the receiving agent
            message_type: Type of message that failed
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if sender_agent:
            details["sender_agent"] = sender_agent
        if receiver_agent:
            details["receiver_agent"] = receiver_agent
        if message_type:
            details["message_type"] = message_type

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class AgentMemoryError(AgentError):
    """
    Raised when there are issues with agent memory operations.

    This includes problems with conversation storage, retrieval,
    and memory management operations.

    Examples:
        - Failed to save conversation
        - Memory corruption
        - Storage quota exceeded
        - Search index problems
    """

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        operation: str | None = None,
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an agent memory error.

        Args:
            message: Human-readable error message
            agent_id: ID of the agent experiencing memory issues
            operation: The memory operation that failed
            conversation_id: ID of the conversation involved
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if agent_id:
            details["agent_id"] = agent_id
        if operation:
            details["operation"] = operation
        if conversation_id:
            details["conversation_id"] = conversation_id

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class AgentInitializationError(AgentError):
    """
    Raised when agent initialization fails.

    This exception is thrown when agents cannot be properly initialized
    due to configuration issues, resource problems, or dependency failures.

    Examples:
        - Missing required configuration
        - Resource allocation failures
        - Dependency initialization errors
        - Permission issues
    """

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        agent_type: str | None = None,
        initialization_stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an agent initialization error.

        Args:
            message: Human-readable error message
            agent_id: ID of the agent that failed to initialize
            agent_type: Type of agent that failed to initialize
            initialization_stage: Stage at which initialization failed
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if agent_id:
            details["agent_id"] = agent_id
        if agent_type:
            details["agent_type"] = agent_type
        if initialization_stage:
            details["initialization_stage"] = initialization_stage

        kwargs["details"] = details
        super().__init__(message, **kwargs)


# Persistence-related exceptions
class PersistenceError(CAAFError):
    """
    Base exception for all data persistence errors.

    This exception and its subclasses handle errors that occur during
    data storage, retrieval, and management operations.
    """

    pass


class ConversationSaveError(PersistenceError):
    """
    Raised when conversation data cannot be saved.

    This exception is thrown when there are problems persisting
    conversation data to storage.

    Examples:
        - Disk space issues
        - Permission problems
        - File corruption
        - Serialization errors
    """

    def __init__(
        self,
        message: str,
        conversation_id: str | None = None,
        storage_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a conversation save error.

        Args:
            message: Human-readable error message
            conversation_id: ID of the conversation that failed to save
            storage_path: Path where the save operation failed
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if conversation_id:
            details["conversation_id"] = conversation_id
        if storage_path:
            details["storage_path"] = storage_path

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ConversationLoadError(PersistenceError):
    """
    Raised when conversation data cannot be loaded.

    This exception is thrown when there are problems retrieving
    conversation data from storage.

    Examples:
        - File not found
        - Corrupted data
        - Permission issues
        - Deserialization errors
    """

    def __init__(
        self,
        message: str,
        conversation_id: str | None = None,
        storage_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a conversation load error.

        Args:
            message: Human-readable error message
            conversation_id: ID of the conversation that failed to load
            storage_path: Path where the load operation failed
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if conversation_id:
            details["conversation_id"] = conversation_id
        if storage_path:
            details["storage_path"] = storage_path

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class SearchIndexError(PersistenceError):
    """
    Raised when there are issues with search index operations.

    This exception is thrown when search functionality encounters
    problems with indexing or querying conversation data.

    Examples:
        - Index corruption
        - Search service unavailable
        - Query syntax errors
        - Index rebuild failures
    """

    def __init__(
        self,
        message: str,
        index_name: str | None = None,
        query: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a search index error.

        Args:
            message: Human-readable error message
            index_name: Name of the search index
            query: Search query that failed
            operation: Type of search operation that failed
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if index_name:
            details["index_name"] = index_name
        if query:
            details["query"] = query
        if operation:
            details["operation"] = operation

        kwargs["details"] = details
        super().__init__(message, **kwargs)


# Configuration-related exceptions
class ConfigurationError(CAAFError):
    """
    Raised when there are configuration loading or validation issues.

    This exception is thrown when configuration files cannot be loaded,
    parsed, or validated, or when configuration values are invalid.

    Examples:
        - Missing configuration files
        - Invalid YAML/JSON syntax
        - Schema validation failures
        - Environment variable issues
    """

    def __init__(
        self,
        message: str,
        config_file: str | None = None,
        config_section: str | None = None,
        validation_error: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a configuration error.

        Args:
            message: Human-readable error message
            config_file: Path to the configuration file
            config_section: Configuration section that has issues
            validation_error: Specific validation error message
            **kwargs: Additional arguments passed to CAAFError
        """
        details = kwargs.get("details", {})
        if config_file:
            details["config_file"] = config_file
        if config_section:
            details["config_section"] = config_section
        if validation_error:
            details["validation_error"] = validation_error

        kwargs["details"] = details
        super().__init__(message, **kwargs)


# Convenience functions for error handling
def handle_caaf_error(error: Exception) -> CAAFError:
    """
    Convert a generic exception to a CAAF exception if it isn't already.

    Args:
        error: The exception to convert

    Returns:
        A CAAFError instance
    """
    if isinstance(error, CAAFError):
        return error

    return CAAFError(
        message=str(error),
        error_code="GENERIC_ERROR",
        original_error=error,
    )


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is potentially retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error might be resolved by retrying
    """
    return isinstance(
        error,
        ModelConnectionError
        | ModelTimeoutError
        | ModelRateLimitError
        | ConversationSaveError
        | ConversationLoadError
        | SearchIndexError,
    )


def get_error_severity(error: Exception) -> str:
    """
    Get the severity level of an error.

    Args:
        error: The exception to assess

    Returns:
        Severity level: 'low', 'medium', 'high', or 'critical'
    """
    if isinstance(error, ModelRateLimitError | ModelTimeoutError):
        return "medium"
    elif isinstance(error, ModelConnectionError | AgentCommunicationError):
        return "high"
    elif isinstance(error, AgentInitializationError | ConfigurationError):
        return "critical"
    elif isinstance(error, AgentMemoryError | PersistenceError):
        return "high"
    else:
        return "low"


# Export all exception classes for easy importing
__all__ = [
    # Base exceptions
    "CAAFError",
    # Model exceptions
    "ModelError",
    "ModelConnectionError",
    "ModelTimeoutError",
    "ModelRateLimitError",
    "ModelConfigurationError",
    # Agent exceptions
    "AgentError",
    "AgentCommunicationError",
    "AgentMemoryError",
    "AgentInitializationError",
    # Persistence exceptions
    "PersistenceError",
    "ConversationSaveError",
    "ConversationLoadError",
    "SearchIndexError",
    # Configuration exceptions
    "ConfigurationError",
    # Utility functions
    "handle_caaf_error",
    "is_retryable_error",
    "get_error_severity",
]
