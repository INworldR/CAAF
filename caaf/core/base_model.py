#!/usr/bin/env python3
"""
Base Model Module - Core abstraction for AI model integrations.

This module provides the foundational BaseModel abstract class that defines
the interface and common functionality for all AI model integrations in CAAF.
Supports both Claude API and Ollama with async operations, rate limiting,
and robust error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import UUID, uuid4

import yaml

if TYPE_CHECKING:
    pass
from asyncio_throttle.throttler import Throttler
from pydantic import BaseModel, Field

# Type variables for generic response handling
T = TypeVar("T")
ResponseType = TypeVar("ResponseType")


# Custom exception classes for model-related errors
class ModelError(Exception):
    """Base exception for all model-related errors."""

    pass


class ModelInitializationError(ModelError):
    """Raised when model initialization fails."""

    pass


class ModelConnectionError(ModelError):
    """Raised when model connection fails."""

    pass


class ModelTimeoutError(ModelError):
    """Raised when model operations timeout."""

    pass


class ModelRateLimitError(ModelError):
    """Raised when rate limits are exceeded."""

    pass


class ModelConfigurationError(ModelError):
    """Raised when model configuration is invalid."""

    pass


class ModelResponseError(ModelError):
    """Raised when model response is invalid or empty."""

    pass


# Data structures for model operations
class ModelRequest(BaseModel, Generic[T]):
    """Represents a request to an AI model."""

    id: UUID = Field(default_factory=uuid4)
    messages: list[dict[str, str]] = Field(..., description="Messages to send to model")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Model parameters"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Request metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class ModelResponse(BaseModel, Generic[T]):
    """Represents a response from an AI model."""

    id: UUID = Field(default_factory=uuid4)
    request_id: UUID = Field(..., description="ID of the corresponding request")
    content: str = Field(..., description="Generated response content")
    model_info: dict[str, Any] = Field(
        default_factory=dict, description="Model metadata"
    )
    usage_stats: dict[str, Any] = Field(
        default_factory=dict, description="Usage statistics"
    )
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class ModelInfo(BaseModel):
    """Information about a model's capabilities and configuration."""

    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider (claude, ollama, etc.)")
    version: str | None = Field(None, description="Model version")
    max_tokens: int | None = Field(None, description="Maximum tokens supported")
    supports_streaming: bool = Field(default=False, description="Streaming support")
    context_window: int | None = Field(None, description="Context window size")
    capabilities: list[str] = Field(
        default_factory=list, description="Model capabilities"
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Model config"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class ModelConnectionStatus(BaseModel):
    """Status of model connection and health."""

    is_connected: bool = Field(..., description="Connection status")
    is_healthy: bool = Field(..., description="Health status")
    response_time: float | None = Field(None, description="Last response time")
    last_check: datetime = Field(default_factory=datetime.utcnow)
    error_message: str | None = Field(None, description="Error message if any")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class AIBaseModel(ABC):
    """
    Abstract base class for all AI model integrations in CAAF.

    This class provides the core interface and common functionality that all
    model implementations must provide. It handles rate limiting, error recovery,
    configuration management, and connection lifecycle.

    Attributes:
        model_name: Name of the model
        provider: Provider name (claude, ollama, etc.)
        config: Model configuration
        logger: Logger instance for model activities
        throttler: Rate limiter for API calls
        connection_status: Current connection status
        retry_config: Configuration for retry logic
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the base model.

        Args:
            model_name: Name of the model to use
            provider: Provider name (claude, ollama, etc.)
            config: Optional configuration dictionary
            config_path: Optional path to configuration file

        Raises:
            ModelInitializationError: If model initialization fails
            ModelConfigurationError: If configuration is invalid
        """
        try:
            self.model_name = model_name
            self.provider = provider
            self.logger = logging.getLogger(f"caaf.model.{provider}.{model_name}")

            # Load configuration
            self.config = self._load_configuration(config, config_path)
            self._validate_configuration()

            # Initialize rate limiting
            self.throttler = self._initialize_throttler()

            # Initialize retry configuration
            self.retry_config = self.config.get("retry", {})
            self.max_retries = self.retry_config.get("max_retries", 3)
            self.base_delay = self.retry_config.get("base_delay", 1.0)
            self.max_delay = self.retry_config.get("max_delay", 60.0)

            # Connection status tracking
            self.connection_status = ModelConnectionStatus(
                is_connected=False,
                is_healthy=False,
                response_time=None,
                error_message=None,
            )

            # Initialize model-specific components
            self._initialize_model()

            self.logger.info(f"Initialized {provider} model: {model_name}")

        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize {provider} model {model_name}: {e}"
            ) from e

    def _load_configuration(
        self, config: dict[str, Any] | None, config_path: str | Path | None
    ) -> dict[str, Any]:
        """
        Load configuration from dictionary or file.

        Args:
            config: Configuration dictionary
            config_path: Path to configuration file

        Returns:
            Merged configuration dictionary
        """
        # Start with default configuration
        default_config = {
            "timeout": 30.0,
            "rate_limit": {"calls_per_second": 1.0, "burst": 5},
            "retry": {"max_retries": 3, "base_delay": 1.0, "max_delay": 60.0},
        }

        # Load from file if provided
        file_config: dict[str, Any] = {}
        if config_path:
            try:
                path = Path(config_path)
                if path.exists():
                    with open(path) as f:
                        file_config = yaml.safe_load(f) or {}
                    self.logger.debug(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")

        # Merge configurations (config parameter takes precedence)
        merged_config = {**default_config, **file_config, **(config or {})}

        return merged_config

    def _validate_configuration(self) -> None:
        """
        Validate the model configuration.

        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        required_fields = ["timeout"]
        for field in required_fields:
            if field not in self.config:
                raise ModelConfigurationError(f"Missing required config field: {field}")

        # Validate timeout
        if (
            not isinstance(self.config["timeout"], int | float)
            or self.config["timeout"] <= 0
        ):
            raise ModelConfigurationError("Timeout must be a positive number")

        # Validate rate limiting configuration
        rate_config = self.config.get("rate_limit", {})
        if "calls_per_second" in rate_config:
            if (
                not isinstance(rate_config["calls_per_second"], int | float)
                or rate_config["calls_per_second"] <= 0
            ):
                raise ModelConfigurationError(
                    "calls_per_second must be a positive number"
                )

    def _initialize_throttler(self) -> Throttler:
        """
        Initialize the rate limiter based on configuration.

        Returns:
            Configured throttler instance
        """
        rate_config = self.config.get("rate_limit", {})
        calls_per_second = rate_config.get("calls_per_second", 1.0)
        burst = rate_config.get("burst", 5)

        throttler = Throttler(rate_limit=calls_per_second)
        self.logger.debug(
            f"Initialized throttler: {calls_per_second} calls/sec, burst: {burst}"
        )

        return throttler

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize model-specific components. Override in subclasses."""
        pass

    @abstractmethod
    def _validate_model_configuration(self) -> None:
        """Validate model-specific configuration. Override in subclasses."""
        pass

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the model with rate limiting and error recovery.

        Args:
            messages: List of messages to send to the model
            **kwargs: Additional parameters for response generation

        Returns:
            Generated response from the model

        Raises:
            ModelConnectionError: If model connection fails
            ModelResponseError: If response generation fails
            ModelTimeoutError: If operation times out
        """
        request: ModelRequest[Any] = ModelRequest(messages=messages, parameters=kwargs)

        try:
            self.logger.debug(f"Generating response for request {request.id}")

            # Apply rate limiting
            async with self.throttler:
                # Generate response with retry logic
                response = await self._generate_response_with_retry(request)

            self.logger.info(f"Generated response in {response.response_time:.2f}s")
            return response.content

        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            if isinstance(e, ModelError):
                raise
            raise ModelResponseError(f"Response generation failed: {e}") from e

    async def _generate_response_with_retry(
        self, request: ModelRequest[Any]
    ) -> ModelResponse[Any]:
        """
        Generate response with exponential backoff retry logic.

        Args:
            request: The model request to process

        Returns:
            Model response

        Raises:
            ModelResponseError: If all retry attempts fail
        """
        last_error = None
        delay = self.base_delay

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                # Call model-specific implementation
                content = await self._call_model_api(
                    request.messages, **request.parameters
                )

                response_time = time.time() - start_time

                if not content or not content.strip():
                    raise ModelResponseError("Model returned empty response")

                # Create successful response
                response: ModelResponse[Any] = ModelResponse(
                    request_id=request.id,
                    content=content.strip(),
                    response_time=response_time,
                )

                # Update connection status on success
                self.connection_status.is_connected = True
                self.connection_status.is_healthy = True
                self.connection_status.response_time = response_time
                self.connection_status.last_check = datetime.utcnow()
                self.connection_status.error_message = None

                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")

                # Update connection status on error
                self.connection_status.is_connected = False
                self.connection_status.is_healthy = False
                self.connection_status.last_check = datetime.utcnow()
                self.connection_status.error_message = str(e)

                if attempt < self.max_retries:
                    # Wait before retry with exponential backoff
                    actual_delay = min(delay, self.max_delay)
                    self.logger.info(f"Retrying in {actual_delay:.2f}s...")
                    await asyncio.sleep(actual_delay)
                    delay *= 2  # Exponential backoff
                else:
                    self.logger.error("All retry attempts failed")

        raise ModelResponseError(
            f"Failed to generate response after {self.max_retries + 1} attempts"
        ) from last_error

    @abstractmethod
    async def _call_model_api(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        """
        Call the model-specific API. Must be implemented by subclasses.

        Args:
            messages: Messages to send to the model
            **kwargs: Additional parameters

        Returns:
            Raw response content from the model
        """
        pass

    async def validate_connection(self) -> ModelConnectionStatus:
        """
        Validate the connection to the model and update status.

        Returns:
            Current connection status

        Raises:
            ModelConnectionError: If connection validation fails
        """
        try:
            self.logger.debug("Validating model connection")

            # Perform model-specific connection validation
            start_time = time.time()
            is_connected = await self._check_model_connection()
            response_time = time.time() - start_time

            # Update connection status
            self.connection_status.is_connected = is_connected
            self.connection_status.is_healthy = is_connected
            self.connection_status.response_time = response_time
            self.connection_status.last_check = datetime.utcnow()
            self.connection_status.error_message = None

            self.logger.info(
                f"Connection validation: {'OK' if is_connected else 'FAILED'}"
            )

        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            self.connection_status.is_connected = False
            self.connection_status.is_healthy = False
            self.connection_status.last_check = datetime.utcnow()
            self.connection_status.error_message = str(e)

            raise ModelConnectionError(f"Connection validation failed: {e}") from e

        return self.connection_status

    @abstractmethod
    async def _check_model_connection(self) -> bool:
        """
        Check model-specific connection. Must be implemented by subclasses.

        Returns:
            True if connection is healthy, False otherwise
        """
        pass

    async def get_model_info(self) -> ModelInfo:
        """
        Get information about the model's capabilities and configuration.

        Returns:
            Model information

        Raises:
            ModelConnectionError: If unable to retrieve model info
        """
        try:
            self.logger.debug("Retrieving model information")

            # Get model-specific information
            info_data = await self._fetch_model_info()

            model_info = ModelInfo(
                name=self.model_name,
                provider=self.provider,
                **info_data,
            )

            self.logger.debug(f"Retrieved model info: {model_info.name}")
            return model_info

        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            raise ModelConnectionError(f"Failed to get model info: {e}") from e

    @abstractmethod
    async def _fetch_model_info(self) -> dict[str, Any]:
        """
        Fetch model-specific information. Must be implemented by subclasses.

        Returns:
            Dictionary with model information
        """
        pass

    def is_available(self) -> bool:
        """
        Check if the model is currently available.

        Returns:
            True if model is available, False otherwise
        """
        return self.connection_status.is_connected and self.connection_status.is_healthy

    def get_status(self) -> dict[str, Any]:
        """
        Get current model status and statistics.

        Returns:
            Dictionary with model status information
        """
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "is_available": self.is_available(),
            "connection_status": self.connection_status.model_dump(),
            "config": {
                "timeout": self.config.get("timeout"),
                "rate_limit": self.config.get("rate_limit"),
                "retry": self.retry_config,
            },
        }

    @asynccontextmanager
    async def connection_context(self) -> AsyncGenerator[AIBaseModel, None]:
        """
        Context manager for model connections with automatic cleanup.

        Yields:
            The model instance
        """
        try:
            self.logger.debug("Entering model connection context")
            await self.validate_connection()
            yield self
        except Exception as e:
            self.logger.error(f"Error in connection context: {e}")
            raise
        finally:
            self.logger.debug("Exiting model connection context")
            await self._cleanup_connection()

    @abstractmethod
    async def _cleanup_connection(self) -> None:
        """
        Clean up model connections and resources.
        Override in subclasses if needed.
        """
        pass

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the model and clean up resources.
        """
        try:
            self.logger.info(f"Shutting down {self.provider} model: {self.model_name}")
            await self._cleanup_connection()
            self.logger.info("Model shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during model shutdown: {e}")

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name={self.model_name}, provider={self.provider})"
