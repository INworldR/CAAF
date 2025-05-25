#!/usr/bin/env python3
"""
Simple Configuration Management Module for CAAF.

This module provides a lightweight, dictionary-based configuration system
for the CAAF framework. It supports YAML configuration files with environment
variable substitution and basic validation.

Features:
- YAML configuration file loading
- Environment variable support with CAAF_ prefix
- Simple dictionary-based storage
- Basic validation for required fields
- Type hints for clarity
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


class Config:
    """
    Simple configuration manager for CAAF.

    This class provides basic configuration loading and management functionality
    using standard Python dictionaries and YAML files.
    """

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self._config: dict[str, Any] = {}
        self._loaded: bool = False
        self.logger = logging.getLogger("caaf.config")

    @staticmethod
    def load_config(config_path: str | Path | None = None) -> Config:
        """
        Load configuration from YAML file with environment variable support.

        Args:
            config_path: Path to configuration file (defaults to config/config.yaml)

        Returns:
            Configured Config instance

        Raises:
            ConfigurationError: If configuration loading fails
        """
        config = Config()

        try:
            # Determine config file path
            if config_path is None:
                config_path = config._find_config_file()

            config_path = Path(config_path)

            if not config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {config_path}",
                    config_file=str(config_path),
                )

            # Load YAML configuration
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

            if not isinstance(raw_config, dict):
                raise ConfigurationError(
                    "Configuration file must contain a YAML dictionary",
                    config_file=str(config_path),
                )

            # Apply environment variable overrides
            config._config = config._apply_env_overrides(raw_config)

            # Validate required fields
            config._validate_required_fields()

            config._loaded = True
            config.logger.info(f"Configuration loaded from {config_path}")

            return config

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                config_file=str(config_path) if config_path else None,
                original_error=e,
            ) from e

    def _find_config_file(self) -> Path:
        """
        Find configuration file in standard locations.

        Returns:
            Path to configuration file

        Raises:
            ConfigurationError: If no configuration file found
        """
        search_paths = [
            Path("config/config.yaml"),
            Path("config/config.yml"),
            Path("config.yaml"),
            Path("config.yml"),
        ]

        # Check environment variable for config path
        env_config_path = os.getenv("CAAF_CONFIG_PATH")
        if env_config_path:
            search_paths.insert(0, Path(env_config_path))

        for path in search_paths:
            if path.exists() and path.is_file():
                return path

        raise ConfigurationError(
            "No configuration file found",
            details={"searched_paths": [str(p) for p in search_paths]},
        )

    def _apply_env_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables with CAAF_ prefix override config values.
        Use dot notation for nested keys: CAAF_models__claude__api_key

        Args:
            config: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        result = config.copy()

        # Get all CAAF_ environment variables
        for env_key, env_value in os.environ.items():
            if not env_key.startswith("CAAF_"):
                continue

            # Convert CAAF_models__claude__api_key to ["models", "claude", "api_key"]
            config_path = env_key[5:].lower().split("__")

            # Apply the override
            self._set_nested_value(result, config_path, env_value)

        return result

    def _set_nested_value(
        self, config: dict[str, Any], path: list[str], value: str
    ) -> None:
        """
        Set a nested configuration value using a path.

        Args:
            config: Configuration dictionary to modify
            path: List of keys representing the path
            value: Value to set
        """
        current = config

        # Navigate to the parent of the target key
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        final_key = path[-1]

        # Try to convert to appropriate type
        converted_value = self._convert_env_value(value)
        current[final_key] = converted_value

    def _convert_env_value(self, value: str) -> str | int | float | bool:
        """
        Convert environment variable string to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value
        """
        # Boolean conversion
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Number conversion
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _validate_required_fields(self) -> None:
        """
        Validate that required configuration fields are present.

        Raises:
            ConfigurationError: If required fields are missing
        """
        required_fields = [
            "models",
            "logging.level",
        ]

        missing_fields = []

        for field in required_fields:
            if not self._has_nested_key(field):
                missing_fields.append(field)

        if missing_fields:
            raise ConfigurationError(
                f"Missing required configuration fields: {', '.join(missing_fields)}",
                details={"missing_fields": missing_fields},
            )

    def _has_nested_key(self, key_path: str) -> bool:
        """
        Check if a nested key exists in the configuration.

        Args:
            key_path: Dot-separated key path (e.g., "models.claude.api_key")

        Returns:
            True if key exists, False otherwise
        """
        keys = key_path.split(".")
        current = self._config

        try:
            for key in keys:
                current = current[key]
            return True
        except (KeyError, TypeError):
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.

        Supports dot notation for nested keys: "models.claude.api_key"

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if not self._loaded:
            raise ConfigurationError(
                "Configuration not loaded. Call load_config() first."
            )

        keys = key.split(".")
        current = self._config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Supports dot notation for nested keys.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        if not self._loaded:
            raise ConfigurationError(
                "Configuration not loaded. Call load_config() first."
            )

        keys = key.split(".")
        self._set_nested_value(self._config, keys, value)

    def get_section(self, section: str) -> dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section as dictionary

        Raises:
            ConfigurationError: If section not found
        """
        section_data = self.get(section)
        if section_data is None:
            raise ConfigurationError(
                f"Configuration section '{section}' not found", config_section=section
            )

        if not isinstance(section_data, dict):
            raise ConfigurationError(
                f"Configuration section '{section}' is not a dictionary",
                config_section=section,
            )

        return section_data

    def has_key(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key: Configuration key (supports dot notation)

        Returns:
            True if key exists, False otherwise
        """
        if not self._loaded:
            return False

        return self._has_nested_key(key)

    def get_model_config(
        self, provider: str, model: str | None = None
    ) -> dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            provider: Provider name (e.g., 'claude', 'ollama')
            model: Model name (if None, uses default for provider)

        Returns:
            Model configuration dictionary

        Raises:
            ConfigurationError: If provider or model not found
        """
        providers = self.get("models", {})

        if provider not in providers:
            raise ConfigurationError(
                f"Provider '{provider}' not found in configuration",
                config_section="models",
                details={"available_providers": list(providers.keys())},
            )

        provider_config = providers[provider]

        # Get model name
        if model is None:
            model = provider_config.get("default_model")
            if not model:
                raise ConfigurationError(
                    f"No default model specified for provider '{provider}'",
                    config_section=f"models.{provider}",
                )

        # Get model configuration
        models = provider_config.get("models", {})
        if model not in models:
            raise ConfigurationError(
                f"Model '{model}' not found in provider '{provider}'",
                config_section=f"models.{provider}.models",
                details={"available_models": list(models.keys())},
            )

        model_config = models[model]
        if not isinstance(model_config, dict):
            raise ConfigurationError(
                f"Model configuration for '{model}' is not a dictionary",
                config_section=f"models.{provider}.models.{model}",
            )
        return model_config

    def to_dict(self) -> dict[str, Any]:
        """
        Get the entire configuration as a dictionary.

        Returns:
            Complete configuration dictionary

        Raises:
            ConfigurationError: If configuration not loaded
        """
        if not self._loaded:
            raise ConfigurationError(
                "Configuration not loaded. Call load_config() first."
            )

        return self._config.copy()

    def export_config(
        self, output_path: str | Path, include_secrets: bool = False
    ) -> None:
        """
        Export configuration to a YAML file.

        Args:
            output_path: Path to export file
            include_secrets: Whether to include sensitive data

        Raises:
            ConfigurationError: If export fails
        """
        try:
            if not self._loaded:
                raise ConfigurationError(
                    "Configuration not loaded. Call load_config() first."
                )

            config_data = self._config.copy()

            if not include_secrets:
                self._remove_secrets(config_data)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Configuration exported to {output_path}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to export configuration: {e}",
                config_file=str(output_path),
                original_error=e,
            ) from e

    def _remove_secrets(self, config_dict: dict[str, Any]) -> None:
        """
        Remove sensitive information from configuration dictionary.

        Args:
            config_dict: Configuration dictionary to modify
        """

        def remove_sensitive_keys(obj: Any) -> None:
            if isinstance(obj, dict):
                sensitive_keys = ["api_key", "password", "secret", "token", "key"]
                for key in list(obj.keys()):
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        obj[key] = "***REDACTED***"
                    else:
                        remove_sensitive_keys(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    remove_sensitive_keys(item)

        remove_sensitive_keys(config_dict)

    def __repr__(self) -> str:
        """String representation of the configuration."""
        status = "loaded" if self._loaded else "not loaded"
        return f"Config(status={status}, keys={len(self._config)})"


# Global configuration instance
_global_config: Config | None = None


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load the global CAAF configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration instance
    """
    global _global_config
    _global_config = Config.load_config(config_path)
    return _global_config


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Global configuration instance

    Raises:
        ConfigurationError: If configuration not loaded
    """
    global _global_config
    if _global_config is None:
        raise ConfigurationError(
            "Global configuration not loaded. Call load_config() first."
        )
    return _global_config


def get_model_config(provider: str, model: str | None = None) -> dict[str, Any]:
    """
    Get configuration for a specific model from global config.

    Args:
        provider: Provider name
        model: Model name (optional)

    Returns:
        Model configuration dictionary
    """
    return get_config().get_model_config(provider, model)


# Export main classes and functions
__all__ = [
    "Config",
    "load_config",
    "get_config",
    "get_model_config",
]
