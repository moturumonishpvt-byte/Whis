"""AI model registry and configuration management.

Provides authoritative definitions and validation for all local models managed
by WHIS, without initiating runtime processes or loading model weights into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

logger = logging.getLogger(__name__)

# Canonical project root derived relative to this source file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"

# Supported schemas and allowed domain values
SUPPORTED_MODEL_TYPES: Set[str] = {
    "llm",
    "thinking",
    "vlm",
    "ocr",
    "embedding",
}

SUPPORTED_RUNTIMES: Set[str] = {
    "llama.cpp",
}

SUPPORTED_STATUSES: Set[str] = {
    "verified",
    "experimental",
    "deprecated",
    "unverified",
}

SUPPORTED_CAPABILITIES: Set[str] = {
    "chat",
    "text_generation",
    "reasoning",
    "vision",
    "video",
    "code",
    "embedding",
    "ocr",
}


class ModelRegistryError(Exception):
    """Base exception for all model registry operations."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model ID is not found in the registry."""


class ModelConfigError(ModelRegistryError):
    """Raised when the model configuration file is missing, unreadable, or malformed."""


class ModelValidationError(ModelRegistryError):
    """Raised when one or more model definitions fail validation."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors: List[str] = errors or []


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML SafeLoader variant that detects duplicate dictionary keys."""


def _construct_mapping_with_dup_check(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key '{key}'",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_with_dup_check,
)


@dataclass(frozen=True)
class ModelDefinition:
    """Immutable metadata and configuration schema for a local model."""

    id: str
    display_name: str
    type: str
    model_path: str
    capabilities: List[str]
    role: str
    status: str
    runtime: str
    enabled: bool = True
    priority: int = 100
    default_context: int = 2048
    mmproj_path: Optional[str] = None
    size_gb: Optional[float] = None
    description: Optional[str] = None

    @property
    def has_projector(self) -> bool:
        """Return True if an mmproj multimodal projector path is configured."""
        return bool(self.mmproj_path)

    @property
    def requires_projector(self) -> bool:
        """Return True if model architecture expects a multimodal projector."""
        return self.type in {"vlm", "ocr"} or bool(self.mmproj_path)

    @property
    def is_experimental(self) -> bool:
        """Return True if model is marked experimental."""
        return self.status.lower() == "experimental"

    @property
    def is_embedding(self) -> bool:
        """Return True if model is an embedding model."""
        return self.type == "embedding" or "embedding" in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert model definition to dictionary representation."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "type": self.type,
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "capabilities": list(self.capabilities),
            "role": self.role,
            "status": self.status,
            "size_gb": self.size_gb,
            "priority": self.priority,
            "default_context": self.default_context,
            "runtime": self.runtime,
            "enabled": self.enabled,
            "description": self.description,
        }


class ModelRegistry:
    """Authoritative local model registry for WHIS.

    Loads and parses model definitions from config/models.yaml, resolves
    paths relative to the project root, and validates model metadata.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        self.project_root: Path = Path(project_root).resolve() if project_root else PROJECT_ROOT
        self.config_path: Path = Path(config_path).resolve() if config_path else (self.project_root / "config" / "models.yaml")
        self._models: Dict[str, ModelDefinition] = {}

        if auto_load:
            self.load()

    def load(self) -> None:
        """Load and parse model definitions from config YAML.

        Raises:
            ModelConfigError: If file is missing, unreadable, or malformed.
        """
        if not self.config_path.exists():
            raise ModelConfigError(f"Model configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=_UniqueKeySafeLoader)
        except yaml.YAMLError as exc:
            raise ModelConfigError(f"Failed to parse YAML file {self.config_path}: {exc}") from exc
        except Exception as exc:
            raise ModelConfigError(f"Error reading configuration file {self.config_path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ModelConfigError(
                f"Invalid configuration format in {self.config_path}: expected root mapping, got {type(data).__name__}"
            )

        raw_models = data.get("models")
        if raw_models is None:
            raise ModelConfigError(f"Configuration file {self.config_path} missing top-level 'models' key.")

        if not isinstance(raw_models, dict):
            raise ModelConfigError(
                f"'models' key in {self.config_path} must be a mapping of model_id -> definition."
            )

        parsed_models: Dict[str, ModelDefinition] = {}

        for key, entry in raw_models.items():
            if not isinstance(entry, dict):
                raise ModelConfigError(f"Model entry for '{key}' must be a mapping, got {type(entry).__name__}")

            model_id = entry.get("id") or key
            if not model_id or not isinstance(model_id, str):
                raise ModelConfigError(f"Model entry '{key}' has missing or non-string 'id'.")

            model_id = model_id.strip()
            if not model_id:
                raise ModelConfigError(f"Model entry '{key}' has an empty model ID.")

            if model_id in parsed_models:
                raise ModelConfigError(f"Duplicate model ID detected in configuration: '{model_id}'")

            # Validate required string fields
            display_name = entry.get("display_name")
            if not display_name or not isinstance(display_name, str):
                raise ModelConfigError(f"Model '{model_id}' is missing required 'display_name'.")

            model_type = entry.get("type")
            if not model_type or not isinstance(model_type, str):
                raise ModelConfigError(f"Model '{model_id}' is missing required 'type'.")

            model_path = entry.get("model_path")
            if not model_path or not isinstance(model_path, str):
                raise ModelConfigError(f"Model '{model_id}' is missing required 'model_path'.")

            capabilities = entry.get("capabilities")
            if not isinstance(capabilities, list) or not capabilities:
                raise ModelConfigError(f"Model '{model_id}' must provide a non-empty list of 'capabilities'.")

            role = entry.get("role")
            if not role or not isinstance(role, str):
                raise ModelConfigError(f"Model '{model_id}' is missing required 'role'.")

            status = entry.get("status")
            if not status or not isinstance(status, str):
                raise ModelConfigError(f"Model '{model_id}' is missing required 'status'.")

            runtime = entry.get("runtime")
            if not runtime or not isinstance(runtime, str):
                raise ModelConfigError(f"Model '{model_id}' is missing required 'runtime'.")

            enabled = bool(entry.get("enabled", True))
            priority = int(entry.get("priority", 100))
            default_context = int(entry.get("default_context", 2048))
            mmproj_path = entry.get("mmproj_path")
            if mmproj_path is not None and not isinstance(mmproj_path, str):
                raise ModelConfigError(f"Model '{model_id}' 'mmproj_path' must be a string or null.")

            size_gb = entry.get("size_gb")
            if size_gb is not None:
                try:
                    size_gb = float(size_gb)
                except (ValueError, TypeError):
                    raise ModelConfigError(f"Model '{model_id}' 'size_gb' must be numeric.")

            description = entry.get("description")

            definition = ModelDefinition(
                id=model_id,
                display_name=display_name,
                type=model_type,
                model_path=model_path,
                capabilities=capabilities,
                role=role,
                status=status,
                runtime=runtime,
                enabled=enabled,
                priority=priority,
                default_context=default_context,
                mmproj_path=mmproj_path,
                size_gb=size_gb,
                description=description,
            )

            parsed_models[model_id] = definition

        self._models = parsed_models
        logger.debug("Successfully loaded %d models into ModelRegistry.", len(self._models))

    def get(self, model_id: str) -> ModelDefinition:
        """Retrieve a model definition by its ID.

        Raises:
            ModelNotFoundError: If model_id is not registered.
        """
        if model_id not in self._models:
            raise ModelNotFoundError(
                f"Model '{model_id}' is not registered. Available models: {list(self._models.keys())}"
            )
        return self._models[model_id]

    def __getitem__(self, model_id: str) -> ModelDefinition:
        return self.get(model_id)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models

    def list_models(self) -> List[ModelDefinition]:
        """Return all registered models."""
        return list(self._models.values())

    def list_model_ids(self) -> List[str]:
        """Return IDs of all registered models."""
        return list(self._models.keys())

    def enabled_models(self) -> List[ModelDefinition]:
        """Return all registered models that are marked enabled."""
        return [m for m in self._models.values() if m.enabled]

    def is_model_enabled(self, model_id: str) -> bool:
        """Check whether a registered model is enabled."""
        return self.get(model_id).enabled

    def has_projector(self, model_id: str) -> bool:
        """Check whether a registered model has a configured multimodal projector."""
        return self.get(model_id).has_projector

    def requires_projector(self, model_id: str) -> bool:
        """Check whether a registered model requires a multimodal projector."""
        return self.get(model_id).requires_projector

    def resolve_model_path(self, model_id: str) -> Path:
        """Resolve the model weight path relative to the WHIS project root.

        Raises:
            ModelNotFoundError: If model_id is not registered.
        """
        model = self.get(model_id)
        path = Path(model.model_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def resolve_mmproj_path(self, model_id: str) -> Optional[Path]:
        """Resolve the projector path relative to the WHIS project root if configured.

        Raises:
            ModelNotFoundError: If model_id is not registered.
        """
        model = self.get(model_id)
        if not model.mmproj_path:
            return None
        path = Path(model.mmproj_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def validate(
        self,
        check_files_exist: bool = True,
        raise_for_errors: bool = True,
    ) -> List[str]:
        """Validate all loaded model definitions for structural and file consistency.

        Checks:
        - Supported model type
        - Supported runtime
        - Supported status (experimental models remain valid)
        - Known capability values
        - Multimodal projector requirement consistency
        - Existence of model weights file on disk (when check_files_exist is True)
        - Existence of projector file on disk (when check_files_exist is True)

        Args:
            check_files_exist: Whether to verify files on local filesystem.
            raise_for_errors: Whether to raise ModelValidationError if errors are found.

        Returns:
            List of error messages describing all validation failures.

        Raises:
            ModelValidationError: If raise_for_errors is True and errors exist.
        """
        errors: List[str] = []

        if not self._models:
            errors.append("Registry contains no model definitions.")

        for model_id, model in self._models.items():
            # Check model type
            if model.type not in SUPPORTED_MODEL_TYPES:
                errors.append(
                    f"Model '{model_id}': Unsupported type '{model.type}'. "
                    f"Allowed types: {sorted(SUPPORTED_MODEL_TYPES)}"
                )

            # Check runtime
            if model.runtime not in SUPPORTED_RUNTIMES:
                errors.append(
                    f"Model '{model_id}': Unsupported runtime '{model.runtime}'. "
                    f"Allowed runtimes: {sorted(SUPPORTED_RUNTIMES)}"
                )

            # Check status
            if model.status not in SUPPORTED_STATUSES:
                errors.append(
                    f"Model '{model_id}': Unsupported status '{model.status}'. "
                    f"Allowed statuses: {sorted(SUPPORTED_STATUSES)}"
                )

            # Check capabilities
            unknown_caps = set(model.capabilities) - SUPPORTED_CAPABILITIES
            if unknown_caps:
                errors.append(
                    f"Model '{model_id}': Unknown capabilities {sorted(unknown_caps)}. "
                    f"Allowed capabilities: {sorted(SUPPORTED_CAPABILITIES)}"
                )

            # Check projector requirement
            if model.requires_projector and not model.mmproj_path:
                errors.append(
                    f"Model '{model_id}': Model type '{model.type}' requires 'mmproj_path' to be specified."
                )

            # Check file paths on disk
            if check_files_exist:
                resolved_model_path = self.resolve_model_path(model_id)
                if not resolved_model_path.is_file():
                    errors.append(
                        f"Model '{model_id}': Model weight file not found at: {resolved_model_path}"
                    )

                if model.mmproj_path:
                    resolved_mmproj_path = self.resolve_mmproj_path(model_id)
                    if resolved_mmproj_path is None or not resolved_mmproj_path.is_file():
                        errors.append(
                            f"Model '{model_id}': Multimodal projector file not found at: {resolved_mmproj_path}"
                        )

        if errors and raise_for_errors:
            error_details = "\n  - " + "\n  - ".join(errors)
            raise ModelValidationError(
                f"Model registry validation failed with {len(errors)} error(s):{error_details}",
                errors=errors,
            )

        return errors
