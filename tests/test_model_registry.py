"""Tests for WHIS Stage 3.1: Model Registry."""

import tempfile
import unittest
from pathlib import Path

import yaml

from app.ai.model_manager import (
    ModelConfigError,
    ModelDefinition,
    ModelNotFoundError,
    ModelRegistry,
    ModelValidationError,
    PROJECT_ROOT,
)


class TestModelRegistry(unittest.TestCase):
    """Test suite for ModelRegistry and ModelDefinition."""

    def setUp(self) -> None:
        self.registry = ModelRegistry()

    def test_01_yaml_loads_successfully(self) -> None:
        """1. YAML loads successfully and populates the registry."""
        self.assertIsNotNone(self.registry)
        self.assertGreater(len(self.registry.list_models()), 0)

    def test_02_all_seven_models_are_registered(self) -> None:
        """2. Exactly all 7 local models are registered."""
        models = self.registry.list_models()
        self.assertEqual(len(models), 7, f"Expected 7 models, got {len(models)}")

    def test_03_expected_model_ids_exist(self) -> None:
        """3. Each expected model ID exists in the registry."""
        expected_ids = {
            "qwen3.5-4b",
            "lfm2.5-thinking-1.2b",
            "qwen3-vl-4b",
            "qwen3-coder-30b",
            "qwen3-embedding-4b",
            "glm-ocr",
            "deepseek-ocr",
        }
        registered_ids = set(self.registry.list_model_ids())
        self.assertEqual(registered_ids, expected_ids)
        for model_id in expected_ids:
            self.assertIn(model_id, self.registry)
            model = self.registry.get(model_id)
            self.assertIsInstance(model, ModelDefinition)
            self.assertEqual(model.id, model_id)

    def test_04_model_paths_resolve_correctly(self) -> None:
        """4. Model paths resolve to actual existing files on disk."""
        for model in self.registry.list_models():
            resolved = self.registry.resolve_model_path(model.id)
            self.assertTrue(resolved.is_absolute(), f"Path must be absolute: {resolved}")
            self.assertTrue(
                resolved.is_file(),
                f"Model file for '{model.id}' does not exist on disk at: {resolved}",
            )
            # Ensure path is under the project root
            self.assertTrue(
                str(resolved).startswith(str(PROJECT_ROOT)),
                f"Resolved path {resolved} is not under project root {PROJECT_ROOT}",
            )

    def test_05_qwen3_vl_has_mmproj_path(self) -> None:
        """5. Qwen3-VL has a valid mmproj projector path."""
        model = self.registry.get("qwen3-vl-4b")
        self.assertTrue(model.has_projector)
        self.assertIsNotNone(model.mmproj_path)
        resolved_mmproj = self.registry.resolve_mmproj_path("qwen3-vl-4b")
        self.assertIsNotNone(resolved_mmproj)
        self.assertTrue(
            resolved_mmproj.is_file(),
            f"Qwen3-VL projector file missing at: {resolved_mmproj}",
        )

    def test_06_glm_ocr_has_mmproj_path(self) -> None:
        """6. GLM-OCR has a valid mmproj projector path."""
        model = self.registry.get("glm-ocr")
        self.assertTrue(model.has_projector)
        self.assertIsNotNone(model.mmproj_path)
        resolved_mmproj = self.registry.resolve_mmproj_path("glm-ocr")
        self.assertIsNotNone(resolved_mmproj)
        self.assertTrue(
            resolved_mmproj.is_file(),
            f"GLM-OCR projector file missing at: {resolved_mmproj}",
        )

    def test_07_deepseek_ocr_has_mmproj_path(self) -> None:
        """7. DeepSeek-OCR has a valid mmproj projector path."""
        model = self.registry.get("deepseek-ocr")
        self.assertTrue(model.has_projector)
        self.assertIsNotNone(model.mmproj_path)
        resolved_mmproj = self.registry.resolve_mmproj_path("deepseek-ocr")
        self.assertIsNotNone(resolved_mmproj)
        self.assertTrue(
            resolved_mmproj.is_file(),
            f"DeepSeek-OCR projector file missing at: {resolved_mmproj}",
        )

    def test_08_qwen3_embedding_is_embedding_model(self) -> None:
        """8. Qwen3-Embedding is registered as an embedding model."""
        model = self.registry.get("qwen3-embedding-4b")
        self.assertEqual(model.type, "embedding")
        self.assertIn("embedding", model.capabilities)
        self.assertTrue(model.is_embedding)
        self.assertNotIn("chat", model.capabilities)

    def test_09_deepseek_ocr_is_marked_experimental(self) -> None:
        """9. DeepSeek-OCR is clearly marked as experimental."""
        model = self.registry.get("deepseek-ocr")
        self.assertEqual(model.status, "experimental")
        self.assertTrue(model.is_experimental)

    def test_10_no_absolute_user_specific_paths_in_config(self) -> None:
        """10. No absolute or user-specific paths exist in models.yaml."""
        config_path = PROJECT_ROOT / "config" / "models.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Reject common absolute path markers
        self.assertNotIn("C:\\Users\\", raw_text)
        self.assertNotIn("c:\\users\\", raw_text.lower())
        self.assertNotIn("/home/", raw_text)
        self.assertNotIn("OneDrive", raw_text)

        # Ensure all parsed paths are relative
        for model in self.registry.list_models():
            self.assertFalse(
                Path(model.model_path).is_absolute(),
                f"Model path for '{model.id}' must be relative, got: {model.model_path}",
            )
            if model.mmproj_path:
                self.assertFalse(
                    Path(model.mmproj_path).is_absolute(),
                    f"mmproj path for '{model.id}' must be relative, got: {model.mmproj_path}",
                )

    def test_11_missing_file_validation_works(self) -> None:
        """11. Missing-file validation detects missing weights or projectors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_config = tmp_path / "models.yaml"
            fake_config.write_text(
                yaml.dump({
                    "models": {
                        "missing-model": {
                            "id": "missing-model",
                            "display_name": "Nonexistent Model",
                            "type": "llm",
                            "model_path": "models/nonexistent/model.gguf",
                            "capabilities": ["chat"],
                            "role": "test",
                            "status": "verified",
                            "runtime": "llama.cpp",
                            "enabled": True,
                        }
                    }
                }),
                encoding="utf-8",
            )
            reg = ModelRegistry(config_path=fake_config, project_root=tmp_path)
            errors = reg.validate(check_files_exist=True, raise_for_errors=False)
            self.assertTrue(any("Model weight file not found" in err for err in errors))
            with self.assertRaises(ModelValidationError):
                reg.validate(check_files_exist=True, raise_for_errors=True)

    def test_12_registry_lookup_invalid_id_fails_cleanly(self) -> None:
        """12. Registry lookup for an invalid model ID raises ModelNotFoundError."""
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.registry.get("nonexistent-model-id")
        self.assertIn("nonexistent-model-id", str(ctx.exception))

        with self.assertRaises(ModelNotFoundError):
            self.registry.resolve_model_path("nonexistent-model-id")

        with self.assertRaises(ModelNotFoundError):
            self.registry.resolve_mmproj_path("nonexistent-model-id")

    def test_validation_passes_for_current_production_catalog(self) -> None:
        """Verify that current repository models pass full validation."""
        errors = self.registry.validate(check_files_exist=True, raise_for_errors=False)
        self.assertEqual(errors, [])

    def test_duplicate_model_id_in_yaml_fails_cleanly(self) -> None:
        """Duplicate model ID in YAML is caught during loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_config = Path(tmp_dir) / "models.yaml"
            content = (
                "models:\n"
                "  model-a:\n"
                "    id: duplicate-id\n"
                "    display_name: First\n"
                "    type: llm\n"
                "    model_path: path/a.gguf\n"
                "    capabilities: [chat]\n"
                "    role: test\n"
                "    status: verified\n"
                "    runtime: llama.cpp\n"
                "  model-b:\n"
                "    id: duplicate-id\n"
                "    display_name: Second\n"
                "    type: llm\n"
                "    model_path: path/b.gguf\n"
                "    capabilities: [chat]\n"
                "    role: test\n"
                "    status: verified\n"
                "    runtime: llama.cpp\n"
            )
            fake_config.write_text(content, encoding="utf-8")
            with self.assertRaises(ModelConfigError):
                ModelRegistry(config_path=fake_config, project_root=Path(tmp_dir))

    def test_enabled_models_filtering(self) -> None:
        """enabled_models() lists only enabled models."""
        enabled = self.registry.enabled_models()
        self.assertEqual(len(enabled), 7)
        for m in enabled:
            self.assertTrue(m.enabled)
            self.assertTrue(self.registry.is_model_enabled(m.id))


if __name__ == "__main__":
    unittest.main()
