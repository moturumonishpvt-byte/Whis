"""Unit tests for ContextManager and context budget calculation in WHIS."""

import unittest

from app.ai.context import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_GENERATION_TOKENS,
    BuiltContext,
    ContextBudget,
    ContextError,
    ContextManager,
    ContextOverflowError,
    Message,
)


class TestContextManager(unittest.TestCase):
    """Test suite for ContextManager, ContextBudget, and prompt construction."""

    def setUp(self) -> None:
        self.cm = ContextManager(generation_tokens=256, chars_per_token=4.0)

    def test_init_validation(self) -> None:
        """Validate initialization parameters."""
        with self.assertRaises(ValueError):
            ContextManager(generation_tokens=0)
        with self.assertRaises(ValueError):
            ContextManager(chars_per_token=0)
        with self.assertRaises(ValueError):
            ContextManager(chars_per_token=-1.5)

    def test_estimate_tokens(self) -> None:
        """Test token estimation based on character length."""
        self.assertEqual(self.cm.estimate_tokens(""), 0)
        # 4 chars = 1 + 1 = 2 tokens conservatively
        self.assertEqual(self.cm.estimate_tokens("test"), 2)
        # 16 chars = 16/4 + 1 = 5 tokens
        self.assertEqual(self.cm.estimate_tokens("1234567890123456"), 5)

    def test_compute_budget_safe(self) -> None:
        """Test budget computation when prompt fits well within context."""
        budget = self.cm.compute_budget(
            model_context_tokens=2048,
            prompt_text="Hello world",
        )
        self.assertEqual(budget.model_context_tokens, 2048)
        self.assertEqual(budget.generation_tokens, 256)
        self.assertEqual(budget.input_budget_tokens, 2048 - 256)
        self.assertTrue(budget.is_safe)
        self.assertGreater(budget.remaining_tokens, 0)

    def test_compute_budget_invalid_context(self) -> None:
        """Test that invalid model context tokens raise ContextError."""
        with self.assertRaises(ContextError):
            self.cm.compute_budget(model_context_tokens=0, prompt_text="hi")

    def test_build_simple_prompt(self) -> None:
        """Test building a basic prompt without system prompt or history."""
        built = self.cm.build(
            user_prompt="What is Python?",
            model_id="qwen3.5-4b",
            model_context_tokens=2048,
        )
        self.assertIn("User: What is Python?", built.prompt)
        self.assertIn("Assistant:", built.prompt)
        self.assertEqual(built.model_id, "qwen3.5-4b")
        self.assertIsNone(built.system_prompt)
        self.assertTrue(built.budget.is_safe)

    def test_build_with_system_prompt_and_history(self) -> None:
        """Test prompt construction with system prompt and conversation history."""
        history = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello! How can I help?"),
        ]
        built = self.cm.build(
            user_prompt="Explain recursion.",
            model_id="qwen3.5-4b",
            model_context_tokens=2048,
            system_prompt="You are a helpful coding assistant.",
            history=history,
        )
        self.assertIn("System: You are a helpful coding assistant.", built.prompt)
        self.assertIn("User: Hi", built.prompt)
        self.assertIn("Assistant: Hello! How can I help?", built.prompt)
        self.assertIn("User: Explain recursion.", built.prompt)
        self.assertIn("Assistant:", built.prompt)

    def test_build_empty_prompt_raises(self) -> None:
        """Test that empty or whitespace-only prompt raises ContextError."""
        with self.assertRaises(ContextError):
            self.cm.build(
                user_prompt="",
                model_id="qwen3.5-4b",
                model_context_tokens=2048,
            )
        with self.assertRaises(ContextError):
            self.cm.build(
                user_prompt="   \n  \t ",
                model_id="qwen3.5-4b",
                model_context_tokens=2048,
            )

    def test_build_context_overflow_raises(self) -> None:
        """Test that prompt exceeding input budget raises ContextOverflowError."""
        # Very small context window: 64 tokens, generation reserved: 32 tokens
        small_cm = ContextManager(generation_tokens=32, chars_per_token=4.0)
        huge_prompt = "A" * 1000  # ~251 estimated tokens > 32 token budget

        with self.assertRaises(ContextOverflowError) as ctx:
            small_cm.build(
                user_prompt=huge_prompt,
                model_id="qwen3.5-4b",
                model_context_tokens=64,
            )

        self.assertIn("exceeds input budget", str(ctx.exception))
        self.assertGreater(ctx.exception.estimated_prompt_tokens, ctx.exception.input_budget_tokens)


if __name__ == "__main__":
    unittest.main()
