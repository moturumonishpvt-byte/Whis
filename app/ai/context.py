"""Context management for WHIS inference.

Determines safe context budgets from model configuration, constructs
deterministic prompt structures, prevents context overflow, and exposes
the final prompt used by InferenceEngine without requiring a tokenizer
framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.core.exceptions import WHISError


# Conservative chars-per-token estimate for context budget calculations.
# Empirical average across English/code text: ~3.5 chars per token (we use 4
# for safety, slightly over-estimating prompt length to avoid overflow).
CHARS_PER_TOKEN_ESTIMATE: float = 4.0

# Default generation token budget reserved from the context window.
DEFAULT_GENERATION_TOKENS: int = 512


class ContextError(WHISError):
    """Raised when context budget or prompt construction fails."""


class ContextOverflowError(ContextError):
    """Raised when the supplied prompt exceeds the available input budget."""

    def __init__(
        self,
        message: str,
        estimated_prompt_tokens: int = 0,
        input_budget_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.estimated_prompt_tokens = estimated_prompt_tokens
        self.input_budget_tokens = input_budget_tokens


@dataclass
class Message:
    """A single turn in a conversation history."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass(frozen=True)
class ContextBudget:
    """Computed context allocation for a single inference request."""

    model_context_tokens: int
    generation_tokens: int
    input_budget_tokens: int
    estimated_prompt_tokens: int
    chars_per_token: float

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining after prompt estimation."""
        return self.input_budget_tokens - self.estimated_prompt_tokens

    @property
    def is_safe(self) -> bool:
        """Return True if estimated prompt fits within the input budget."""
        return self.estimated_prompt_tokens <= self.input_budget_tokens


@dataclass(frozen=True)
class BuiltContext:
    """Final prompt and metadata produced by ContextManager."""

    prompt: str
    system_prompt: Optional[str]
    model_id: str
    budget: ContextBudget

    @property
    def is_empty(self) -> bool:
        return not self.prompt.strip()


class ContextManager:
    """Builds and validates inference prompts within model context limits.

    Uses a conservative character-to-token estimate (default 4 chars/token)
    to avoid context overflow without requiring a tokenizer dependency.
    """

    def __init__(
        self,
        generation_tokens: int = DEFAULT_GENERATION_TOKENS,
        chars_per_token: float = CHARS_PER_TOKEN_ESTIMATE,
    ) -> None:
        if generation_tokens < 1:
            raise ValueError("generation_tokens must be at least 1")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.generation_tokens: int = generation_tokens
        self.chars_per_token: float = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Return a conservative upper-bound token count for the given text."""
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token) + 1)

    def compute_budget(
        self,
        model_context_tokens: int,
        prompt_text: str,
    ) -> ContextBudget:
        """Compute the context budget for a given model context and prompt.

        Args:
            model_context_tokens: Maximum context window size in tokens.
            prompt_text: Fully constructed prompt string to evaluate.

        Returns:
            ContextBudget describing allocation breakdown.
        """
        if model_context_tokens < 1:
            raise ContextError(f"model_context_tokens must be >= 1, got {model_context_tokens}")

        generation_tokens = min(self.generation_tokens, model_context_tokens - 1)
        input_budget_tokens = max(0, model_context_tokens - generation_tokens)
        estimated_prompt_tokens = self.estimate_tokens(prompt_text)

        return ContextBudget(
            model_context_tokens=model_context_tokens,
            generation_tokens=generation_tokens,
            input_budget_tokens=input_budget_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            chars_per_token=self.chars_per_token,
        )

    def build(
        self,
        user_prompt: str,
        model_id: str,
        model_context_tokens: int,
        system_prompt: Optional[str] = None,
        history: Optional[List[Message]] = None,
    ) -> BuiltContext:
        """Construct a deterministic prompt within context limits.

        Combines system prompt, optional conversation history, and the user
        prompt into a single string suitable for llama-cli's -p flag.
        Raises ContextOverflowError if the estimated token count exceeds the
        input budget.

        Args:
            user_prompt: The current user instruction or question.
            model_id: Identifier of the model being used (for error reporting).
            model_context_tokens: Max context size in tokens for the model.
            system_prompt: Optional system instruction prepended to the prompt.
            history: Optional list of previous conversation turns to include.

        Returns:
            BuiltContext with the final prompt string and budget metadata.

        Raises:
            ContextError: If user_prompt is empty or context tokens are invalid.
            ContextOverflowError: If the estimated prompt exceeds input budget.
        """
        if not user_prompt or not user_prompt.strip():
            raise ContextError("user_prompt must not be empty.")

        parts: List[str] = []

        if system_prompt and system_prompt.strip():
            parts.append(f"System: {system_prompt.strip()}")

        if history:
            for msg in history:
                role_label = msg.role.capitalize()
                parts.append(f"{role_label}: {msg.content.strip()}")

        parts.append(f"User: {user_prompt.strip()}")
        parts.append("Assistant:")

        full_prompt = "\n".join(parts)

        budget = self.compute_budget(model_context_tokens, full_prompt)

        if not budget.is_safe:
            raise ContextOverflowError(
                f"Prompt for model '{model_id}' exceeds input budget: "
                f"estimated {budget.estimated_prompt_tokens} tokens > "
                f"{budget.input_budget_tokens} token input budget "
                f"(context={model_context_tokens}, generation_reserved={budget.generation_tokens}).",
                estimated_prompt_tokens=budget.estimated_prompt_tokens,
                input_budget_tokens=budget.input_budget_tokens,
            )

        return BuiltContext(
            prompt=full_prompt,
            system_prompt=system_prompt,
            model_id=model_id,
            budget=budget,
        )
