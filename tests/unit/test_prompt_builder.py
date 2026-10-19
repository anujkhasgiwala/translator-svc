"""Tests for PromptBuilder system/user split and parser-compatible format."""

from time import time

from translator_svc.prompts import PromptBuilder, TranslationParser
from translator_svc.sessions.base import Phrase


def test_build_returns_system_and_user_prompts() -> None:
    system, user = PromptBuilder().build([], "English", "Spanish")
    assert "English" in system and "Spanish" in system
    # System prompt declares the two-line output contract the parser expects,
    # labeled with the plain language names.
    assert "English:" in system
    assert "Spanish:" in system
    # No prior context -> user prompt reports an empty context.
    assert "Previous Context: None" in user
    assert "audio" in user.lower()


def test_target_dialect_hint_steers_register_without_changing_label() -> None:
    """Spanish target carries the Latin American dialect hint in the directive,
    but the output label stays the plain "Spanish"."""
    system, _ = PromptBuilder().build([], "English", "Spanish")
    # Dialect hint appears in the DIALECT ALIGNMENT directive.
    assert "Latin American Spanish" in system
    # The parser-facing output label is the plain name, not the hint.
    assert "Spanish: <" in system
    assert "Latin American Spanish: <" not in system


def test_user_prompt_includes_prior_context() -> None:
    context = [
        Phrase(
            source_text="Hello",
            target_text="Hola",
            source_lang="English",
            target_lang="Spanish",
            timestamp=time(),
        )
    ]
    _, user = PromptBuilder().build(context, "English", "Spanish")
    assert "Previous Context:" in user
    assert "Previous Context: None" not in user
    assert "Hello" in user
    assert "Hola" in user


def test_prompt_format_round_trips_through_parser() -> None:
    """A model obeying the system prompt's format must parse cleanly."""
    system, _ = PromptBuilder().build([], "English", "Spanish")
    assert "English: <" in system and "Spanish: <" in system
    # Simulate a compliant response and confirm the parser splits it.
    sample = "English: Nice to meet you\nSpanish: Mucho gusto"
    source, target = TranslationParser().parse(sample)
    assert source == "Nice to meet you"
    assert target == "Mucho gusto"
