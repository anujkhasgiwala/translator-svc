"""Prompt construction, LLM output parsing, and translation validation.

Three classes with three different reasons to change:
- `PromptBuilder` changes when the prompt template changes.
- `TranslationParser` changes when the LLM output format changes.
- `TranslationValidator` changes when business rules for "valid translation" change.
"""

import re
from collections.abc import Iterable

from .errors import LLMOutputParseError
from .sessions.base import Phrase

# Plain language identifiers map to a richer dialect description used only in
# the prompt's DIALECT ALIGNMENT directive. The wire/label name stays plain
# (e.g. "Spanish") so the identifier is consistent end-to-end; the dialect hint
# (e.g. "Latin American Spanish") steers the model's regional register without
# leaking into the output labels the parser depends on.
_DIALECT_HINTS: dict[str, str] = {
    "English": "American English",
    "Spanish": "Latin American Spanish",
}


class PromptBuilder:
    def __init__(
        self,
        prompt_truncate_chars: int = 200,
    ) -> None:
        self._truncate = prompt_truncate_chars

    def build(
        self,
        context: Iterable[Phrase],
        source_lang: str,
        target_lang: str,
    ) -> tuple[str, str]:
        """Builds the (system_prompt, user_prompt) pair for one phrase.

        The system prompt carries the stable role and output contract; the user
        prompt carries the per-request language pair, prior context, and task.
        The output format (a source transcript line followed by a
        ``<TargetLang>: <translation>`` line) is what ``TranslationParser``
        depends on — keep it in sync if either changes.
        """
        # source_lang / target_lang are plain language names (e.g. "English",
        # "Spanish"). These names are used verbatim as the output-line labels.
        src_name = source_lang
        tgt_name = target_lang

        system_prompt = self._build_system_prompt(src_name, tgt_name)
        user_prompt = self._build_user_prompt(context, src_name, tgt_name)
        return system_prompt, user_prompt

    def _build_system_prompt(self, src_name: str, tgt_name: str) -> str:
        # The dialect hint steers regional register (e.g. Latin American
        # Spanish) without changing the plain label the parser expects.
        tgt_dialect = _DIALECT_HINTS.get(tgt_name, tgt_name)
        return (
            f"You are an expert real-time polyglot interpreter. Your goal is to "
            f"provide a seamless bridge between {src_name} and {tgt_name} speakers."
            "\n\n"
            "Core Directives:\n"
            f"1. DIALECT ALIGNMENT: Use the most common, natural, and contemporary "
            f"spoken forms of {tgt_dialect}. Prioritize modern social idioms and "
            "conversational greetings over formal academic or textbook translations.\n"
            "2. TONE PARITY: Match the speaker's level of familiarity. If the source is "
            "casual, the translation must be casual.\n"
            "3. NO METADATA: Do not include internal thoughts, apologies, or explanations "
            "in the final output.\n"
            "\n"
            "Strict Output Format (Exactly two lines):\n"
            f"{src_name}: <verbatim transcript of the audio>\n"
            f"{tgt_name}: <natural, idiomatic translation>"
        )

    def _build_user_prompt(
        self, context: Iterable[Phrase], src_name: str, tgt_name: str
    ) -> str:
        context_lines: list[str] = []
        for phrase in context:
            context_lines.append(
                f"{src_name}: {self._truncate_text(phrase.source_text)}"
            )
            context_lines.append(
                f"{tgt_name}: {self._truncate_text(phrase.target_text)}"
            )
        context_text = " ".join(context_lines)

        return (
            f"Interpret this {src_name} audio into {tgt_name} using the required two-line format. "
            f"Previous Context: {context_text if context_text else 'None'}"
        )

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self._truncate:
            return text
        return text[: self._truncate] + "..."


class TranslationParser:
    """Splits the LLM's dual-output response into (source_text, target_text).

    The prompt asks the model to emit the transcript first, then a labeled
    `<target_lang>: <translation>` line. In practice, Gemma 4 sometimes labels
    the source line (`English: ...`, `Spanish: ...`) and sometimes leaves it
    bare. We accept either shape: any text before the last labeled line is the
    transcript, and the last labeled line's value is the translation.
    """

    # Label is a plain language name (e.g. "English", "Spanish"). The optional
    # punctuation in the class keeps the parser tolerant of models that emit a
    # qualified label anyway (e.g. "English (US):").
    _LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z\s.()]*?)\s*:\s*(.*)$")

    def parse(self, raw_output: str) -> tuple[str, str]:
        lines = [line for line in raw_output.splitlines() if line.strip()]
        if not lines:
            raise LLMOutputParseError(f"Empty model output: {raw_output!r}")

        last_label_idx: int | None = None
        last_label_value = ""
        for idx, line in enumerate(lines):
            match = self._LINE_RE.match(line)
            if match:
                last_label_idx = idx
                last_label_value = match.group(2).strip()

        if last_label_idx is None or not last_label_value:
            raise LLMOutputParseError(
                f"No labeled translation line found: {raw_output!r}"
            )

        target_text = last_label_value
        source_lines = lines[:last_label_idx]
        source_text = self._strip_optional_label(
            " ".join(line.strip() for line in source_lines).strip()
        )
        if not source_text:
            raise LLMOutputParseError(
                f"Empty source transcript before translation line: {raw_output!r}"
            )
        return source_text, target_text

    def _strip_optional_label(self, text: str) -> str:
        match = self._LINE_RE.match(text)
        return match.group(2).strip() if match else text


class TranslationValidator:
    def __init__(
        self,
        min_ratio: float = 0.3,
        max_ratio: float = 4.0,
        refusal_patterns: Iterable[str] | None = None,
    ) -> None:
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._refusal_patterns = tuple(
            p.lower()
            for p in (
                refusal_patterns
                if refusal_patterns is not None
                else ("i cannot", "i can't translate", "as an ai", "no puedo")
            )
        )

    def is_valid(self, source_text: str, target_text: str) -> bool:
        src = source_text.strip()
        tgt = target_text.strip()
        if not src or not tgt:
            return False
        ratio = len(tgt) / max(len(src), 1)
        if ratio < self._min_ratio or ratio > self._max_ratio:
            return False
        lowered = tgt.lower()
        if any(pattern in lowered for pattern in self._refusal_patterns):
            return False
        return True
