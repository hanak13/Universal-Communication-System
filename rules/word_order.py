"""
Word Order Rule plug-in.

Definition schema:
{
    "pattern": ["pronoun", "question_word", "verb"],
    "english_order": ["question_word", "verb", "pronoun"],
    "label": "question sentence"
}

The pattern is the expected word_type order in the target language.
english_order maps how they reorder in English.
"""
from rule_engine import RulePlugin


class WordOrderRule(RulePlugin):
    RULE_TYPE = "word_order"

    @staticmethod
    def describe(definition: dict) -> str:
        pattern = " + ".join(definition.get("pattern", []))
        eng = " + ".join(definition.get("english_order", []))
        label = definition.get("label", "")
        return f"{label}: {pattern} → English: {eng}"

    @staticmethod
    def apply(definition: dict, sequence_labels: list[str],
              word_lookup: dict) -> list[dict]:
        """
        Check if the word types in the sequence match the pattern.
        """
        pattern = definition.get("pattern", [])
        if not pattern:
            return []

        # Get word types for each label in the sequence
        seq_types = []
        for lbl in sequence_labels:
            info = word_lookup.get(lbl)
            if info and info.get("word_type"):
                seq_types.append(info["word_type"])
            else:
                seq_types.append(None)

        # Check if the sequence types match the pattern
        if len(seq_types) != len(pattern):
            return []

        matched = all(
            st == pt or st is None  # None = unknown type, might match
            for st, pt in zip(seq_types, pattern)
        )

        if matched:
            # Build English reordering
            eng_order = definition.get("english_order", pattern)
            # Map types to word meanings
            type_to_words: dict[str, list[str]] = {}
            for lbl, wtype in zip(sequence_labels, seq_types):
                if wtype:
                    type_to_words.setdefault(wtype, []).append(lbl)

            english_words = []
            type_counters: dict[str, int] = {}
            for etype in eng_order:
                idx = type_counters.get(etype, 0)
                words = type_to_words.get(etype, [])
                if idx < len(words):
                    english_words.append(words[idx])
                else:
                    english_words.append("?")
                type_counters[etype] = idx + 1

            return [{
                "matched": True,
                "message": f"Word order matches pattern: {definition.get('label', '')}",
                "english_reorder": english_words,
                "pattern": pattern,
            }]

        return [{"matched": False, "message": "Word order does not match pattern."}]

    @staticmethod
    def suggest(definition: dict, sequence_labels: list[str],
                gap_index: int, word_lookup: dict) -> list[dict]:
        """
        If a gap is at gap_index, suggest what word type should go there
        based on the pattern.
        """
        pattern = definition.get("pattern", [])
        if gap_index >= len(pattern) or len(sequence_labels) != len(pattern):
            return []

        expected_type = pattern[gap_index]

        # Find all known words of that type
        candidates = []
        for canonical, info in word_lookup.items():
            if info.get("word_type") == expected_type:
                candidates.append({
                    "candidate": canonical,
                    "reason": f"Expected {expected_type} at position {gap_index + 1} "
                              f"(pattern: {definition.get('label', '')})",
                    "confidence": 0.5,
                    "expected_type": expected_type,
                })

        return candidates

