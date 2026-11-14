"""
tests/test_sequences.py — Phase 5 tests.

Tests:
1. "inu ising manji" gives "you how are" and phrase "how are you"
2. "ising manji" with no saved phrase gives "New sentence found"
   with "you (singular)" suggestion first
3. Unknown sound inside a sentence becomes new_word_1
4. N-gram counting
5. Rule application and suggestions
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FEATURE_VECTOR_LEN
from sequences import count_ngrams, compare_phrase_vs_words


def _make_vector(base: float = 0.0) -> list[float]:
    return [base] * FEATURE_VECTOR_LEN


# ── Test 1: inu ising manji → "you how are" and phrase "how are you" ──

def test_word_order_arrangement():
    """
    Given words with meanings, word_order rule should reorder
    pronoun+question_word+verb → question_word+verb+pronoun in English.
    "inu ising manji" gives "you how are" and phrase "how are you".
    """
    from rules.word_order import WordOrderRule
    from rule_engine import arrange_sentence

    definition = {
        "pattern": ["pronoun", "question_word", "verb"],
        "english_order": ["question_word", "verb", "pronoun"],
        "label": "question sentence",
    }
    rules = [{"id": 1, "rule_type": "word_order", "definition": definition}]

    word_lookup = {
        "inu": {"word_type": "pronoun", "meaning": "you", "person_number": "2sg"},
        "ising": {"word_type": "question_word", "meaning": "how"},
        "manji": {"word_type": "verb", "meaning": "are", "person_number": "2sg"},
    }

    sequence_labels = ["inu", "ising", "manji"]

    # 1. Word-by-word check: "you how are"
    wbw_meanings = [word_lookup[w]["meaning"] for w in sequence_labels]
    assert " ".join(wbw_meanings) == "you how are"

    # 2. Rule apply check:
    obs = WordOrderRule.apply(definition, sequence_labels, word_lookup)
    assert len(obs) == 1
    assert obs[0]["matched"] is True
    assert obs[0]["english_reorder"] == ["ising", "manji", "inu"]

    # English order meanings: "how are you"
    english_meanings = [
        word_lookup[w]["meaning"] for w in obs[0]["english_reorder"]
    ]
    assert " ".join(english_meanings) == "how are you"

    # 3. arrange_sentence full pipeline
    arranged = arrange_sentence(sequence_labels, word_lookup, rules)
    assert arranged["wbw_joined"] == "you how are"
    # Word order rule suggestion produces "how are you"
    wo_suggs = [s for s in arranged["ranked_suggestions"] if s["type"] == "word_order"]
    assert len(wo_suggs) >= 1
    assert wo_suggs[0]["suggestion"] == "how are you"


# ── Test 2: "ising manji" → "New sentence found" with "you (singular)" suggestion first ──

def test_agreement_suggestion():
    """
    "ising manji" with no saved phrase gives "New sentence found"
    with the "you (singular)" suggestion first.
    When pronoun is missing, agreement rule infers the likely pronoun
    from the verb form. manji seen only with inu → suggest inu (you singular).
    """
    from rules.agreement import AgreementRule
    from rule_engine import arrange_sentence

    definition = {
        "pairs": [
            {"pronoun": "inu", "verb_form": "manji", "person_number": "2sg"},
            {"pronoun": "anu", "verb_form": "mai", "person_number": "1sg"},
            {"pronoun": "iru", "verb_form": "manjeru", "person_number": "3sg"},
        ],
        "label": "pronoun-verb agreement",
    }
    rules = [{"id": 1, "rule_type": "agreement", "definition": definition}]

    sequence_labels = ["ising", "manji"]
    word_lookup = {
        "inu": {"word_type": "pronoun", "meaning": "you", "person_number": "2sg"},
        "anu": {"word_type": "pronoun", "meaning": "I", "person_number": "1sg"},
        "iru": {"word_type": "pronoun", "meaning": "he/she", "person_number": "3sg"},
        "ising": {"word_type": "question_word", "meaning": "how"},
        "manji": {"word_type": "verb", "meaning": "are", "person_number": "2sg"},
    }

    # Direct agreement plugin check
    suggestions = AgreementRule.suggest(
        definition, sequence_labels, gap_index=0, word_lookup=word_lookup,
    )
    assert len(suggestions) >= 1
    assert suggestions[0]["candidate"] == "inu"
    assert "you (singular)" in suggestions[0]["person_number_desc"]

    # Full sentence arrangement check: no saved phrase -> is_new_sentence is True
    arranged = arrange_sentence(sequence_labels, word_lookup, rules, saved_phrase=None)
    assert arranged["is_new_sentence"] is True
    assert arranged["tentative"] is True

    # "you (singular)" suggestion should be first
    assert len(arranged["ranked_suggestions"]) >= 1
    top_sugg = arranged["ranked_suggestions"][0]
    assert top_sugg["type"] == "missing_pronoun"
    assert top_sugg["candidate"] == "inu"
    assert "you (singular)" in top_sugg["suggestion"]


# ── Test 3: Unknown sound → new_word_1 ──────────────────────────

def test_unknown_in_sequence():
    """
    recognize_sequence assigns new_word_N to unknowns and reuses
    the same name for acoustically similar unknowns.
    """
    from recognizer import Recognizer, find_or_create_pending_name

    r = Recognizer()
    v_hello = [1.0] * FEATURE_VECTOR_LEN
    v_bye = [10.0] * FEATURE_VECTOR_LEN
    r.fit([v_hello, v_bye], ["hello", "bye"])

    # Simulate unknown
    v_far = [9999.0] * FEATURE_VECTOR_LEN
    result = r.recognize(v_far)
    assert result["is_unknown"]

    # Assign pending name
    name, reused = find_or_create_pending_name(v_far, [])
    assert name == "new_word_1"
    assert not reused

    # Similar unknown reuses name
    v_far2 = [9999.01] * FEATURE_VECTOR_LEN
    name2, reused2 = find_or_create_pending_name(
        v_far2, [{"name": "new_word_1", "vector": v_far}],
    )
    assert name2 == "new_word_1"
    assert reused2


# ── Test 4: N-gram counting ──────────────────────────────────────

def test_ngram_counting():
    """count_ngrams counts bigrams and trigrams correctly."""
    labels = ["inu", "ising", "manji", "inu", "ising", "manji"]

    bigrams = count_ngrams(labels, 2)
    assert bigrams["inu ising"] == 2
    assert bigrams["ising manji"] == 2

    trigrams = count_ngrams(labels, 3)
    assert trigrams["inu ising manji"] == 2


# ── Test 5: Phrase vs word meanings comparison ───────────────────

def test_compare_phrase_vs_words():
    """Compare a phrase meaning with joined word meanings."""
    result = compare_phrase_vs_words(
        "how are you",
        ["how", "are", "you"],
    )
    assert result["match"] is True

    result2 = compare_phrase_vs_words(
        "what's up",
        ["how", "are", "you"],
    )
    assert result2["match"] is False


# ── Test 6: agreement apply ──────────────────────────────────────

def test_agreement_apply():
    """Agreement rule detects matching and mismatching pronoun-verb pairs."""
    from rules.agreement import AgreementRule

    definition = {
        "pairs": [
            {"pronoun": "inu", "verb_form": "manji", "person_number": "2sg"},
            {"pronoun": "anu", "verb_form": "mai", "person_number": "1sg"},
        ],
        "label": "agreement",
    }

    word_lookup = {
        "inu": {"word_type": "pronoun"},
        "manji": {"word_type": "verb"},
    }

    # Matching pair
    obs = AgreementRule.apply(definition, ["inu", "manji"], word_lookup)
    assert any(o["matched"] for o in obs)

    # Mismatching: anu with manji
    obs2 = AgreementRule.apply(definition, ["anu", "manji"], word_lookup)
    assert any(not o["matched"] for o in obs2)


# ── Test 7: Rule engine discovers plug-ins ───────────────────────

def test_rule_engine_discovery():
    """Rule engine discovers word_order and agreement plugins."""
    from rule_engine import all_plugins
    plugins = all_plugins()
    assert "word_order" in plugins
    assert "agreement" in plugins


if __name__ == "__main__":
    test_word_order_arrangement()
    test_agreement_suggestion()
    test_unknown_in_sequence()
    test_ngram_counting()
    test_compare_phrase_vs_words()
    test_agreement_apply()
    test_rule_engine_discovery()
    print("All sequence tests passed ✓")

