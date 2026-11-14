"""
UCS Rule Engine — base class for rule plug-ins.
Each rule type is a small class in rules/<type>.py.
The engine auto-discovers them.
"""
import os
import importlib
import inspect

# ── Base class ───────────────────────────────────────────────────

class RulePlugin:
    """
    Base class for all rule plug-ins.
    Subclass this in rules/<type>.py and implement:
      - RULE_TYPE: str  (matches rules.rule_type in the DB)
      - describe(definition) -> str
      - apply(definition, sequence_labels, word_lookup) -> list[dict]
      - suggest(definition, sequence_labels, gap_index, word_lookup) -> list[dict]
    """
    RULE_TYPE: str = ""

    @staticmethod
    def describe(definition: dict) -> str:
        """Human-readable summary of this rule."""
        return str(definition)

    @staticmethod
    def apply(definition: dict, sequence_labels: list[str],
              word_lookup: dict) -> list[dict]:
        """
        Check if the rule matches the sequence.
        word_lookup: {canonical: {word_type, meaning, person_number, ...}}
        Returns list of observations: {"message", "matched": True/False, ...}
        """
        return []

    @staticmethod
    def suggest(definition: dict, sequence_labels: list[str],
                gap_index: int, word_lookup: dict) -> list[dict]:
        """
        Given a gap (unknown) at gap_index, suggest candidate words.
        Returns [{"candidate", "reason", "confidence"}, ...]
        """
        return []


# ── Registry ─────────────────────────────────────────────────────

_REGISTRY: dict[str, type[RulePlugin]] = {}


def _discover_plugins():
    """Import all .py files in rules/ and register RulePlugin subclasses."""
    rules_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")
    if not os.path.isdir(rules_dir):
        return

    for fname in sorted(os.listdir(rules_dir)):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        mod_name = f"rules.{fname[:-3]}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, RulePlugin) and obj is not RulePlugin and obj.RULE_TYPE:
                _REGISTRY[obj.RULE_TYPE] = obj


def get_plugin(rule_type: str) -> type[RulePlugin] | None:
    """Return the plugin class for a rule_type, or None."""
    if not _REGISTRY:
        _discover_plugins()
    return _REGISTRY.get(rule_type)


def all_plugins() -> dict[str, type[RulePlugin]]:
    """Return {rule_type: PluginClass} for all discovered plugins."""
    if not _REGISTRY:
        _discover_plugins()
    return dict(_REGISTRY)


def apply_rules(rules: list[dict], sequence_labels: list[str],
                word_lookup: dict) -> list[dict]:
    """
    Apply all given rules to a sequence.
    Returns list of observations from each rule.
    """
    observations = []
    for rule in rules:
        plugin = get_plugin(rule["rule_type"])
        if plugin is None:
            continue
        defn = rule["definition"]
        obs = plugin.apply(defn, sequence_labels, word_lookup)
        for o in obs:
            o["rule_id"] = rule["id"]
            o["rule_type"] = rule["rule_type"]
        observations.extend(obs)
    return observations


def suggest_for_gap(rules: list[dict], sequence_labels: list[str],
                    gap_index: int, word_lookup: dict) -> list[dict]:
    """
    Gather suggestions for an unknown word at gap_index.
    Returns sorted list of candidates from all rules.
    """
    suggestions = []
    for rule in rules:
        plugin = get_plugin(rule["rule_type"])
        if plugin is None:
            continue
        defn = rule["definition"]
        suggs = plugin.suggest(defn, sequence_labels, gap_index, word_lookup)
        for s in suggs:
            s["rule_id"] = rule["id"]
            s["rule_type"] = rule["rule_type"]
        suggestions.extend(suggs)

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s.get("confidence", 0), reverse=True)
    return suggestions


# ── Sentence Arrangement ─────────────────────────────────────────

def arrange_sentence(
    sequence_labels: list[str],
    word_lookup: dict,
    rules: list[dict],
    saved_phrase: dict | None = None,
) -> dict:
    """
    Arrange words into a sentence hypothesis using rules and known words.
    Fulfills Phase 5 Task 3, 5, 6:
      - word-by-word meaning and phrase-level meaning
      - "New sentence found" with ranked suggestions if not saved
      - Infer missing pronouns via agreement rules (e.g. manji -> you singular)
      - Fills gaps with expected word type and candidates, marked "tentative"
    """
    word_by_word = []
    for lbl in sequence_labels:
        info = word_lookup.get(lbl, {})
        word_by_word.append({
            "word": lbl,
            "meaning": info.get("meaning"),
            "word_type": info.get("word_type"),
            "person_number": info.get("person_number"),
            "status": info.get("status", "unverified" if lbl in word_lookup else "unknown"),
        })

    wbw_joined = " ".join(item["meaning"] or "?" for item in word_by_word)
    is_new = saved_phrase is None
    phrase_meaning = saved_phrase.get("meaning") if saved_phrase else None

    # Check for missing pronoun when verbs are present
    has_pronoun = any(item["word_type"] == "pronoun" for item in word_by_word)
    verbs = [item["word"] for item in word_by_word if item["word_type"] == "verb"]

    inferred_pronouns = []
    if not has_pronoun and verbs:
        for r in rules:
            if r.get("rule_type") == "agreement":
                plugin = get_plugin("agreement")
                if plugin:
                    # Look at agreement candidates for this sequence
                    suggs = plugin.suggest(r["definition"], sequence_labels, gap_index=-1, word_lookup=word_lookup)
                    for s in suggs:
                        if s.get("person_number_desc"):
                            inferred_pronouns.append(s)
                        elif s.get("candidate") in word_lookup and word_lookup[s["candidate"]].get("word_type") == "pronoun":
                            inferred_pronouns.append(s)

    # Apply word order rules to determine English reordering
    observations = apply_rules(rules, sequence_labels, word_lookup)
    english_reordered = None
    for obs in observations:
        if obs.get("matched") and "english_reorder" in obs:
            eng_tokens = obs["english_reorder"]
            eng_meanings = [
                word_lookup.get(t, {}).get("meaning", t) for t in eng_tokens
            ]
            english_reordered = " ".join(eng_meanings)
            break

    # Build ranked suggestions
    ranked_suggestions = []
    if is_new:
        if inferred_pronouns:
            # Top suggestion uses inferred pronoun
            top_p = inferred_pronouns[0]
            desc = top_p.get("person_number_desc") or top_p.get("candidate")
            ranked_suggestions.append({
                "type": "missing_pronoun",
                "candidate": top_p["candidate"],
                "suggestion": desc,
                "reason": top_p["reason"],
                "confidence": top_p.get("confidence", 0.85),
                "tentative": True,
            })
        if english_reordered:
            ranked_suggestions.append({
                "type": "word_order",
                "suggestion": english_reordered,
                "reason": "Reordered by grammar word-order rule",
                "confidence": 0.8,
                "tentative": True,
            })
        if wbw_joined and wbw_joined != english_reordered:
            ranked_suggestions.append({
                "type": "literal",
                "suggestion": wbw_joined,
                "reason": "Direct word-by-word joined meaning",
                "confidence": 0.5,
                "tentative": True,
            })

    return {
        "word_by_word": word_by_word,
        "wbw_joined": wbw_joined,
        "is_new_sentence": is_new,
        "phrase_meaning": phrase_meaning,
        "observations": observations,
        "inferred_pronouns": inferred_pronouns,
        "ranked_suggestions": ranked_suggestions,
        "tentative": True if is_new else False,
    }

