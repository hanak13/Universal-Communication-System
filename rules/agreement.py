"""
Agreement Rule plug-in.

Definition schema:
{
    "pairs": [
        {"pronoun": "inu", "verb_form": "manji", "person_number": "2sg"},
        {"pronoun": "anu", "verb_form": "mai",   "person_number": "1sg"},
        {"pronoun": "iru", "verb_form": "manjeru", "person_number": "3sg"}
    ],
    "label": "pronoun-verb agreement"
}

The pairs list defines which pronoun goes with which verb form.
"""
from rule_engine import RulePlugin


def _format_person_number(pn: str) -> str:
    """Map person/number codes to readable forms like 'you (singular)'."""
    pnl = pn.lower().strip()
    if pnl in ("2sg", "2nd sg", "2nd singular", "2_sg"):
        return "you (singular)"
    if pnl in ("1sg", "1st sg", "1st singular", "1_sg"):
        return "I (singular)"
    if pnl in ("3sg", "3rd sg", "3rd singular", "3_sg"):
        return "he/she (singular)"
    if pnl in ("1pl", "1st pl", "1st plural", "1_pl"):
        return "we (plural)"
    if pnl in ("2pl", "2nd pl", "2nd plural", "2_pl"):
        return "you all (plural)"
    if pnl in ("3pl", "3rd pl", "3rd plural", "3_pl"):
        return "they (plural)"
    return pn


class AgreementRule(RulePlugin):
    RULE_TYPE = "agreement"

    @staticmethod
    def describe(definition: dict) -> str:
        pairs = definition.get("pairs", [])
        parts = [f"{p['pronoun']}+{p['verb_form']}" for p in pairs]
        label = definition.get("label", "agreement")
        return f"{label}: {', '.join(parts)}"

    @staticmethod
    def apply(definition: dict, sequence_labels: list[str],
              word_lookup: dict) -> list[dict]:
        """
        Check if pronouns and verb forms in the sequence agree.
        """
        pairs = definition.get("pairs", [])
        if not pairs:
            return []

        # Build lookup: pronoun → expected verb forms, verb → expected pronouns
        pronoun_to_verbs: dict[str, list[str]] = {}
        verb_to_pronouns: dict[str, list[str]] = {}
        for p in pairs:
            pron = p["pronoun"]
            verb = p["verb_form"]
            pronoun_to_verbs.setdefault(pron, []).append(verb)
            verb_to_pronouns.setdefault(verb, []).append(pron)

        # Find pronouns and verbs in the sequence
        seq_pronouns = []
        seq_verbs = []
        for lbl in sequence_labels:
            if lbl in pronoun_to_verbs:
                seq_pronouns.append(lbl)
            if lbl in verb_to_pronouns:
                seq_verbs.append(lbl)

        observations = []
        for pron in seq_pronouns:
            expected_verbs = pronoun_to_verbs.get(pron, [])
            for verb in seq_verbs:
                if verb in expected_verbs:
                    observations.append({
                        "matched": True,
                        "message": f"Agreement: {pron} + {verb} ✓",
                        "pronoun": pron,
                        "verb": verb,
                    })
                else:
                    # Check if this verb belongs to a different pronoun
                    expected_for_verb = verb_to_pronouns.get(verb, [])
                    if expected_for_verb:
                        observations.append({
                            "matched": False,
                            "message": (
                                f"Agreement mismatch: {pron} + {verb} "
                                f"(expected {pron} + {' or '.join(expected_verbs)})"
                            ),
                            "pronoun": pron,
                            "verb": verb,
                        })

        return observations

    @staticmethod
    def suggest(definition: dict, sequence_labels: list[str],
                gap_index: int, word_lookup: dict) -> list[dict]:
        """
        If a gap exists, use agreement rules to infer the likely word.
        
        Key: only suggest pronouns/verbs that the data actually contains
        in agreement pairs. Do NOT guess forms the data doesn't have.
        """
        pairs = definition.get("pairs", [])
        if not pairs:
            return []

        pronoun_to_verbs: dict[str, list[dict]] = {}
        verb_to_pronouns: dict[str, list[dict]] = {}
        for p in pairs:
            pron = p["pronoun"]
            verb = p["verb_form"]
            pn = p.get("person_number", "")
            pronoun_to_verbs.setdefault(pron, []).append(
                {"verb": verb, "person_number": pn}
            )
            verb_to_pronouns.setdefault(verb, []).append(
                {"pronoun": pron, "person_number": pn}
            )

        # Look at what else is in the sequence (other than the gap)
        other_labels = [
            lbl for i, lbl in enumerate(sequence_labels) if i != gap_index
        ]

        suggestions = []

        # If there's a verb in the sequence, suggest the pronoun that agrees
        for lbl in other_labels:
            if lbl in verb_to_pronouns:
                for entry in verb_to_pronouns[lbl]:
                    raw_pn = entry.get("person_number", "")
                    pn_desc = _format_person_number(raw_pn) if raw_pn else ""
                    desc_str = f" ({pn_desc})" if pn_desc else ""
                    suggestions.append({
                        "candidate": entry["pronoun"],
                        "reason": (
                            f"Agreement: {lbl} seen only with {entry['pronoun']}"
                            f"{desc_str}"
                        ),
                        "confidence": 0.85,
                        "person_number": raw_pn,
                        "person_number_desc": pn_desc,
                    })

        # If there's a pronoun in the sequence, suggest the verb that agrees
        for lbl in other_labels:
            if lbl in pronoun_to_verbs:
                for entry in pronoun_to_verbs[lbl]:
                    raw_pn = entry.get("person_number", "")
                    pn_desc = _format_person_number(raw_pn) if raw_pn else ""
                    desc_str = f" ({pn_desc})" if pn_desc else ""
                    suggestions.append({
                        "candidate": entry["verb"],
                        "reason": (
                            f"Agreement: {lbl} agrees with {entry['verb']}"
                            f"{desc_str}"
                        ),
                        "confidence": 0.85,
                        "person_number": raw_pn,
                        "person_number_desc": pn_desc,
                    })

        return suggestions

