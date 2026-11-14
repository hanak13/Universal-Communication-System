"""
tests/test_words.py — confidence calculation, variant lookup, undo.
Run:  python -m pytest tests/ -v
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import (
    init_db, save_word, get_word_by_id, get_all_words,
    update_word, confirm_word, reject_word, word_confidence,
    add_variant, get_variants, lookup_by_variant,
    log_history, undo_last_change, get_recent_history,
    get_word_by_canonical,
)


def _fresh_conn():
    """Create a temp DB and return connection."""
    tmp = tempfile.mkdtemp()
    return init_db(os.path.join(tmp, "test.db")), tmp


def test_confidence_calculation():
    """Confidence = confirmations / (confirmations + rejections)."""
    conn, _ = _fresh_conn()

    wid = save_word(conn, canonical="inu", meaning="dog")
    w = get_word_by_id(conn, wid)

    # Initially 0/0
    conf, total = word_confidence(w)
    assert conf == 0.0
    assert total == 0

    # 3 confirmations
    confirm_word(conn, wid)
    confirm_word(conn, wid)
    confirm_word(conn, wid)
    w = get_word_by_id(conn, wid)
    conf, total = word_confidence(w)
    assert total == 3
    assert conf == 1.0  # 3/3

    # 1 rejection  → 3/4 = 0.75
    reject_word(conn, wid, correction="cat, not dog")
    w = get_word_by_id(conn, wid)
    conf, total = word_confidence(w)
    assert total == 4
    assert conf == 0.75

    conn.close()


def test_variant_lookup():
    """lookup_by_variant finds a word by canonical or variant spelling."""
    conn, _ = _fresh_conn()

    wid = save_word(conn, canonical="inu", meaning="dog")
    add_variant(conn, wid, "eenu")
    add_variant(conn, wid, "inuu")

    # By canonical
    w = lookup_by_variant(conn, "inu")
    assert w is not None
    assert w["id"] == wid

    # By variant
    w2 = lookup_by_variant(conn, "eenu")
    assert w2 is not None
    assert w2["id"] == wid

    w3 = lookup_by_variant(conn, "inuu")
    assert w3 is not None
    assert w3["id"] == wid

    # Not found
    w4 = lookup_by_variant(conn, "neko")
    assert w4 is None

    # Variants list
    variants = get_variants(conn, wid)
    assert set(variants) == {"eenu", "inuu"}

    conn.close()


def test_undo_meaning_change():
    """Undo reverts the most recent meaning change."""
    conn, _ = _fresh_conn()

    wid = save_word(conn, canonical="neko", meaning="cat")
    w = get_word_by_id(conn, wid)
    assert w["meaning"] == "cat"

    # Change meaning
    update_word(conn, wid, meaning="kitten")
    w = get_word_by_id(conn, wid)
    assert w["meaning"] == "kitten"

    # Undo
    result = undo_last_change(conn)
    assert result is not None
    assert "neko" not in result or "meaning" in result  # description mentions it

    w = get_word_by_id(conn, wid)
    assert w["meaning"] == "cat"

    conn.close()


def test_undo_status_change():
    """Undo reverts a status change."""
    conn, _ = _fresh_conn()

    wid = save_word(conn, canonical="ohayo", meaning="good morning")
    update_word(conn, wid, status="verified")
    w = get_word_by_id(conn, wid)
    assert w["status"] == "verified"

    undo_last_change(conn)
    w = get_word_by_id(conn, wid)
    assert w["status"] == "unverified"

    conn.close()


def test_undo_nothing():
    """Undo on empty history returns None."""
    conn, _ = _fresh_conn()
    result = undo_last_change(conn)
    assert result is None
    conn.close()


def test_word_status_default():
    """New words default to 'unverified'."""
    conn, _ = _fresh_conn()
    wid = save_word(conn, canonical="test_word")
    w = get_word_by_id(conn, wid)
    assert w["status"] == "unverified"
    conn.close()


def test_history_logged():
    """Changes produce history entries."""
    conn, _ = _fresh_conn()
    wid = save_word(conn, canonical="kawa", meaning="river")
    update_word(conn, wid, meaning="stream")

    history = get_recent_history(conn, limit=5)
    assert len(history) >= 1
    # Find the meaning change
    found = False
    for h in history:
        if h["table_name"] == "words" and h["row_id"] == wid:
            if h["old_value"] and "river" in h["old_value"]:
                found = True
    assert found, "History entry for meaning change not found"
    conn.close()


if __name__ == "__main__":
    test_confidence_calculation()
    test_variant_lookup()
    test_undo_meaning_change()
    test_undo_status_change()
    test_undo_nothing()
    test_word_status_default()
    test_history_logged()
    print("All word tests passed ✓")
