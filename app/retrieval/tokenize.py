"""Tokenization for sparse retrieval.

Indic scripts use combining marks (matras, virama) that Python's ``\\w``
does not classify as word characters, so plain regex word-matching breaks
Devanagari etc. into single characters. We instead keep only Unicode word
characters plus the Indic/Arabic script blocks and combining diacritics.
"""

from __future__ import annotations

import re

_SCRIPT_BLOCKS = (
    "\u0900-\u0dff"  # major Indic blocks (Devanagari..Sinhala)
    "\ua8e0-\ua8ff"  # Devanagari extended
    "\u0600-\u06ff"  # Arabic (Urdu)
    "\u0750-\u077f"  # Arabic supplement
    "\u08a0-\u08ff"  # Arabic extended-A
    "\u0300-\u036f"  # combining diacritical marks
)

_WORD_CHAR = re.compile(rf"[\w{_SCRIPT_BLOCKS}]")
_WS = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for token in _WS.split(text.lower()):
        clean = "".join(_WORD_CHAR.findall(token))
        if clean:
            out.append(clean)
    return out
