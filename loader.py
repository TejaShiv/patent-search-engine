"""
loader.py — reads the patent dataset into memory.

We do NOT modify the source data. We only:
  1. Read all 64 JSON files into one list of patents.
  2. Strip out empty-string paragraphs from detailed_description (they carry no info).
  3. Tag each patent with `has_garbled_claims` so downstream code / the user
     knows a claim may contain XML-parsing artifacts.
"""

import json
import glob
import os


def _looks_garbled(claim: str) -> bool:
    """
    A claim is treated as garbled if it shows the signature of XML-parsing
    corruption: a duplicated fragment ('( the ...') or two-or-more unclosed
    parentheses (orphaned reference numerals). Single spaced numerals like
    '( 1 )' are NORMAL patent notation, so we do not flag those.
    """
    if "( the" in claim:
        return True
    if claim.count("(") - claim.count(")") >= 2:
        return True
    return False


def load_patents(data_dir: str = "data/patent_data_small") -> list[dict]:
    """Load every patent from every JSON file in data_dir into a single list."""
    patents = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            batch = json.load(f)  # each file is a list of patent dicts

        for patent in batch:
            # Remove empty paragraphs; keep the real text untouched.
            patent["detailed_description"] = [
                para for para in patent.get("detailed_description", [])
                if para.strip()
            ]

            # Flag — but never alter — claims that look corrupted.
            claims = patent.get("claims", [])
            patent["has_garbled_claims"] = any(_looks_garbled(c) for c in claims)

            patents.append(patent)

    return patents


if __name__ == "__main__":
    patents = load_patents()

    total = len(patents)
    flagged = sum(1 for p in patents if p["has_garbled_claims"])

    print(f"Loaded {total} patents from the dataset.")
    print(f"  {flagged} flagged as possibly containing garbled claims "
          f"({100 * flagged / total:.1f}%).")
    print(f"  Source data was not modified.")
    print()
    print("Example record:")
    p = patents[0]
    print(f"  title:          {p['title']}")
    print(f"  doc_number:     {p['doc_number']}")
    print(f"  classification: {p['classification']}")
    print(f"  claims:         {len(p['claims'])} claim(s)")
    print(f"  garbled flag:   {p['has_garbled_claims']}")
