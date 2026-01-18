import argparse
import csv
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "wikipedia-nlp-ml-vs-dl/0.1 (contact:arafatyousufomar3322@gmail.com)"
}



HEADING_RE = re.compile(r"^(=+)\s*(.*?)\s*\1$")


def fetch_plaintext_extract(title: str) -> str:
    """
    Fetch plaintext extract for a Wikipedia page.
    Uses MediaWiki API 'extracts' which returns plain text with section headings like '== Heading =='.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "titles": title,
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)

    r.raise_for_status()
    data = r.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return ""

    page = next(iter(pages.values()))
    extract = page.get("extract", "") or ""
    return extract.strip()


def split_into_section_paragraphs(extract_text: str) -> List[Tuple[str, str, int]]:
    """
    Parse plaintext that includes wiki-style headings and split into (section, paragraph, paragraph_index).
    paragraph_index is per-section starting at 0.
    """
    if not extract_text:
        return []

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", extract_text.strip())
    current_section = "Lead"
    per_section_index: Dict[str, int] = {}
    rows: List[Tuple[str, str, int]] = []

    for b in blocks:
        block = b.strip()
        if not block:
            continue

        # Detect headings like "== Early life ==" or "=== Career ==="
        m = HEADING_RE.match(block)
        if m:
            section_name = m.group(2).strip()
            current_section = section_name if section_name else current_section
            continue

        idx = per_section_index.get(current_section, 0)
        rows.append((current_section, block, idx))
        per_section_index[current_section] = idx + 1

    return rows


def make_paragraph_id(title: str, section: str, paragraph_index: int, text: str) -> str:
    """
    Deterministic ID so reruns don't create new IDs for identical content.
    """
    base = f"{title}||{section}||{paragraph_index}||{text[:120]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def load_existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    existing = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id"):
                existing.add(row["id"])
    return existing


def append_rows(
    out_csv: Path,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()

    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        n = 0
        for r in rows:
            writer.writerow(r)
            n += 1
        return n


def read_titles_from_file(path: Path) -> List[str]:
    titles: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        titles.append(t)
    return titles


def main():
    p = argparse.ArgumentParser(description="Fetch Wikipedia articles and output paragraph-level CSV.")
    p.add_argument("--titles", type=str, default="", help="Pipe-separated titles, e.g. 'Ada Lovelace|Alan Turing'")
    p.add_argument("--titles-file", type=str, default="", help="Path to a text file containing one title per line.")
    p.add_argument("--out", type=str, default="data/raw/paragraphs_unlabeled.csv", help="Output CSV path.")
    p.add_argument("--min-chars", type=int, default=120, help="Minimum characters per paragraph to keep.")
    p.add_argument("--max-paragraphs-per-article", type=int, default=50, help="Cap paragraphs kept per article.")
    args = p.parse_args()

    out_csv = Path(args.out)
    titles: List[str] = []

    if args.titles_file:
        titles.extend(read_titles_from_file(Path(args.titles_file)))
    if args.titles:
        titles.extend([t.strip() for t in args.titles.split("|") if t.strip()])

    if not titles:
        print("No titles provided. Use --titles or --titles-file.", file=sys.stderr)
        sys.exit(1)

    existing_ids = load_existing_ids(out_csv)
    fieldnames = ["id", "title", "url", "section", "paragraph_index", "text"]

    all_new_rows: List[Dict[str, str]] = []
    total_added = 0

    for title in titles:
        try:
            extract = fetch_plaintext_extract(title)
        except Exception as e:
            print(f"[WARN] Failed to fetch '{title}': {e}", file=sys.stderr)
            continue

        if not extract:
            print(f"[WARN] Empty extract for '{title}'", file=sys.stderr)
            continue

        sec_paras = split_into_section_paragraphs(extract)
        kept = 0

        for section, para, para_idx in sec_paras:
            if kept >= args.max_paragraphs_per_article:
                break
            if len(para) < args.min_chars:
                continue

            pid = make_paragraph_id(title, section, para_idx, para)
            if pid in existing_ids:
                continue

            row = {
                "id": pid,
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "section": section,
                "paragraph_index": str(para_idx),
                "text": para.replace("\n", " ").strip(),
            }
            all_new_rows.append(row)
            existing_ids.add(pid)
            kept += 1

        if kept == 0:
            print(f"[INFO] No paragraphs kept for '{title}' (maybe too short).", file=sys.stderr)

        added = append_rows(out_csv, all_new_rows[total_added:], fieldnames)
        total_added += added
        print(f"[OK] '{title}': added {added} paragraphs")

    print(f"\nDone. Total new paragraphs added: {total_added}")
    print(f"Output: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
