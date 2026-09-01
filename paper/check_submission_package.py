from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
MAIN_PDF = PAPER_DIR / "main.pdf"
EC_PDF = PAPER_DIR / "electronic_companion.pdf"
MAIN_AUX = PAPER_DIR / "main.aux"


def command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=PAPER_DIR, text=True, errors="replace")


def pdf_pages(path: Path) -> int:
    info = command_output("pdfinfo", path.name)
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not read page count from {path}")
    return int(match.group(1))


def pdf_text(path: Path, *, first_page_only: bool = False) -> str:
    args = ["pdftotext"]
    if first_page_only:
        args.extend(["-f", "1", "-l", "1"])
    args.extend([path.name, "-"])
    return command_output(*args)


def abstract_word_count() -> int:
    first_page = pdf_text(MAIN_PDF, first_page_only=True)
    match = re.search(
        r"Abstract\.\s*(.*?)\s*Key words:",
        first_page,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError("could not isolate the rendered abstract")
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", match.group(1)))


def references_start_page() -> int:
    aux = MAIN_AUX.read_text(errors="replace")
    match = re.search(
        r"\\newlabel\{page:references-start\}\{\{.*?\}\{(\d+)\}",
        aux,
    )
    if match is None:
        raise RuntimeError("main.aux does not record the references start page")
    return int(match.group(1))


def main() -> None:
    abstract_words = abstract_word_count()
    main_pages = pdf_pages(MAIN_PDF)
    ec_pages = pdf_pages(EC_PDF)
    reference_page = references_start_page()
    manuscript_pages = reference_page - 1
    main_text = pdf_text(MAIN_PDF)
    ec_text = pdf_text(EC_PDF)

    checks = {
        "abstract_at_most_200_words": abstract_words <= 200,
        "lengthy_manuscript_at_most_40_pages_excluding_references": manuscript_pages <= 40,
        "electronic_companion_not_longer_than_manuscript": ec_pages <= manuscript_pages,
        "main_excludes_ec_proof_body": "EC. Conventional Proof Details" not in main_text,
        "ec_contains_proof_body": "EC. Conventional Proof Details" in ec_text,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "submission_type": "Lengthy Manuscript",
        "abstract_words": abstract_words,
        "main_pdf_pages_including_references": main_pages,
        "references_start_page": reference_page,
        "manuscript_pages_excluding_references": manuscript_pages,
        "electronic_companion_pages": ec_pages,
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
