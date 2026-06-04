"""
Extract all content from TRIBE v2 PDF: text, images, tables → TRIBE_v2_paper.md
"""
import fitz  # PyMuPDF
import pdfplumber
import os
import re
from pathlib import Path

PDF_PATH = "/Users/deep/Desktop/BoldSignal/2605.04326v1.pdf"
OUT_DIR = Path("/Users/deep/Desktop/BoldSignal")
IMG_DIR = OUT_DIR / "tribe_v2_figures"
OUT_MD = OUT_DIR / "TRIBE_v2_paper.md"

IMG_DIR.mkdir(exist_ok=True)

# ── 1. Extract tables with pdfplumber ─────────────────────────────────────────
print("Extracting tables...")
tables_by_page = {}
with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        tbls = page.extract_tables()
        if tbls:
            tables_by_page[i] = tbls

def table_to_md(table):
    """Convert pdfplumber table (list of rows) to Markdown."""
    if not table or not table[0]:
        return ""
    rows = []
    header = [str(c or "").replace("\n", " ").strip() for c in table[0]]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in table[1:]:
        cells = [str(c or "").replace("\n", " ").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)

# ── 2. Extract text + images with PyMuPDF ─────────────────────────────────────
print("Extracting text and images...")
doc = fitz.open(PDF_PATH)
md_parts = []
img_counter = 0
table_counter = 0

# Track which images we've already saved (by xref) to avoid duplicates
seen_xrefs = set()

for page_num in range(len(doc)):
    page = doc[page_num]
    page_label = page_num + 1

    # ── Images on this page ────────────────────────────────────────────────────
    img_list = page.get_images(full=True)
    page_img_mds = []
    for img_info in img_list:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            base_img = doc.extract_image(xref)
            ext = base_img["ext"]
            img_bytes = base_img["image"]
            # Skip tiny images (icons, bullets < 5KB)
            if len(img_bytes) < 5000:
                continue
            img_counter += 1
            fname = f"figure_{img_counter:02d}_p{page_label}.{ext}"
            fpath = IMG_DIR / fname
            fpath.write_bytes(img_bytes)
            w, h = base_img["width"], base_img["height"]
            rel = f"tribe_v2_figures/{fname}"
            page_img_mds.append(f"![Figure {img_counter} (p{page_label}, {w}×{h}px)]({rel})")
        except Exception as e:
            pass

    # ── Text blocks on this page ───────────────────────────────────────────────
    blocks = page.get_text("blocks", sort=True)  # sorted top→bottom, left→right
    text_parts = []
    for b in blocks:
        # b = (x0, y0, x1, y1, text, block_no, block_type)
        if b[6] != 0:  # skip image blocks (type=1)
            continue
        txt = b[4].strip()
        if not txt:
            continue
        text_parts.append(txt)

    # ── Tables on this page ────────────────────────────────────────────────────
    page_table_mds = []
    if page_label in tables_by_page:
        for tbl in tables_by_page[page_label]:
            table_counter += 1
            md_tbl = table_to_md(tbl)
            if md_tbl:
                page_table_mds.append(f"\n**Table {table_counter} (page {page_label})**\n\n{md_tbl}\n")

    # ── Assemble page section ──────────────────────────────────────────────────
    if text_parts or page_img_mds or page_table_mds:
        section = [f"\n\n---\n<!-- Page {page_label} -->\n"]
        if page_img_mds:
            section.extend(page_img_mds)
            section.append("")
        section.extend(text_parts)
        section.extend(page_table_mds)
        md_parts.append("\n".join(section))

doc.close()

# ── 3. Post-process text: detect headings ─────────────────────────────────────
HEADING_PATTERNS = [
    (r"^(\d+)\s+([A-Z][A-Za-z ]{3,60})$", "##"),       # "1 Introduction"
    (r"^(\d+\.\d+)\s+([A-Z][A-Za-z ]{3,60})$", "###"),  # "2.1 Dataset"
    (r"^(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References|Appendix)$", "##"),
]

def upgrade_headings(text):
    lines = text.split("\n")
    out = []
    for line in lines:
        matched = False
        for pattern, prefix in HEADING_PATTERNS:
            if re.match(pattern, line.strip()):
                out.append(f"{prefix} {line.strip()}")
                matched = True
                break
        if not matched:
            out.append(line)
    return "\n".join(out)

full_text = "\n".join(md_parts)
full_text = upgrade_headings(full_text)

# ── 4. Write final Markdown ────────────────────────────────────────────────────
header = """# TRIBE v2: A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience

> **Paper:** arXiv 2605.04326v1
> **Authors:** Stéphane d'Ascoli, Jérémy Rapin, Yohann Benchetrit, Teon Brooks, et al. (FAIR at Meta)
> **Date:** May 7, 2026
> **Code:** https://github.com/facebookresearch/tribev2
> **Weights:** https://huggingface.co/facebook/tribev2
> **Demo:** https://aidemos.atmeta.com/tribev2

---

"""

OUT_MD.write_text(header + full_text, encoding="utf-8")

print(f"\n✓ Done!")
print(f"  Markdown : {OUT_MD}")
print(f"  Figures  : {img_counter} images saved to {IMG_DIR}/")
print(f"  Tables   : {table_counter} tables extracted")
print(f"  File size: {OUT_MD.stat().st_size / 1024:.1f} KB")
