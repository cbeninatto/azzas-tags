import io
import os
import zipfile
import re
from pathlib import Path

import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader, PdfWriter  # for duplicating pages, merging, rotating


# ---------- Config ----------
st.set_page_config(
    page_title="ZPL Label Helper",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ ZPL Label Helper")
st.caption(
    "Option 1: Product Hangtag (Arezzo-style: OpenAI standardization → LabelZoom PDF with PQ duplication, merged per SKU) • "
    "Option 2: Carton barcodes (10×4 cm @ 203 DPI → PDF + sequence check, merged per SKU) • "
    "Option 3: Reserva hangtag barcode (8×3.5 cm @ 203 DPI → rotated PDF, merged per SKU)"
)


# ---------- Secrets / clients ----------
OPENAI_API_KEY = st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
LABELZOOM_API_KEY = st.secrets.get("labelzoom_api_key", os.getenv("LABELZOOM_API_KEY"))

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

LABELZOOM_ZPL_TO_PDF_URL = "https://prod-api.labelzoom.net/api/v2/convert/zpl/to/pdf"


# ---------- OpenAI system prompt for Option 1 ----------
STANDARDIZATION_SYSTEM_PROMPT = """
You are a ZPL label expert that standardizes PRODUCT HANGTAG labels.

General rules:
- Input is raw ZPL from a .prn file for a product hangtag.
- Output must be VALID ZPL ONLY (no explanations), starting with ^XA and ending with ^XZ.
- Never change any dynamic text content (product line, material, code, barcodes, SKUs, etc.).
- Never change the print quantity ^PQ value; keep exactly the same quantity and parameters as in the input.

Physical size (5.5 cm x 2.5 cm @ 203 DPI):
- Assume printer density 203 DPI (~8 dots/mm).
- Set the label size to:
  - ^PW440   (5.5 cm width)
  - ^LL200   (2.5 cm height)
- Origin must be top-left:
  - ^LH0,0
- If other ^PW, ^LL or ^LH commands are present, replace them with these values and make sure there is only one set (^PW440, ^LL200, ^LH0,0).

Coordinate standardization:
- Replace every occurrence of ^FO30 with ^FO40 (only when 30 is the X coordinate, e.g. ^FO30,xxx).
- Replace every occurrence of ^FO330 with ^FO340 (only when 330 is the X coordinate, e.g. ^FO330,xxx).
- Leave all other coordinates unchanged.

Layout cleanup:
- Remove any duplicate HUMAN-READABLE barcode text:
  - If the same numeric string that is encoded in a barcode (^BC, ^BEN, etc.) also appears as a separate text field, remove ONLY that text field and keep the barcode command.
- Keep all other text, fonts (^A...), box/graphic elements (^GB, ^FR, etc.) and content exactly as in the input.
- Ensure the final label contains ^PW440, ^LL200 and ^LH0,0 near the top of the format (after ^XA).

Spacing & alignment:
- Preserve the existing logical order but ensure the vertical hierarchy is:
  1) Product Line
  2) Material
  3) Code
  4) Barcode
  5) Box/border
- Align barcode, its surrounding box (if any), and related text vertically in a clean, readable layout by adjusting only X/Y coordinates when needed.
- Use consistent spacing between lines; you may adjust Y coordinates slightly to tidy up spacing.

Important:
- DO NOT add any comments or explanations.
- DO NOT wrap the ZPL in Markdown code fences.
- DO NOT remove or change ^PQ except to preserve exactly the same value and parameters as in the input.
"""


# ---------- General helpers ----------
def read_prn_file(uploaded_file):
    """Read a .prn file as text, trying UTF-8 then Latin-1."""
    content = uploaded_file.read()
    if isinstance(content, str):
        return content
    for encoding in ("utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except Exception:
            continue
    return content.decode("latin-1", errors="ignore")


def split_zpl_into_formats(zpl: str):
    """
    Split a raw ZPL string into individual ^XA ... ^XZ formats.
    Returns a list of ZPL segments. If no ^XA/^XZ found, returns [zpl] or [] if empty.
    """
    segments = re.findall(r"\^XA.*?\^XZ", zpl, flags=re.IGNORECASE | re.DOTALL)
    segments = [s.strip() for s in segments if s.strip()]
    if not segments:
        cleaned = zpl.strip()
        return [cleaned] if cleaned else []
    return segments


def strip_code_fences(text: str) -> str:
    """Strip ```...``` code fences if the model adds them."""
    if "```" not in text:
        return text.strip()
    lines = text.splitlines()
    cleaned = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            cleaned.append(line)
    if cleaned:
        return "\n".join(cleaned).strip()
    return text.strip()


def standardize_zpl_with_openai(raw_zpl: str, model: str = "gpt-5.1-mini") -> str:
    if client is None:
        raise RuntimeError("OpenAI API key is not configured.")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": STANDARDIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": raw_zpl},
        ],
    )
    zpl_text = response.choices[0].message.content or ""
    return strip_code_fences(zpl_text)


def enforce_hangtag_standard(zpl: str) -> str:
    """
    Deterministic post-processing to guarantee:
    - ^PW440, ^LL200, ^LH0,0
    - ^FO30, -> ^FO40, and ^FO330, -> ^FO340,
    """
    text = zpl.replace("\r\n", "\n")

    # Replace specific FO coordinates
    text = re.sub(r"\^FO30,", "^FO40,", text)
    text = re.sub(r"\^FO330,", "^FO340,", text)

    # Remove existing ^PW, ^LL, ^LH
    text = re.sub(r"\^PW-?\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\^LL-?\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\^LH-?\d+,-?\d+", "", text, flags=re.IGNORECASE)

    # Insert our standard size/origin right after ^XA
    match = re.search(r"\^XA", text, flags=re.IGNORECASE)
    size_cmds = "^PW440^LL200^LH0,0"
    if match:
        insert_at = match.end()
        text = text[:insert_at] + size_cmds + text[insert_at:]
    else:
        text = "^XA" + size_cmds + text

    return text


def apply_manual_label_size(
    zpl: str,
    width_cm: float,
    height_cm: float,
    dpi: int = 203,
) -> str:
    """Apply label size using ^PW (width) and ^LL (height) for a given size and DPI."""
    dpmm = int(round(dpi / 25.4))
    width_mm = width_cm * 10.0
    height_mm = height_cm * 10.0
    width_dots = int(round(width_mm * dpmm))
    height_dots = int(round(height_mm * dpmm))

    cleaned = re.sub(r"\^PW-?\d+", "", zpl, flags=re.IGNORECASE)
    cleaned = re.sub(r"\^LL-?\d+", "", cleaned, flags=re.IGNORECASE)

    match = re.search(r"\^XA", cleaned, flags=re.IGNORECASE)
    size_cmds = f"^PW{width_dots}^LL{height_dots}"
    if match:
        insert_at = match.end()
        return cleaned[:insert_at] + size_cmds + cleaned[insert_at:]
    else:
        return "^XA" + size_cmds + cleaned


def convert_zpl_to_pdf_with_labelzoom(zpl: str) -> bytes:
    if not LABELZOOM_API_KEY:
        raise RuntimeError("LabelZoom API key is not configured.")
    headers = {
        "Authorization": f"Bearer {LABELZOOM_API_KEY}",
        "Content-Type": "text/plain",
        "Accept": "application/pdf",
    }
    resp = requests.post(
        LABELZOOM_ZPL_TO_PDF_URL,
        data=zpl.encode("utf-8"),
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def rotate_pdf(pdf_bytes: bytes, degrees: int = 90) -> bytes:
    """
    Rotate all pages in a PDF by `degrees`.
    Positive values are usually counterclockwise in pypdf.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for page in reader.pages:
        try:
            page = page.rotate(degrees)
        except Exception:
            try:
                page.rotate_clockwise(degrees)
            except Exception:
                pass
        writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()


# ---------- PQ helpers for Option 1 ----------
def extract_pq_value(zpl: str) -> int:
    """
    Extract PQ quantity from ^PQ in ZPL (first integer).
    Example: ^PQ1200 or ^PQ1200,0,1,Y -> 1200
    """
    m = re.search(r"\^PQ\s*(\d+)", zpl, flags=re.IGNORECASE)
    if not m:
        return 1
    try:
        value = int(m.group(1))
        return max(value, 1)
    except ValueError:
        return 1


def override_pq_to_one(zpl: str) -> str:
    """
    Return a ZPL version where PQ is forced to 1 (for LabelZoom rendering).
    Keeps additional parameters (^PQ1,0,1,Y etc.).
    """
    def repl(match: re.Match) -> str:
        rest = match.group(2) or ""
        return f"^PQ1{rest}"

    if re.search(r"\^PQ", zpl, flags=re.IGNORECASE):
        return re.sub(
            r"\^PQ\s*(\d+)([^\^]*)",
            repl,
            zpl,
            count=1,
            flags=re.IGNORECASE,
        )

    m = re.search(r"\^XA", zpl, flags=re.IGNORECASE)
    if m:
        idx = m.end()
        return zpl[:idx] + "^PQ1" + zpl[idx:]
    else:
        return "^XA^PQ1" + zpl


def replicate_single_page_pdf(single_page_pdf: bytes, copies: int) -> bytes:
    """Duplicate a 1-page PDF `copies` times. If copies <= 1, return original."""
    if copies <= 1:
        return single_page_pdf

    reader = PdfReader(io.BytesIO(single_page_pdf))
    if len(reader.pages) == 0:
        return single_page_pdf

    writer = PdfWriter()
    page = reader.pages[0]
    for _ in range(copies):
        writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()


def extract_hangtag_model_code(zpl: str) -> str | None:
    """
    Extract the hangtag 'code row' used for naming Option 1 PDFs.

    We:
      - Scan every ^FD ... ^FS block
      - Look for pattern:
          <LETTER> 12345 1234 1234 [optional LETTER]
        e.g. "C 40008 0012 0002 U", "C 50039 0020 0002 U", "S 50019 0023 0002"
    """
    fd_fields = re.findall(r"\^FD(.*?)\^FS", zpl, flags=re.IGNORECASE | re.DOTALL)
    for field in fd_fields:
        m = re.search(
            r"[A-Z]\s*\d{5}\s*\d{4}\s*\d{4}(?:\s*[A-Z])?",
            field,
            flags=re.IGNORECASE,
        )
        if m:
            code = m.group(0)
            code = re.sub(r"\s+", " ", code).strip()
            code = code[0].upper() + code[1:]
            return code
    return None


# ---------- Carton helpers for Option 2 ----------
def extract_carton_numbers(zpl: str):
    """
    Extract carton sequence numbers from ^FD ... ^FS with 10 digits,
    allowing an optional >: prefix (e.g. ^FD>:1044217560^FS).
    """
    matches = re.findall(
        r"\^FD\s*(?:>:\s*)?(\d{10})\s*\^FS",
        zpl,
        flags=re.IGNORECASE,
    )
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def group_sequences_from_codes(codes):
    """Group numeric string codes into contiguous sequences for one file."""
    if not codes:
        return []
    unique_ints = sorted({int(c) for c in codes})
    sequences = []
    start = prev = unique_ints[0]
    for v in unique_ints[1:]:
        if v == prev + 1:
            prev = v
        else:
            sequences.append((start, prev))
            start = prev = v
    sequences.append((start, prev))
    return sequences


def extract_produto_code(zpl: str) -> str | None:
    """
    Robustly extract the 'Produto' code like: C 50039 0020 0002 or S 50019 0023 0002.
    """
    fd_fields = re.findall(r"\^FD(.*?)\^FS", zpl, flags=re.IGNORECASE | re.DOTALL)
    for field in fd_fields:
        m = re.search(r"[A-Z]\s*\d{5}\s*\d{4}\s*\d{4}", field, flags=re.IGNORECASE)
        if m:
            code = m.group(0)
            code = re.sub(r"\s+", " ", code).strip()
            code = code[0].upper() + code[1:]
            return code
    return None


# ---------- Reserva helpers for Option 3 ----------
def extract_reserva_code(zpl: str) -> str | None:
    """
    Extract Reserva product code for naming Option 3 PDFs.

    Example:
      ^FDR4602200040001^FS  -> code: R4602200040001

    We capture 'R' + 13 digits. If that fails, fall back to any 13-digit code.
    """
    fd_fields = re.findall(r"\^FD(.*?)\^FS", zpl, flags=re.IGNORECASE | re.DOTALL)
    for field in fd_fields:
        m = re.search(r"R\d{13}", field, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    # fallback: plain 13-digit sequence (likely an EAN)
    for field in fd_fields:
        m = re.search(r"\d{13}", field)
        if m:
            return m.group(0)
    return None


# ---------- UI ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    option = st.radio(
        "Choose what you want to do:",
        [
            "Option 1: Product Hangtag (Arezzo-style: standardize with OpenAI, then PDF with PQ duplication, merged per SKU)",
            "Option 2: Carton Barcodes (10×4 cm @ 203 DPI → PDF + per-SKU sequence)",
            "Option 3: Reserva Hangtag Barcode (8×3.5 cm @ 203 DPI → rotated PDF, merged per SKU)",
        ],
        index=0,
    )

    is_option1 = option.startswith("Option 1")
    is_option2 = option.startswith("Option 2")
    is_option3 = option.startswith("Option 3")

    uploaded_files = st.file_uploader(
        "Upload your .prn files (or .zpl/.txt with ZPL):",
        type=["prn", "zpl", "txt"],
        accept_multiple_files=True,
        help=(
            "You can drag & drop multiple files at once.\n"
            "If a file has many ^XA…^XZ blocks, each label will be split.\n"
            "- Option 1: all labels with the same hangtag code → 1 merged PDF.\n"
            "- Option 2: all carton labels with the same Produto code → 1 merged PDF.\n"
            "- Option 3: all labels with the same Reserva code → 1 merged PDF."
        ),
    )

    model_name = None
    if is_option1:
        model_name = st.selectbox(
            "OpenAI model for standardization (Option 1 only):",
            ["gpt-5.1-mini", "gpt-5.1"],
            index=0,
            help="Cheaper model first; use the larger one if needed.",
        )

with col_right:
    st.subheader("Status")
    st.markdown(
        """
**Multi-label .prn files**

- Any file with multiple `^XA ... ^XZ` blocks is split into individual labels.

**Option 1 – Product Hangtag (Arezzo)**  
- 5.5 × 2.5 cm @ 203 DPI  
- ^PW440 ^LL200 ^LH0,0, ^FO30→^FO40, ^FO330→^FO340  
- OpenAI cleans and standardizes layout  
- LabelZoom renders **PQ=1**, then Python duplicates the page to match original **PQ** per label  
- All labels with the same hangtag code are **merged into ONE PDF**  
- Filenames: `HANGTAG BARCODE - C 50039 0020 0002 U.pdf`

**Option 2 – Carton Barcodes**  
- 10 × 4 cm @ 203 DPI  
- Each label becomes a 1-page barcode  
- All labels with the same Produto code are **merged into ONE PDF**  
- Shows carton number sequences **per merged PDF**  
- Filenames: `CARTON BARCODE C 50039 0020 0002.pdf`

**Option 3 – Reserva Hangtag Barcode**  
- 8 × 3.5 cm @ 203 DPI  
- Each label converted with LabelZoom, rotated -90° (vertical, upright)  
- All labels with the same Reserva code (SKU) are **merged into ONE PDF**  
- Filenames: `RESERVA HANGTAG BARCODE - R4602200040001.pdf`
        """
    )


process_clicked = st.button("Process files", type="primary", disabled=not uploaded_files)

if process_clicked and uploaded_files:
    if is_option1 and not OPENAI_API_KEY:
        st.error(
            "OpenAI API key not configured. Set `openai_api_key` in Streamlit secrets or `OPENAI_API_KEY` env var."
        )
    elif not LABELZOOM_API_KEY:
        st.error(
            "LabelZoom API key not configured. Set `labelzoom_api_key` in Streamlit secrets or `LABELZOOM_API_KEY` env var."
        )
    else:
        # Pre-split all files into segments to have a correct total count
        file_segments = []
        total_segments = 0
        for uploaded in uploaded_files:
            raw_zpl = read_prn_file(uploaded)
            segments = split_zpl_into_formats(raw_zpl)
            if segments:
                file_segments.append((uploaded, segments))
                total_segments += len(segments)

        if total_segments == 0:
            st.error("No ^XA...^XZ blocks found in the uploaded files.")
        else:
            # Temporary segment storage for grouping
            option1_segments = []
            option2_segments = []
            option3_segments = []

            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            processed_count = 0

            for uploaded, segments in file_segments:
                for seg_idx, segment_zpl in enumerate(segments, start=1):
                    processed_count += 1
                    status_placeholder.info(
                        f"Processing {uploaded.name} – label {seg_idx}/{len(segments)} "
                        f"({processed_count}/{total_segments} total)"
                    )

                    try:
                        if is_option1:
                            # ---------- OPTION 1: Arezzo hangtags (per label first) ----------
                            pq_val = extract_pq_value(segment_zpl)

                            standardized = standardize_zpl_with_openai(
                                segment_zpl, model=model_name
                            )
                            final_zpl = enforce_hangtag_standard(standardized)

                            zpl_for_render = override_pq_to_one(final_zpl)
                            single_pdf_bytes = convert_zpl_to_pdf_with_labelzoom(
                                zpl_for_render
                            )

                            pdf_bytes = replicate_single_page_pdf(
                                single_pdf_bytes, pq_val
                            )

                            code_row = extract_hangtag_model_code(final_zpl)

                            option1_segments.append(
                                {
                                    "group_key": code_row,  # can be None, handled later
                                    "original_name": uploaded.name,
                                    "segment_index": seg_idx,
                                    "segment_total": len(segments),
                                    "pdf_bytes": pdf_bytes,
                                    "zpl": final_zpl,
                                    "pq": pq_val,
                                }
                            )

                        elif is_option2:
                            # ---------- OPTION 2: Carton barcodes (per label first) ----------
                            final_zpl = apply_manual_label_size(
                                segment_zpl,
                                width_cm=10.0,
                                height_cm=4.0,
                                dpi=203,
                            )
                            carton_numbers = extract_carton_numbers(final_zpl)
                            produto_code = extract_produto_code(final_zpl)

                            pdf_bytes = convert_zpl_to_pdf_with_labelzoom(final_zpl)

                            option2_segments.append(
                                {
                                    "group_key": produto_code,  # can be None
                                    "original_name": uploaded.name,
                                    "segment_index": seg_idx,
                                    "segment_total": len(segments),
                                    "pdf_bytes": pdf_bytes,
                                    "zpl": final_zpl,
                                    "carton_numbers": carton_numbers,
                                }
                            )

                        elif is_option3:
                            # ---------- OPTION 3: Reserva hangtags (per label first) ----------
                            final_zpl = apply_manual_label_size(
                                segment_zpl,
                                width_cm=8.0,
                                height_cm=3.5,
                                dpi=203,
                            )
                            reserva_code = extract_reserva_code(final_zpl)

                            raw_pdf = convert_zpl_to_pdf_with_labelzoom(final_zpl)
                            # Rotate -90 so it’s vertical and upright
                            pdf_bytes = rotate_pdf(raw_pdf, degrees=-90)

                            option3_segments.append(
                                {
                                    "group_key": reserva_code,  # can be None
                                    "original_name": uploaded.name,
                                    "segment_index": seg_idx,
                                    "segment_total": len(segments),
                                    "pdf_bytes": pdf_bytes,
                                    "zpl": final_zpl,
                                }
                            )

                    except Exception as e:
                        st.error(
                            f"Error processing {uploaded.name} – label {seg_idx}/{len(segments)}: {e}"
                        )

                    progress_bar.progress(processed_count / total_segments)

            status_placeholder.empty()

            # ---------- Build final results (grouped per SKU/code) ----------
            results = []

            # Option 1: group by hangtag model code
            if is_option1 and option1_segments:
                grouped = {}
                for e in option1_segments:
                    key = e["group_key"]
                    if not key:
                        key = f"{e['original_name']}_part{e['segment_index']}"
                    grouped.setdefault(key, []).append(e)

                for code, entries in grouped.items():
                    writer = PdfWriter()
                    pq_total = 0
                    for e in entries:
                        reader = PdfReader(io.BytesIO(e["pdf_bytes"]))
                        for page in reader.pages:
                            writer.add_page(page)
                        pq_total += e.get("pq") or 0
                    out = io.BytesIO()
                    writer.write(out)
                    out.seek(0)
                    merged_pdf = out.read()

                    pdf_name = f"HANGTAG BARCODE - {code}.pdf"
                    original_names = sorted({e["original_name"] for e in entries})
                    example_zpl = entries[0]["zpl"]

                    results.append(
                        {
                            "pdf_name": pdf_name,
                            "pdf_bytes": merged_pdf,
                            "original_name": ", ".join(original_names),
                            "zpl": example_zpl,
                            "pq_total": pq_total,
                            "labels_count": len(entries),
                            "code": code,
                            "tipo": "option1",
                        }
                    )

            # Option 2: group by Produto code, aggregate carton numbers
            if is_option2 and option2_segments:
                grouped = {}
                for e in option2_segments:
                    key = e["group_key"]
                    if not key:
                        key = f"{e['original_name']}_part{e['segment_index']}"
                    grouped.setdefault(key, []).append(e)

                for produto_code, entries in grouped.items():
                    writer = PdfWriter()
                    all_carton_numbers = []
                    for e in entries:
                        reader = PdfReader(io.BytesIO(e["pdf_bytes"]))
                        for page in reader.pages:
                            writer.add_page(page)
                        all_carton_numbers.extend(e.get("carton_numbers", []))
                    out = io.BytesIO()
                    writer.write(out)
                    out.seek(0)
                    merged_pdf = out.read()

                    pdf_name = f"CARTON BARCODE {produto_code}.pdf"
                    original_names = sorted({e["original_name"] for e in entries})
                    example_zpl = entries[0]["zpl"]

                    results.append(
                        {
                            "pdf_name": pdf_name,
                            "pdf_bytes": merged_pdf,
                            "original_name": ", ".join(original_names),
                            "zpl": example_zpl,
                            "carton_numbers": all_carton_numbers,
                            "labels_count": len(entries),
                            "code": produto_code,
                            "tipo": "option2",
                        }
                    )

            # Option 3: group by Reserva code
            if is_option3 and option3_segments:
                grouped = {}
                for e in option3_segments:
                    key = e["group_key"]
                    if not key:
                        key = f"{e['original_name']}_part{e['segment_index']}"
                    grouped.setdefault(key, []).append(e)

                for reserva_code, entries in grouped.items():
                    writer = PdfWriter()
                    for e in entries:
                        reader = PdfReader(io.BytesIO(e["pdf_bytes"]))
                        for page in reader.pages:
                            writer.add_page(page)
                    out = io.BytesIO()
                    writer.write(out)
                    out.seek(0)
                    merged_pdf = out.read()

                    pdf_name = f"RESERVA HANGTAG BARCODE - {reserva_code}.pdf"
                    original_names = sorted({e["original_name"] for e in entries})
                    example_zpl = entries[0]["zpl"]

                    results.append(
                        {
                            "pdf_name": pdf_name,
                            "pdf_bytes": merged_pdf,
                            "original_name": ", ".join(original_names),
                            "zpl": example_zpl,
                            "labels_count": len(entries),
                            "code": reserva_code,
                            "tipo": "option3",
                        }
                    )

            # ---------- Show results ----------
            if results:
                st.success(
                    f"Done! Generated {len(results)} PDF file(s) across {len(uploaded_files)} uploaded file(s)."
                )

                for i, item in enumerate(results):
                    with st.expander(f"📄 {item['pdf_name']} ({item['original_name']})"):
                        st.download_button(
                            "⬇️ Download PDF",
                            data=item["pdf_bytes"],
                            file_name=item["pdf_name"],
                            mime="application/pdf",
                            key=f"download_pdf_{i}",
                        )
                        st.text_area(
                            "Example ZPL sent to LabelZoom:",
                            value=item["zpl"],
                            height=260,
                            key=f"zpl_text_{i}",
                        )

                        if is_option1 and item.get("tipo") == "option1":
                            st.markdown(
                                f"- **Hangtag code:** `{item['code']}`\n"
                                f"- **Labels merged:** `{item['labels_count']}`\n"
                                f"- **Total PQ across all labels:** `{item['pq_total']}` pages"
                            )

                        if is_option2 and item.get("tipo") == "option2":
                            st.markdown(
                                f"- **Produto code:** `{item['code']}`\n"
                                f"- **Labels merged:** `{item['labels_count']}`"
                            )
                            carton_numbers = item.get("carton_numbers", [])
                            if carton_numbers:
                                st.markdown(
                                    "**Carton number(s) across this SKU (deduped):**"
                                )
                                unique_codes = sorted({int(c) for c in carton_numbers})
                                st.code(", ".join(f"{c:010d}" for c in unique_codes))

                                sequences = group_sequences_from_codes(
                                    [str(c) for c in unique_codes]
                                )
                                st.markdown("**Carton sequence(s) for this SKU:**")
                                for j, (start_int, end_int) in enumerate(
                                    sequences, start=1
                                ):
                                    if start_int == end_int:
                                        label = f"{start_int:010d}"
                                    else:
                                        label = f"{start_int:010d} - {end_int:010d}"
                                    st.markdown(f"- Sequence {j}: {label}")
                            else:
                                st.markdown("_No carton numbers detected for this SKU._")

                        if is_option3 and item.get("tipo") == "option3":
                            st.markdown(
                                f"- **Reserva code (SKU):** `{item['code']}`\n"
                                f"- **Labels merged in this PDF:** `{item['labels_count']}`"
                            )

                # Download all PDFs as one ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for item in results:
                        zf.writestr(item["pdf_name"], item["pdf_bytes"])
                zip_bytes = zip_buffer.getvalue()

                st.download_button(
                    "⬇️ Download all PDFs (ZIP)",
                    data=zip_bytes,
                    file_name="labels.zip",
                    mime="application/zip",
                    key="download_zip_all_pdfs",
                )
