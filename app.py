import io
import os
import zipfile
import re
from pathlib import Path

import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader, PdfWriter  # for duplicating pages and rotating


# ---------- Config ----------
st.set_page_config(
    page_title="ZPL Label Helper",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ ZPL Label Helper")
st.caption(
    "Option 1: Product Hangtag (Arezzo-style: OpenAI standardization → LabelZoom PDF with PQ duplication) • "
    "Option 2: Carton barcodes (10×4 cm @ 203 DPI → PDF + per-file sequence) • "
    "Option 3: Reserva hangtag barcode (8×3.5 cm @ 203 DPI → rotated PDF)"
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
            # fallback for older pypdf versions
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

    We don't rely on specific coordinates anymore.
    Instead we:
      - Scan every ^FD ... ^FS block
      - Look for any pattern like:
          <LETTER> 12345 1234 1234 [optional LETTER]
        e.g. "C 40008 0012 0002 U", "C 50039 0020 0002 U", "S 50019 0023 0002"

    We return the normalized code, e.g. "C 40008 0012 0002 U".
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
            code = code[0].upper() + code[1:]  # ensure first letter uppercase
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

    It will:
      - Scan every ^FD ... ^FS block
      - Look for any single LETTER followed by 5-4-4 digits (with flexible spacing)
        e.g.: C 50039 0020 0002, S 50019 0023 0002
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

    Example from your sample:
      ^FO495,030^ADR,20,10^FH^FDR4602200040001^FS

    We capture: R4602200040001  (R + 13 digits).
    """
    fd_fields = re.findall(r"\^FD(.*?)\^FS", zpl, flags=re.IGNORECASE | re.DOTALL)
    for field in fd_fields:
        m = re.search(r"R\d{13}", field, flags=re.IGNORECASE)
        if m:
            code = m.group(0).upper()
            return code
    # fallback: just capture a 13-digit EAN if no R-code found
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
            "Option 1: Product Hangtag (Arezzo-style: standardize with OpenAI, then PDF with PQ duplication)",
            "Option 2: Carton Barcodes (10×4 cm @ 203 DPI → PDF + per-file sequence)",
            "Option 3: Reserva Hangtag Barcode (8×3.5 cm @ 203 DPI → rotated PDF)",
        ],
        index=0,
    )

    uploaded_files = st.file_uploader(
        "Upload your .prn files (or .zpl/.txt with ZPL):",
        type=["prn", "zpl", "txt"],
        accept_multiple_files=True,
        help="You can drag & drop multiple files at once.\n"
             "If a file has many ^XA…^XZ blocks, each label will be split and processed separately.",
    )

    model_name = None
    if "Product Hangtag" in option:
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

- Any file with multiple `^XA ... ^XZ` blocks is automatically split into multiple labels.

**Option 1 – Product Hangtag (Arezzo)**  
- 5.5 × 2.5 cm @ 203 DPI  
- ^PW440 ^LL200 ^LH0,0, ^FO30→^FO40, ^FO330→^FO340  
- OpenAI cleans and standardizes layout  
- LabelZoom renders **PQ=1**, then Python duplicates the page to match original **PQ**  
- Filenames: `HANGTAG BARCODE - <code_row>.pdf`  
  (e.g. `HANGTAG BARCODE - C 40008 0012 0002 U.pdf`)

**Option 2 – Carton Barcodes**  
- 10 × 4 cm @ 203 DPI  
- Shows sequences from 10-digit ^FD lines per label  
- Filenames: `CARTON BARCODE <Produto code>.pdf`

**Option 3 – Reserva Hangtag Barcode**  
- 8 × 3.5 cm @ 203 DPI  
- Uses LabelZoom, then rotates the PDF by 90° so the sticker is vertical  
- Filenames: `RESERVA HANGTAG BARCODE - R4602200040001.pdf` (or similar)
        """
    )


process_clicked = st.button("Process files", type="primary", disabled=not uploaded_files)

if process_clicked and uploaded_files:
    if "Product Hangtag" in option and not OPENAI_API_KEY:
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
            results = []
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

                    final_zpl = segment_zpl
                    carton_numbers = []
                    produto_code = None
                    reserva_code = None
                    pq_value = None
                    pdf_bytes = b""

                    try:
                        if "Product Hangtag" in option:
                            # ----- OPTION 1: Standardize + PQ-based duplication (per label) -----
                            pq_value = extract_pq_value(segment_zpl)

                            standardized = standardize_zpl_with_openai(
                                segment_zpl, model=model_name
                            )
                            final_zpl = enforce_hangtag_standard(standardized)

                            # For LabelZoom: PQ=1 for rendering a single page
                            zpl_for_render = override_pq_to_one(final_zpl)
                            single_pdf_bytes = convert_zpl_to_pdf_with_labelzoom(
                                zpl_for_render
                            )

                            # Duplicate page PQ times (based on original PQ)
                            pdf_bytes = replicate_single_page_pdf(
                                single_pdf_bytes, pq_value
                            )

                            code_row = extract_hangtag_model_code(final_zpl)
                            if code_row:
                                pdf_name = f"HANGTAG BARCODE - {code_row}.pdf"
                            else:
                                base_name = Path(uploaded.name).stem
                                pdf_name = f"{base_name}_part{seg_idx}.pdf"

                        elif "Carton Barcodes" in option:
                            # ----- OPTION 2: Carton barcodes (per label) -----
                            final_zpl = apply_manual_label_size(
                                segment_zpl,
                                width_cm=10.0,
                                height_cm=4.0,
                                dpi=203,
                            )
                            carton_numbers = extract_carton_numbers(final_zpl)
                            produto_code = extract_produto_code(final_zpl)

                            if produto_code:
                                pdf_name = f"CARTON BARCODE {produto_code}.pdf"
                            else:
                                base_name = Path(uploaded.name).stem
                                pdf_name = f"CARTON BARCODE {base_name}_part{seg_idx}.pdf"

                            pdf_bytes = convert_zpl_to_pdf_with_labelzoom(final_zpl)

                        else:
                            # ----- OPTION 3: Reserva hangtag barcode (per label) -----
                            # Set label size: 8 cm width × 3.5 cm height @ 203 DPI
                            final_zpl = apply_manual_label_size(
                                segment_zpl,
                                width_cm=8.0,
                                height_cm=3.5,
                                dpi=203,
                            )
                            reserva_code = extract_reserva_code(final_zpl)

                            if reserva_code:
                                pdf_name = f"RESERVA HANGTAG BARCODE - {reserva_code}.pdf"
                            else:
                                base_name = Path(uploaded.name).stem
                                pdf_name = f"RESERVA HANGTAG BARCODE - {base_name}_part{seg_idx}.pdf"

                            # Convert to PDF
                            raw_pdf = convert_zpl_to_pdf_with_labelzoom(final_zpl)
                            # Rotate PDF so label is vertical (90° rotation)
                            pdf_bytes = rotate_pdf(raw_pdf, degrees=90)

                        results.append(
                            {
                                "original_name": uploaded.name,
                                "segment_index": seg_idx,
                                "segment_total": len(segments),
                                "pdf_name": pdf_name,
                                "pdf_bytes": pdf_bytes,
                                "zpl": final_zpl,
                                "carton_numbers": carton_numbers,
                                "produto_code": produto_code,
                                "reserva_code": reserva_code,
                                "pq": pq_value,
                            }
                        )

                    except Exception as e:
                        st.error(
                            f"Error processing {uploaded.name} – label {seg_idx}/{len(segments)}: {e}"
                        )

                    progress_bar.progress(processed_count / total_segments)

            status_placeholder.empty()

            if results:
                st.success(
                    f"Done! Processed {len(results)} label(s) across {len(uploaded_files)} file(s)."
                )

                # Per-label display
                for i, item in enumerate(results):
                    original_label = item["original_name"]
                    if item.get("segment_total", 1) > 1:
                        original_label += f" – part {item['segment_index']}/{item['segment_total']}"

                    with st.expander(f"📄 {item['pdf_name']} ({original_label})"):
                        st.download_button(
                            "⬇️ Download PDF",
                            data=item["pdf_bytes"],
                            file_name=item["pdf_name"],
                            mime="application/pdf",
                            key=f"download_pdf_{i}",
                        )
                        st.text_area(
                            "Final ZPL sent to LabelZoom:",
                            value=item["zpl"],
                            height=260,
                            key=f"zpl_text_{i}",
                        )

                        # Extra info for Option 1
                        if "Product Hangtag" in option:
                            if item["pq"] is not None:
                                st.markdown(
                                    f"**PQ detected in original ZPL for this label:** `{item['pq']}` "
                                    "label(s). PDF pages were duplicated to match this quantity."
                                )

                        # Extra info for Option 2
                        if "Carton Barcodes" in option:
                            if item["produto_code"]:
                                st.markdown(
                                    f"**Produto code detected:** `{item['produto_code']}`"
                                )

                            if item["carton_numbers"]:
                                st.markdown(
                                    "**Carton number(s) found in this label (from ^FD lines):**"
                                )
                                st.code(", ".join(item["carton_numbers"]))

                                sequences = group_sequences_from_codes(
                                    item["carton_numbers"]
                                )
                                st.markdown("**Carton sequence(s) for this label:**")
                                for j, (start_int, end_int) in enumerate(
                                    sequences, start=1
                                ):
                                    if start_int == end_int:
                                        label = f"{start_int:010d}"
                                    else:
                                        label = f"{start_int:010d} - {end_int:010d}"
                                    st.markdown(f"- Sequence {j}: {label}")
                            else:
                                st.markdown("_No carton numbers detected in this label._")

                        # Extra info for Option 3
                        if "Reserva Hangtag" in option:
                            if item.get("reserva_code"):
                                st.markdown(
                                    f"**Reserva code detected:** `{item['reserva_code']}`"
                                )

                # Download all PDFs as one ZIP (fixed)
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
