import io
import os
import zipfile
import re
from pathlib import Path

import requests
import streamlit as st
from openai import OpenAI


# ---------- Config ----------
st.set_page_config(
    page_title="ZPL Label Helper",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ ZPL Label Helper")
st.caption(
    "Option 1: Standardize hangtag ZPL with OpenAI → LabelZoom PDF • "
    "Option 2: Carton barcodes → LabelZoom PDF (10×4 cm @ 203 DPI) + per-file sequence check"
)


# ---------- Secrets / clients ----------
OPENAI_API_KEY = st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
LABELZOOM_API_KEY = st.secrets.get("labelzoom_api_key", os.getenv("LABELZOOM_API_KEY"))

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

LABELZOOM_ZPL_TO_PDF_URL = "https://prod-api.labelzoom.net/api/v2/convert/zpl/to/pdf"

# 🔧 Put your real template + rules here for hangtags
STANDARDIZATION_SYSTEM_PROMPT = """
You are a ZPL label expert.

Task:
- Receive raw ZPL code from a `.prn` file for a PRODUCT HANGTAG.
- Transform it so it matches our canonical template and layout rules.
- Keep *all* dynamic data (texts, barcodes, EANs, SKUs) identical.
- Only adjust:
  - Field positions (^FO / ^FT),
  - Fonts (^A*),
  - Static text,
  - Label size commands (^PW, ^LL, etc.)
  to follow the template.

Output rules:
- Return **only** valid ZPL, without explanations.
- Do NOT wrap the result in backticks or Markdown fences.
- Ensure the code starts with ^XA and ends with ^XZ.

[IMPORTANT]
Replace this text block with your real canonical ZPL template and detailed rules.
"""


# ---------- Helpers ----------
def read_prn_file(uploaded_file) -> str:
    """Read a .prn file as text, trying UTF-8 then Latin-1."""
    content = uploaded_file.read()
    if isinstance(content, str):
        return content
    for encoding in ("utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except Exception:
            continue
    # Fallback: decode ignoring errors
    return content.decode("latin-1", errors="ignore")


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


def apply_manual_label_size(
    zpl: str,
    width_cm: float = 10.0,
    height_cm: float = 4.0,
    dpi: int = 203,
) -> str:
    """
    Force the ZPL label to a fixed physical size by setting ^PW (width)
    and ^LL (length) in dots.

    For 203 DPI, we use ~8 dpmm (8 dots/mm).
    10 cm  -> 100 mm -> 100 * 8 = 800 dots
    4 cm   ->  40 mm ->  40 * 8 = 320 dots
    """
    dpmm = int(round(dpi / 25.4))  # 203 -> ~8

    width_mm = width_cm * 10.0
    height_mm = height_cm * 10.0

    width_dots = int(round(width_mm * dpmm))
    height_dots = int(round(height_mm * dpmm))

    # Remove existing ^PW / ^LL to avoid conflicts
    cleaned = re.sub(r"\^PW-?\d+", "", zpl, flags=re.IGNORECASE)
    cleaned = re.sub(r"\^LL-?\d+", "", cleaned, flags=re.IGNORECASE)

    # Insert ^PW / ^LL right after the first ^XA
    match = re.search(r"\^XA", cleaned, flags=re.IGNORECASE)
    size_cmds = f"^PW{width_dots}^LL{height_dots}"

    if match:
        insert_at = match.end()
        return cleaned[:insert_at] + size_cmds + cleaned[insert_at:]
    else:
        # Best-effort fallback: just prepend size commands at the start
        return f"^XA{size_cmds}{cleaned}"


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


def extract_carton_numbers(zpl: str) -> list[str]:
    """
    Extract the carton sequence number from lines like:

    ^FO160,197^A0N,105,100^FR^FD1044217560^FS
    ^FO430,35^BY2^BCN,100^FD>:1044217560^FS

    We look specifically at ^FD ... ^FS segments that contain a 10-digit number,
    with optional >: prefix.
    """
    matches = re.findall(
        r"\^FD\s*(?:>:\s*)?(\d{10})\s*\^FS",
        zpl,
        flags=re.IGNORECASE,
    )
    # Deduplicate while preserving order within this file
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def group_sequences_from_codes(codes: list[str]) -> list[tuple[int, int]]:
    """
    Group a list of numeric string codes (e.g. '1044217560') into contiguous
    sequences for ONE FILE.

    Returns a list of (start_int, end_int).
    """
    if not codes:
        return []

    unique_ints = sorted({int(c) for c in codes})
    sequences: list[tuple[int, int]] = []

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
    Extract the 'Produto' code like: S 50019 0023 0002 (optionally followed by a letter).

    Example expected pattern in ZPL line:
    ^FD S 50019 0023 0002 U^FS

    We capture only "S 50019 0023 0002" (no trailing letter).
    """
    # Look for S + 5 digits + 4 digits + 4 digits, with optional variable spacing
    match = re.search(r"(S\s*\d{5}\s*\d{4}\s*\d{4})", zpl)
    if not match:
        return None
    code = match.group(1)
    # Normalize spaces
    code = re.sub(r"\s+", " ", code).strip()
    return code


# ---------- UI ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    option = st.radio(
        "Choose what you want to do:",
        [
            "Option 1: Product Hangtag (standardize ZPL with OpenAI, then PDF)",
            "Option 2: Carton Barcodes (10×4 cm @ 203 DPI → PDF + per-file sequence check)",
        ],
        index=0,
    )

    uploaded_files = st.file_uploader(
        "Upload your .prn files (or .zpl/.txt with ZPL):",
        type=["prn", "zpl", "txt"],
        accept_multiple_files=True,
        help="You can drag & drop multiple files at once.",
    )

    model_name = None
    if "Product Hangtag" in option:
        model_name = st.selectbox(
            "OpenAI model for standardization:",
            ["gpt-5.1-mini", "gpt-5.1"],
            index=0,
            help="Cheaper model first; swap to the larger one if you need more robustness.",
        )

with col_right:
    st.subheader("Status")
    st.markdown(
        """
- ✅ **Option 1**: `.prn` → OpenAI standardization → LabelZoom PDF  
- ✅ **Option 2**: `.prn` → LabelZoom PDF with **fixed 10×4 cm @ 203 DPI**  
- 🔍 Option 2 shows **carton sequences per file** based only on the ^FD lines with the 10-digit number  
- 📁 Option 2 filenames: `CARTON BARCODE S 50019 0023 0002.pdf` (Produto code from label)  
- 🔐 API keys loaded from **secrets** or **environment variables**:
  - `openai_api_key` / `OPENAI_API_KEY`
  - `labelzoom_api_key` / `LABELZOOM_API_KEY`
        """
    )


process_clicked = st.button("Process files", type="primary", disabled=not uploaded_files)

if process_clicked and uploaded_files:
    if "Product Hangtag" in option and not OPENAI_API_KEY:
        st.error("OpenAI API key not configured. Set `openai_api_key` in Streamlit secrets or `OPENAI_API_KEY` env var.")
    elif not LABELZOOM_API_KEY:
        st.error("LabelZoom API key not configured. Set `labelzoom_api_key` in Streamlit secrets or `LABELZOOM_API_KEY` env var.")
    else:
        results = []
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()

        for idx, uploaded in enumerate(uploaded_files, start=1):
            status_placeholder.info(f"Processing file {idx}/{len(uploaded_files)}: **{uploaded.name}**")

            raw_zpl = read_prn_file(uploaded)
            final_zpl = raw_zpl
            carton_numbers: list[str] = []
            produto_code = None

            try:
                if "Product Hangtag" in option:
                    # Option 1: standardize via OpenAI, template controls size
                    final_zpl = standardize_zpl_with_openai(raw_zpl, model=model_name)
                    base_name = Path(uploaded.name).stem
                    pdf_name = f"{base_name}.pdf"
                else:
                    # Option 2: enforce manual size 10×4 cm at 203 DPI
                    final_zpl = apply_manual_label_size(
                        raw_zpl,
                        width_cm=10.0,
                        height_cm=4.0,
                        dpi=203,
                    )
                    # Extract carton sequence number(s) and Produto code for THIS FILE
                    carton_numbers = extract_carton_numbers(final_zpl)
                    produto_code = extract_produto_code(final_zpl)

                    if produto_code:
                        pdf_name = f"CARTON BARCODE {produto_code}.pdf"
                    else:
                        base_name = Path(uploaded.name).stem
                        pdf_name = f"CARTON BARCODE {base_name}.pdf"

                pdf_bytes = convert_zpl_to_pdf_with_labelzoom(final_zpl)

                results.append(
                    {
                        "original_name": uploaded.name,
                        "pdf_name": pdf_name,
                        "pdf_bytes": pdf_bytes,
                        "zpl": final_zpl,
                        "carton_numbers": carton_numbers,
                        "produto_code": produto_code,
                    }
                )

            except Exception as e:
                st.error(f"Error processing {uploaded.name}: {e}")

            progress_bar.progress(idx / len(uploaded_files))

        status_placeholder.empty()

        if results:
            st.success(f"Done! Processed {len(results)} file(s).")

            # Per-file display (including per-file sequences for Option 2)
            for item in results:
                with st.expander(f"📄 {item['pdf_name']} ({item['original_name']})"):
                    st.download_button(
                        "⬇️ Download PDF",
                        data=item["pdf_bytes"],
                        file_name=item["pdf_name"],
                        mime="application/pdf",
                    )
                    st.text_area(
                        "Final ZPL sent to LabelZoom:",
                        value=item["zpl"],
                        height=260,
                    )

                    if "Carton Barcodes" in option:
                        if item["produto_code"]:
                            st.markdown(f"**Produto code detected:** `{item['produto_code']}`")

                        if item["carton_numbers"]:
                            st.markdown("**Carton number(s) found in this file (from ^FD lines):**")
                            st.code(", ".join(item["carton_numbers"]))

                            # 🔹 Per-file sequences here
                            sequences = group_sequences_from_codes(item["carton_numbers"])
                            st.markdown("**Carton sequence(s) for this file:**")
                            for i, (start_int, end_int) in enumerate(sequences, start=1):
                                if start_int == end_int:
                                    label = f"{start_int:010d}"
                                else:
                                    label = f"{start_int:010d} - {end_int:010d}"
                                st.markdown(f"- Sequence {i}: {label}")
                        else:
                            st.markdown("_No carton numbers detected in this file._")

            # ---- Download all PDFs as one ZIP (both options) ----
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in results:
                    zf.writestr(item["pdf_name"], item["pdf_bytes"])
            zip_buffer.seek(0)

            st.download_button(
                "⬇️ Download all PDFs (ZIP)",
                data=zip_buffer,
                file_name="labels.zip",
                mime="application/zip",
            )
