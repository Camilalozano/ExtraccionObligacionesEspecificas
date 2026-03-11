import io
import json
import re
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from pypdf import PdfReader


# =========================
# Utilidades de extracción
# =========================

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extrae todo el texto de un PDF usando pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    """Normaliza espacios para facilitar búsquedas con regex, sin cambiar el contenido semántico."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def search_first(patterns: List[str], text: str, flags: int = re.IGNORECASE | re.DOTALL) -> Optional[re.Match]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match
    return None


# =========================
# Reglas de extracción
# =========================

def extract_contract_number(text: str, filename: str = "") -> str:
    patterns = [
        r"CONTRATO\s+DE\s+PRESTACI[ÓO]N\s+DE\s+SERVICIOS(?:\s+PROFESIONALES)?\s+No\.??\s*([A-ZÁÉÍÓÚ0-9\-_/]+)",
        r"CONTRATO\s+No\.??\s*([A-ZÁÉÍÓÚ0-9\-_/]+)",
        r"No\.??\s*(ATENEA\-[0-9]+\-[0-9]{4})",
        r"(ATENEA\-[0-9]+\-[0-9]{4})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,:;\n\t")

    m_file = re.search(r"(ATENEA\-[0-9]+\-[0-9]{4})", filename, re.IGNORECASE)
    if m_file:
        return m_file.group(1)
    return ""


def extract_contractor_name(text: str) -> str:
    patterns = [
        r"y por la otra,\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?),\s*mayor de edad, identificad[oa]",
        r"celebrado entre .*? y\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?)\s+DIANA CONSUELO BLANCO GARZ[ÓO]N",
        r"y\s+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?),\s*mayor de edad",
        r"quien en adelante se denominar[áa]\s+EL CONTRATISTA.*?por la otra,\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?),",
    ]
    m = search_first(patterns, text)
    if m:
        contractor_name = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:\n\t")
        contractor_name = re.sub(r"^la\s+tecnolog[íi]a\s+y\s+", "", contractor_name, flags=re.IGNORECASE)
        return contractor_name
    return ""


def extract_contractor_document(text: str) -> str:
    cedula_token = r"(?:No\.?|No,?|N[°º]\.?|Nro\.?|Número|Num\.?|#)?"
    cedula_number = r"([0-9OIl][0-9OIl\.,/\-\s]{4,}[0-9OIl])"

    def clean_cedula(candidate: str) -> str:
        normalized = candidate.translate(str.maketrans({"O": "0", "I": "1", "l": "1"}))
        normalized = re.sub(r"[^0-9]", "", normalized)
        return normalized

    # 1) Prioridad: cédula ubicada después de "por la otra / por el otro".
    m_context = re.search(
        rf"por\s+la\s+(?:otra|el\s+otro).*?identificad[oa]\s+con\s+la\s+c[ée]dula\s+de\s+ciudadan[íi]a\s*{cedula_token}\s*{cedula_number}",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m_context:
        return clean_cedula(m_context.group(1))

    # 2) Alternativa: capturar todas las cédulas encontradas y usar la última (suele ser la del contratista).
    matches = re.findall(
        rf"identificad[oa]\s+con\s+la\s+c[ée]dula\s+de\s+ciudadan[íi]a\s*{cedula_token}\s*{cedula_number}",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return clean_cedula(matches[-1])

    # 3) Fallback más laxo para formatos menos estructurados.
    fallback_patterns = [
        rf"EL\s+CONTRATISTA.*?c[ée]dula\s+de\s+ciudadan[íi]a\s*{cedula_token}\s*{cedula_number}",
        rf"c[ée]dula\s+de\s+ciudadan[íi]a\s*{cedula_token}\s*{cedula_number}\s+expedida",
    ]
    m = search_first(fallback_patterns, text)
    if m:
        return clean_cedula(m.group(1))

    # 4) Fallback contextual: valor más cercano al bloque del contratista.
    m_contratista = re.search(
        rf"EL\s+CONTRATISTA.*?(?:c[ée]dula\s+de\s+ciudadan[íi]a\s*{cedula_token}\s*{cedula_number})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m_contratista:
        return clean_cedula(m_contratista.group(1))
    return ""


def extract_obligaciones_especificas(text: str) -> str:
    """
    Extrae el bloque textual de obligaciones específicas sin reescribirlo.
    Toma el texto entre el encabezado de obligaciones específicas y la cláusula siguiente.
    """
    start_patterns = [
        r"B\)\s*OBLIGACIONES\s+ESPEC[ÍI]FICAS\s*:\s*A\s*EL\s+CONTRATISTA\s+le\s+corresponde\s+el\s+cumplimiento\s+de\s+las\s+siguientes\s+obligaciones\s*:",
        r"B\)\s*OBLIGACIONES\s+ESPEC[ÍI]FICAS\s*:",
        r"OBLIGACIONES\s+ESPEC[ÍI]FICAS\s*:",
    ]
    end_patterns = [
        r"CL[ÁA]USULA\s+TERCERA\s*:",
        r"C\)\s*OBLIGACIONES\s+DE\s+LA\s+CONTRATANTE",
        r"OBLIGACIONES\s+DE\s+LA\s+CONTRATANTE",
    ]

    start_match = search_first(start_patterns, text)
    if not start_match:
        return ""

    tail = text[start_match.end():]
    end_match = search_first(end_patterns, tail)
    if end_match:
        obligaciones = tail[: end_match.start()]
    else:
        obligaciones = tail

    return obligaciones.strip()


def process_single_pdf(pdf_bytes: bytes, filename: str) -> Dict:
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    text = normalize_text(raw_text)

    result = {
        "archivo": Path(filename).name,
        "numero_contrato": extract_contract_number(text, filename),
        "nombre_contratista": extract_contractor_name(text),
        "numero_documento_contratista": extract_contractor_document(text),
        "obligaciones_especificas": extract_obligaciones_especificas(text),
    }
    return result


def process_zip(zip_bytes: bytes) -> List[Dict]:
    results = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        pdf_files = [name for name in zf.namelist() if name.lower().endswith(".pdf")]
        for name in pdf_files:
            pdf_bytes = zf.read(name)
            try:
                results.append(process_single_pdf(pdf_bytes, name))
            except Exception as e:
                results.append(
                    {
                        "archivo": Path(name).name,
                        "numero_contrato": "",
                        "nombre_contratista": "",
                        "numero_documento_contratista": "",
                        "obligaciones_especificas": "",
                        "error": str(e),
                    }
                )
    return results


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Extractor de contratos desde ZIP", page_icon="📄", layout="wide")

st.title("📄 Extractor de contratos desde archivo .zip")
st.write(
    "Carga un archivo **.zip** que contenga contratos en **PDF**. "
    "La app extrae por cada documento: número del contrato, nombre del contratista, "
    "número de documento y el bloque textual de **obligaciones específicas**."
)

uploaded_zip = st.file_uploader("Sube el archivo .zip", type=["zip"])

if uploaded_zip is not None:
    zip_bytes = uploaded_zip.read()

    with st.spinner("Procesando contratos..."):
        data = process_zip(zip_bytes)

    st.success(f"Se procesaron {len(data)} archivo(s).")

    if data:
        df = pd.DataFrame(data)
        st.subheader("Vista previa")
        st.dataframe(df, use_container_width=True)

        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        st.subheader("Vista previa del JSON")
        st.code(json_str[:10000] + ("\n\n..." if len(json_str) > 10000 else ""), language="json")

        st.download_button(
            label="⬇️ Descargar JSON",
            data=json_str.encode("utf-8"),
            file_name="contratos_extraidos.json",
            mime="application/json",
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_bytes,
            file_name="contratos_extraidos.csv",
            mime="text/csv",
        )

with st.expander("⚙️ Requisitos y ejecución local"):
    st.markdown(
        """
```bash
pip install streamlit pypdf pandas
streamlit run app_contratos_streamlit.py
```
        """
    )

with st.expander("📝 Notas"):
    st.markdown(
        """
- La extracción depende del texto legible dentro del PDF. Si el PDF es una imagen escaneada, haría falta OCR.
- El campo **obligaciones_especificas** se captura como bloque textual entre el encabezado de obligaciones específicas y la siguiente cláusula.
- El código intenta ser flexible con pequeñas variaciones de formato entre contratos.
        """
    )
