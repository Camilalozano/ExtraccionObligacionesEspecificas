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


def extract_contractor_document(text: str, contractor_name: str = "") -> str:
    """
    Extrae el número de documento del contratista con una estrategia más robusta:
    1) Busca cerca del nombre del contratista.
    2) Busca en el bloque 'por la otra'.
    3) Busca patrones generales de cédula/CC.
    """

    def clean_cedula(candidate: str) -> str:
        # Corrige posibles errores OCR comunes
        candidate = candidate.translate(str.maketrans({
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1"
        }))
        # Deja solo dígitos
        return re.sub(r"[^0-9]", "", candidate)

    def normalize_for_search(s: str) -> str:
        s = s.replace("\xa0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    text_norm = normalize_for_search(text)

    # Patrones flexibles para capturar números tipo 79.741.213 / 79741213 / 79 741 213
    num_pattern = r"([0-9OIl][0-9OIl\.\,\-\s]{5,}[0-9OIl])"

    id_patterns = [
        rf"c[ée]dula\s+de\s+ciudadan[íi]a\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
        rf"\bC\.?\s*C\.?\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
        rf"\bc[ée]dula\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
        rf"identificad[oa]\s*(?:\(a\))?\s+con\s+(?:la\s+)?c[ée]dula\s+de\s+ciudadan[íi]a\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
    ]

    # 1) Buscar cerca del nombre del contratista
    if contractor_name:
        contractor_name_esc = re.escape(normalize_for_search(contractor_name))
        m_name = re.search(contractor_name_esc, text_norm, re.IGNORECASE)
        if m_name:
            start = max(0, m_name.start() - 80)
            end = min(len(text_norm), m_name.end() + 250)
            window = text_norm[start:end]

            for pattern in id_patterns:
                m = re.search(pattern, window, re.IGNORECASE)
                if m:
                    return clean_cedula(m.group(1))

    # 2) Buscar en el bloque contextual de "por la otra"
    m_other = re.search(
        r"por\s+la\s+otra,?(.*?)(?:actuando\s+en\s+nombre\s+propio|qu[ií]en\s+declara|EL\s+CONTRATISTA)",
        text_norm,
        re.IGNORECASE | re.DOTALL
    )
    if m_other:
        block = m_other.group(1)
        for pattern in id_patterns:
            m = re.search(pattern, block, re.IGNORECASE)
            if m:
                return clean_cedula(m.group(1))

    # 3) Buscar en todo el texto y tomar la última coincidencia razonable
    found = []
    for pattern in id_patterns:
        for m in re.finditer(pattern, text_norm, re.IGNORECASE):
            value = clean_cedula(m.group(1))
            if 6 <= len(value) <= 12:
                found.append(value)

    if found:
        return found[-1]

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
