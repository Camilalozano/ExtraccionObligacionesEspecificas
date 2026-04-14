import io
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# =========================
# Utilidades generales
# =========================
TARGET_FIELDS = [
    "numero_contrato",
    "Tipo_contrato",
    "nombre_contratista",
    "numero_documento_contratista",
    "obligaciones_especificas",
    "nombre_supervisor",
]


def safe_json_loads(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)



def normalize_nullable_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()



def only_digits(x: Optional[str]) -> str:
    if not x:
        return ""
    x = str(x).translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    return re.sub(r"\D", "", x)



def limpiar_texto_para_llm(text: str) -> str:
    if not text:
        return ""

    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u00A0", " ")
    t = t.replace("\u200B", "")
    t = t.replace("\u200E", "")
    t = t.replace("\u200F", "")

    cleaned = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ["\n", "\t"]:
            continue
        cleaned.append(ch)

    t = "".join(cleaned)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()



def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = limpiar_texto_para_llm(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()



def search_first(patterns: List[str], text: str, flags: int = re.IGNORECASE | re.DOTALL) -> Optional[re.Match]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match
    return None



def looks_like_person_name(value: str) -> bool:
    if not value:
        return False
    value = re.sub(r"\s+", " ", value).strip(" ,.;:\n\t")
    words = [w for w in value.split() if w]
    if len(words) < 2:
        return False
    upper_words = sum(1 for w in words if re.fullmatch(r"[A-ZÁÉÍÓÚÑ]+(?:[-'][A-ZÁÉÍÓÚÑ]+)?", w))
    return upper_words >= min(2, len(words))



def cut_text(text: str, limit: int = 15000) -> str:
    return text[:limit] if len(text) > limit else text


# =========================
# Extracción de texto PDF
# =========================
def extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            try:
                pages.append(page.get_text("text") or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        return ""



def extract_text_with_pypdf(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        return ""



def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = extract_text_with_pymupdf(pdf_bytes)
    if len(normalize_text(text)) >= 200:
        return text

    text_alt = extract_text_with_pypdf(pdf_bytes)
    if len(normalize_text(text_alt)) > len(normalize_text(text)):
        text = text_alt

    return text


# =========================
# Reglas de extracción
# =========================
def extract_contract_number(text: str, filename: str = "") -> str:
    patterns = [
        r"CONTRATO\s+DE\s+[A-ZÁÉÍÓÚÑ\s]+?\s+No\.?\s*([A-Z0-9\-_/]+)",
        r"CONTRATO\s+No\.?\s*([A-Z0-9\-_/]+)",
        r"No\.?\s*(ATENEA\s*[-–]\s*\d+\s*[-–]\s*\d{4})",
        r"(ATENEA\s*[-–]\s*\d+\s*[-–]\s*\d{4})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            value = re.sub(r"\s+", "", m.group(1)).replace("–", "-")
            return value.strip(" .,:;\n\t")

    m_file = re.search(r"(ATENEA[-_]\d+[-_]\d{4})", filename, re.IGNORECASE)
    if m_file:
        return m_file.group(1).replace("_", "-")
    return ""



def extract_contract_type(text: str) -> str:
    patterns = [
        r"CONTRATO\s+DE\s+([A-ZÁÉÍÓÚÑ\s]+?)\s+No\.?,",
        r"CONTRATO\s+DE\s+([A-ZÁÉÍÓÚÑ\s]+?)\s+No\.?",
        r"presente\s+contrato\s+de\s+([A-ZÁÉÍÓÚÑ\s]+?)(?:\s+el\s+cual|\s+que\s+se\s+regir[áa]|\s+de\s+conformidad)",
    ]
    m = search_first(patterns, text)
    if not m:
        return ""
    value = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:\n\t")
    return value.upper()



def extract_contractor_name(text: str) -> str:
    patterns = [
        r"y\s+por\s+la\s+otra,\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?),\s*mayor\s+de\s+edad,\s*identificad[oa]",
        r"AGENCIA\s+ATENEA\s+y\s+por\s+la\s+otra,\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?),\s*mayor\s+de\s+edad",
        r"celebrado\s+entre.*?y\s+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?)\.\s*ANG[ÉE]LICA",
        r"celebrado\s+entre.*?y\s+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?)\.\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+,\s*identificad",
        r"quien\s+en\s+adelante\s+se\s+denominar[áa]\s+EL\s+CONTRATISTA.*?por\s+la\s+otra,\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+?),",
    ]
    m = search_first(patterns, text)
    if m:
        contractor_name = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:\n\t")
        contractor_name = re.sub(r"^la\s+tecnolog[íi]a\s+y\s+", "", contractor_name, flags=re.IGNORECASE)
        return contractor_name
    return ""



def extract_contractor_document(text: str, contractor_name: str = "") -> str:
    def normalize_for_search(s: str) -> str:
        s = s.replace("\xa0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    text_norm = normalize_for_search(text)
    num_pattern = r"([0-9OIl][0-9OIl\.\,\-\s]{5,}[0-9OIl])"

    id_patterns = [
        rf"c[ée]dula\s+de\s+ciudadan[íi]a\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
        rf"\bC\.?\s*C\.?\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
        rf"\bc[ée]dula\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
        rf"identificad[oa]\s*(?:\(a\))?\s+con\s+(?:la\s+)?c[ée]dula\s+de\s+ciudadan[íi]a\s*(?:No\.?|N°|Nº|#|:)?\s*{num_pattern}",
    ]

    if contractor_name:
        contractor_name_esc = re.escape(normalize_for_search(contractor_name))
        m_name = re.search(contractor_name_esc, text_norm, re.IGNORECASE)
        if m_name:
            start = max(0, m_name.start() - 80)
            end = min(len(text_norm), m_name.end() + 300)
            window = text_norm[start:end]
            for pattern in id_patterns:
                m = re.search(pattern, window, re.IGNORECASE)
                if m:
                    return only_digits(m.group(1))

    m_other = re.search(
        r"por\s+la\s+otra,?(.*?)(?:actuando\s+en\s+nombre\s+propio|qu[ií]en\s+declara|EL\s+CONTRATISTA)",
        text_norm,
        re.IGNORECASE | re.DOTALL,
    )
    if m_other:
        block = m_other.group(1)
        for pattern in id_patterns:
            m = re.search(pattern, block, re.IGNORECASE)
            if m:
                return only_digits(m.group(1))

    found = []
    for pattern in id_patterns:
        for m in re.finditer(pattern, text_norm, re.IGNORECASE):
            value = only_digits(m.group(1))
            if 6 <= len(value) <= 12:
                found.append(value)

    return found[-1] if found else ""



def extract_obligaciones_especificas(text: str) -> str:
    start_patterns = [
        r"B\)\s*OBLIGACIONES\s+ESPEC[ÍI]FICAS\s*:\s*A\s*EL\s+CONTRATISTA\s+le\s+corresponde\s+el\s+cumplimiento\s+de\s+las\s+siguientes\s+obligaciones\s*:",
        r"B\)\s*OBLIGACIONES\s+ESPEC[ÍI]FICAS\s*:",
        r"OBLIGACIONES\s+ESPEC[ÍI]FICAS\s*:",
    ]
    end_patterns = [
        r"C\)\s*OBLIGACIONES\s+DE\s+LA\s+CONTRATANTE",
        r"OBLIGACIONES\s+DE\s+LA\s+CONTRATANTE",
        r"CL[ÁA]USULA\s+TERCERA\s*:",
        r"CL[ÁA]USULA\s+CUARTA\s*:",
        r"CL[ÁA]USULA\s+DE\s+SUPERVISI[ÓO]N",
    ]

    start_match = search_first(start_patterns, text)
    if not start_match:
        return ""

    tail = text[start_match.end():]
    end_match = search_first(end_patterns, tail)
    obligaciones = tail[: end_match.start()] if end_match else tail
    obligaciones = obligaciones.strip()

    obligaciones = re.sub(r"\n{3,}", "\n\n", obligaciones)
    return obligaciones.strip()



def extract_supervisor_name(text: str) -> str:
    patterns = [
        r"(?:la\s+)?supervisi[óo]n\s+(?:del\s+presente\s+contrato|contractual)?\s*(?:ser[áa]\s+ejercida|estar[áa]\s+a\s+cargo|corresponder[áa])\s+por\s+([^\n\.]+)",
        r"supervisor(?:a)?\s+del\s+contrato\s*(?:ser[áa]|es|:)?\s*([^\n\.]+)",
        r"la\s+supervisi[óo]n\s+ser[áa]\s+ejercida\s+por\s+([^\n\.]+)",
    ]

    m = search_first(patterns, text)
    if not m:
        return ""

    value = m.group(1)
    value = re.split(r";|,\s*quien\s+|,\s*o\s+por\s+quien|\s+o\s+por\s+quien", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\s+", " ", value).strip(" ,.;:\n\t")
    value = re.sub(r"^(el|la|los|las)\s+", "", value, flags=re.IGNORECASE)
    return value


# =========================
# Extracción con IA
# =========================
def get_openai_client(api_key: Optional[str]) -> Optional[OpenAI]:
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None



def build_focus_context(text: str, obligaciones_regla: str, supervisor_regla: str) -> str:
    parts = []
    header = cut_text(text, 12000)
    parts.append("=== INICIO DEL CONTRATO / CONTEXTO GENERAL ===\n" + header)

    if obligaciones_regla:
        parts.append("=== BLOQUE DETECTADO DE OBLIGACIONES ESPECÍFICAS ===\n" + cut_text(obligaciones_regla, 8000))

    if supervisor_regla:
        parts.append("=== CANDIDATO DE SUPERVISIÓN DETECTADO ===\n" + supervisor_regla)

    # ventana adicional de supervisión
    m = re.search(r"supervisi[óo]n", text, re.IGNORECASE)
    if m:
        start = max(0, m.start() - 1000)
        end = min(len(text), m.start() + 2500)
        parts.append("=== CONTEXTO DE SUPERVISIÓN ===\n" + text[start:end])

    return "\n\n".join(parts)



def extract_contract_fields_raw(client: OpenAI, text: str, filename: str, rule_candidates: dict) -> str:
    focus_text = build_focus_context(
        text=text,
        obligaciones_regla=rule_candidates.get("obligaciones_especificas", ""),
        supervisor_regla=rule_candidates.get("nombre_supervisor", ""),
    )

    prompt = f"""
A partir del siguiente contrato en español, extrae SOLO estos campos y devuelve SOLO JSON válido:
- numero_contrato
- Tipo_contrato
- nombre_contratista
- numero_documento_contratista
- obligaciones_especificas
- nombre_supervisor

Reglas obligatorias:
1. No inventes datos.
2. Si un campo no aparece claramente, devuelve "".
3. numero_documento_contratista debe quedar SOLO con dígitos.
4. obligaciones_especificas debe devolver el bloque textual de obligaciones específicas del contratista; conserva el contenido sustancial del contrato y no resumas.
5. nombre_supervisor debe devolver el nombre de la persona supervisora. Si el documento no trae nombre propio pero sí cargo explícito de supervisión, devuelve ese cargo. Si no aparece nada claro, devuelve "".
6. Devuelve únicamente JSON válido, sin explicación, sin markdown.

Archivo: {filename}

Candidatos por reglas:
{json.dumps(rule_candidates, ensure_ascii=False, indent=2)}

Texto del contrato:
{focus_text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Devuelve SOLO JSON válido. Sin markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content



def normalize_ai_result(data: dict) -> dict:
    data = data or {}
    normalized = {field: normalize_nullable_text(data.get(field, "")) for field in TARGET_FIELDS}
    normalized["numero_documento_contratista"] = only_digits(normalized.get("numero_documento_contratista"))
    normalized["numero_contrato"] = normalized.get("numero_contrato", "").replace("–", "-")
    return normalized



def merge_results(rule_result: dict, ai_result: Optional[dict]) -> dict:
    result = {field: normalize_nullable_text(rule_result.get(field, "")) for field in TARGET_FIELDS}
    result["numero_documento_contratista"] = only_digits(result.get("numero_documento_contratista"))

    if not ai_result:
        return result

    ai_result = normalize_ai_result(ai_result)

    for field in ["numero_contrato", "Tipo_contrato", "nombre_contratista"]:
        if ai_result.get(field):
            result[field] = ai_result[field]

    if ai_result.get("numero_documento_contratista"):
        result["numero_documento_contratista"] = ai_result["numero_documento_contratista"]

    # obligaciones: prioriza el bloque por reglas si es amplio; usa IA como respaldo
    reglas_obl = result.get("obligaciones_especificas", "")
    ia_obl = ai_result.get("obligaciones_especificas", "")
    if len(reglas_obl) < 80 and ia_obl:
        result["obligaciones_especificas"] = ia_obl
    elif len(ia_obl) > len(reglas_obl) and len(reglas_obl) < 300:
        result["obligaciones_especificas"] = ia_obl

    # supervisor: prioriza nombre propio sobre cargo genérico
    supervisor_reglas = result.get("nombre_supervisor", "")
    supervisor_ia = ai_result.get("nombre_supervisor", "")
    if looks_like_person_name(supervisor_ia):
        result["nombre_supervisor"] = supervisor_ia
    elif not supervisor_reglas and supervisor_ia:
        result["nombre_supervisor"] = supervisor_ia

    return result


# =========================
# Procesamiento principal
# =========================
def process_single_pdf(pdf_bytes: bytes, filename: str, client: Optional[OpenAI] = None, use_ai: bool = True) -> Dict:
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    text = normalize_text(raw_text)

    contractor_name = extract_contractor_name(text)
    rule_result = {
        "numero_contrato": extract_contract_number(text, filename),
        "Tipo_contrato": extract_contract_type(text),
        "nombre_contratista": contractor_name,
        "numero_documento_contratista": extract_contractor_document(text, contractor_name),
        "obligaciones_especificas": extract_obligaciones_especificas(raw_text or text),
        "nombre_supervisor": extract_supervisor_name(text),
    }

    ai_result = None
    metodo = "reglas"
    error_ia = ""

    if use_ai and client is not None:
        try:
            raw_ai = extract_contract_fields_raw(client, text=text, filename=filename, rule_candidates=rule_result)
            ai_result = normalize_ai_result(safe_json_loads(raw_ai))
            metodo = "hibrido_reglas_ia"
        except Exception as e:
            error_ia = str(e)
            metodo = "reglas_con_fallo_ia"

    final_result = merge_results(rule_result, ai_result)
    final_result.update(
        {
            "archivo": Path(filename).name,
            "metodo_extraccion": metodo,
            "error_ia": error_ia,
            "texto_extraido_chars": len(text),
        }
    )
    return final_result



def process_zip(zip_bytes: bytes, client: Optional[OpenAI] = None, use_ai: bool = True, progress_bar=None, status_text=None) -> List[Dict]:
    results = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        pdf_files = [name for name in zf.namelist() if name.lower().endswith(".pdf")]
        total = len(pdf_files)

        for i, name in enumerate(pdf_files, start=1):
            if status_text is not None:
                status_text.write(f"Procesando {i}/{total}: {Path(name).name}")
            if progress_bar is not None and total > 0:
                progress_bar.progress(i / total)

            pdf_bytes = zf.read(name)
            try:
                results.append(process_single_pdf(pdf_bytes, name, client=client, use_ai=use_ai))
            except Exception as e:
                results.append(
                    {
                        "archivo": Path(name).name,
                        "numero_contrato": "",
                        "Tipo_contrato": "",
                        "nombre_contratista": "",
                        "numero_documento_contratista": "",
                        "obligaciones_especificas": "",
                        "nombre_supervisor": "",
                        "metodo_extraccion": "error",
                        "error_ia": "",
                        "texto_extraido_chars": 0,
                        "error": str(e),
                    }
                )
    return results



def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="contratos")
    return output.getvalue()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Extractor de contratos desde ZIP", page_icon="📄", layout="wide")

st.title("📄 Extractor de contratos desde archivo .zip")
st.write(
    "Carga un archivo **.zip** con contratos en **PDF**. La app genera una **base de datos en Excel** "
    "con: número del contrato, tipo de contrato, nombre del contratista, número de documento, "
    "obligaciones específicas y nombre del supervisor."
)

api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY")

with st.expander("🔑 Configuración de OpenAI"):
    api_key_input = st.text_input("OpenAI API Key", type="password", value="")
    if api_key_input:
        api_key = api_key_input
    st.caption("La IA es opcional, pero ayuda a fortalecer la extracción cuando las reglas no alcanzan.")

uploaded_zip = st.file_uploader("Sube el archivo .zip", type=["zip"])
use_ai = st.checkbox("Usar IA para fortalecer la extracción", value=True)
show_preview_json = st.checkbox("Mostrar vista previa JSON", value=False)

if uploaded_zip is not None:
    zip_bytes = uploaded_zip.read()
    client = get_openai_client(api_key) if use_ai else None

    if use_ai and client is None:
        st.warning("No encontré OPENAI_API_KEY. Se procesará solo con reglas.")

    if st.button("🚀 Procesar contratos"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        with st.spinner("Procesando contratos..."):
            data = process_zip(
                zip_bytes,
                client=client,
                use_ai=use_ai and client is not None,
                progress_bar=progress_bar,
                status_text=status_text,
            )

        progress_bar.progress(1.0)
        status_text.write("✅ Proceso terminado")
        st.success(f"Se procesaron {len(data)} archivo(s).")

        if data:
            df = pd.DataFrame(data)

            preferred_columns = [
                "archivo",
                "numero_contrato",
                "Tipo_contrato",
                "nombre_contratista",
                "numero_documento_contratista",
                "nombre_supervisor",
                "obligaciones_especificas",
                "metodo_extraccion",
                "texto_extraido_chars",
                "error_ia",
                "error",
            ]
            df = df[[c for c in preferred_columns if c in df.columns]]

            st.subheader("Vista previa")
            st.dataframe(df, use_container_width=True)

            excel_bytes = dataframe_to_excel_bytes(df)
            st.download_button(
                label="⬇️ Descargar Excel",
                data=excel_bytes,
                file_name="contratos_extraidos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if show_preview_json:
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                st.subheader("Vista previa del JSON")
                st.code(json_str[:12000] + ("\n\n..." if len(json_str) > 12000 else ""), language="json")

with st.expander("⚙️ Requisitos y ejecución local"):
    st.markdown(
        """
```bash
pip install streamlit openai pandas openpyxl pymupdf pypdf
streamlit run Extract_streamlit.py
```

Para permitir cargas grandes en Streamlit, crea `.streamlit/config.toml` con algo así:

```toml
[server]
maxUploadSize = 1024
maxMessageSize = 1024
```
        """
    )

with st.expander("📝 Notas"):
    st.markdown(
        """
- La extracción es **híbrida**: primero aplica reglas y luego, si hay API Key, usa IA para reforzar los campos más difíciles.
- `obligaciones_especificas` intenta conservar el bloque textual del contrato.
- `nombre_supervisor` devuelve preferiblemente el nombre de la persona supervisora; si el contrato solo trae el cargo, devuelve ese cargo.
- El resultado final está pensado para salir como **base de datos en Excel**.
        """
    )
