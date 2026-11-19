# ==============================================================================
# APLICACIÓN WEB INTERACTIVA: MACHINE LEARNING + GROQ API
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ==============================================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Sistema Predictivo de Riesgo", page_icon="⚡")

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = "gsk_TU_API_KEY_LOCAL_AQUI"

client = Groq(api_key=api_key)


@st.cache_resource
def cargar_modelo_ml():
    ruta_modelo = Path(__file__).resolve().parent / "modelo_riesgo.pkl"
    return joblib.load(ruta_modelo)


modelo = cargar_modelo_ml()

st.title("⚡ Evaluador Predictivo de Riesgo Académico")
st.write(
    "Sistema híbrido: **Machine Learning (Scikit-Learn)** para cálculo de "
    "probabilidad + **Groq API** para tutoría personalizada."
)

st.sidebar.header("Parámetros del Alumno")
asistencia = st.sidebar.slider("Porcentaje de Asistencia (%)", 0, 100, 80)
promedio = st.sidebar.number_input(
    "Promedio General (0.0 - 10.0)", 0.0, 10.0, 7.5, step=0.1
)
horas_estudio = st.sidebar.slider("Horas de Estudio Semanales", 0, 30, 8)

if st.button("Evaluar Estudiante"):
    datos_entrada = pd.DataFrame(
        [
            {
                "asistencia": asistencia,
                "promedio": promedio,
                "horas_estudio": horas_estudio,
            }
        ]
    )

    prediccion = modelo.predict(datos_entrada)[0]
    probabilidades = modelo.predict_proba(datos_entrada)[0]
    clases = modelo.classes_.tolist()

    st.subheader("📊 Diagnóstico del Modelo Predictivo")
    if prediccion == 1:
        prob_riesgo = probabilidades[clases.index(1)] * 100
        st.error(f"⚠️ **Riesgo Alto de Deserción** (Probabilidad: {prob_riesgo:.1f}%)")
        estado_texto = "ALTO RIESGO ACADÉMICO"
    else:
        prob_exito = probabilidades[clases.index(0)] * 100
        st.success(
            f"✅ **Bajo Riesgo Académico** (Probabilidad de Éxito: {prob_exito:.1f}%)"
        )
        estado_texto = "BAJO RIESGO / BUEN RENDIMIENTO"

    st.subheader("🤖 Plan de Tutoría Automatizado (Groq API)")

    with st.spinner("Generando plan pedagógico con Groq..."):
        prompt_sistema = "Eres un Tutor Académico universitario experto y motivador."
        prompt_usuario = f"""Un modelo predictivo de Machine Learning evaluó a un estudiante:
- Asistencia: {asistencia}%
- Promedio: {promedio}
- Horas de estudio semanales: {horas_estudio} hrs
- Diagnóstico del Modelo: {estado_texto}

Redacta un dictamen breve en Markdown:
1. **Análisis de Indicadores:** Explica qué variables influyeron en el diagnóstico.
2. **Plan de Acción (3 pasos):** Genera 3 acciones concretas y aplicables."""

        try:
            modelos_disponibles = [
                m.id
                for m in client.models.list().data
                if not any(
                    palabra in m.id.lower()
                    for palabra in ["whisper", "guard", "tts", "playai", "embed"]
                )
            ]

            if not modelos_disponibles:
                st.error(
                    "❌ No se encontraron modelos de texto disponibles en tu cuenta de Groq."
                )
                st.stop()

            modelo_activo = next(
                (m for m in modelos_disponibles if "llama" in m.lower()),
                modelos_disponibles[0],
            )

            completion = client.chat.completions.create(
                model=modelo_activo,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
                temperature=0.7,
                max_tokens=1024,
            )

            st.caption(f"⚡ Respuesta generada con el modelo: `{modelo_activo}`")
            st.markdown(completion.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ Error al consultar Groq API: {e}")
