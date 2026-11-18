import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la página
st.set_page_config(page_title="Optimización Agrícola", layout="wide")
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola")
st.markdown("Monitoreo de telemetría en invernaderos inteligentes: Regresión Lineal y Prueba de Tratamientos.")

# ==========================================
# FASE 1: SIMULACIÓN DE DATOS CON DUCKDB
# ==========================================
@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")

    query = """
    CREATE TABLE telemetria AS
    WITH base AS (
        SELECT 
            range AS parcela_id,
            ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
            ROUND(50.0 + random() * 100.0, 2) AS fertilizante_kg,
            ROUND(18.0 + random() * 10.0, 2) AS temperatura_c,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            (-0.8 + random() * 1.6) AS ruido
        FROM range(1, 101)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        temperatura_c,
        tipo_tratamiento,
        ROUND(
            5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
            CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END + ruido, 
        2) AS rendimiento_ton
    FROM base;
    """
    con.execute(query)
    df = con.execute("SELECT * FROM telemetria").df()
    return df

df_completo = cargar_datos()

# ==========================================
# CONTROLES INTERACTIVOS (BARRA LATERAL)
# ==========================================
st.sidebar.header("🎛️ Filtros de Telemetría")
tratamiento_sel = st.sidebar.multiselect(
    "Selecciona el Tipo de Tratamiento:",
    options=df_completo['tipo_tratamiento'].unique(),
    default=df_completo['tipo_tratamiento'].unique()
)

riego_min, riego_max = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    float(df_completo['volumen_riego'].min()),
    float(df_completo['volumen_riego'].max()),
    (float(df_completo['volumen_riego'].min()), float(df_completo['volumen_riego'].max()))
)

df = df_completo[
    (df_completo['tipo_tratamiento'].isin(tratamiento_sel)) &
    (df_completo['volumen_riego'].between(riego_min, riego_max))
]

if len(df) < 5:
    st.warning("Selecciona más datos en los filtros para calcular los modelos estadísticos.")
    st.stop()

# ==========================================
# FASE 2: REGRESIÓN LINEAL MÚLTIPLE (OLS)
# ==========================================
X = df[['volumen_riego', 'fertilizante_kg']]
X = sm.add_constant(X)
y = df['rendimiento_ton']

modelo_ols = sm.OLS(y, X).fit()
r2_ajustado = modelo_ols.rsquared_adj
residuos = modelo_ols.resid

# ==========================================
# FASE 3: PRUEBA DE HIPÓTESIS (t-STUDENT)
# ==========================================
grupo_a = df_completo[df_completo['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_b = df_completo[df_completo['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

t_stat, p_valor = stats.ttest_ind(grupo_b, grupo_a, alternative='greater')

diff_medias = grupo_b.mean() - grupo_a.mean()
se_diff = np.sqrt(grupo_b.var(ddof=1)/len(grupo_b) + grupo_a.var(ddof=1)/len(grupo_a))
gl = len(grupo_b) + len(grupo_a) - 2
ic_95 = stats.t.interval(0.95, gl, loc=diff_medias, scale=se_diff)

# ==========================================
# FASE 5: TARJETAS KPI DE RESUMEN
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric("Rendimiento Promedio Global", f"{df['rendimiento_ton'].mean():.2f} ton/ha")
col2.metric("R² Ajustado del Modelo OLS", f"{r2_ajustado * 100:.1f}%")
col3.metric("P-Valor (Tratamiento B vs A)", f"{p_valor:.2e}", delta="Significativo (p < 0.05)" if p_valor < 0.05 else "No significativo")

st.divider()

# ==========================================
# FASE 4: CUADRÍCULA 2x2 DE GRÁFICAS
# ==========================================
fila1_col1, fila1_col2 = st.columns(2)

with fila1_col1:
    st.subheader("1. Mapa de Calor de Correlación (Pearson)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    cols_num = ['volumen_riego', 'fertilizante_kg', 'temperatura_c', 'rendimiento_ton']
    corr = df[cols_num].corr()
    sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax1)
    st.pyplot(fig1)

with fila1_col2:
    st.subheader("2. Regresión OLS: Riego vs Rendimiento")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df, x='volumen_riego', y='rendimiento_ton', ax=ax2, 
                scatter_kws={'alpha':0.6, 'color':'#2e7d32'}, line_kws={'color':'#d32f2f'})
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig2)

fila2_col1, fila2_col2 = st.columns(2)

with fila2_col1:
    st.subheader("3. Comparativa de Tratamientos (Prueba t)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_completo, x='tipo_tratamiento', y='rendimiento_ton', hue='tipo_tratamiento', palette="Set2", legend=False, ax=ax3)
    ax3.set_xlabel("Tipo de Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig3)

with fila2_col2:
    st.subheader("4. Diagnóstico de Errores (Residuos OLS)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color='#1976d2', ax=ax4)
    ax4.axvline(0, color='red', linestyle='--')
    ax4.set_xlabel("Residuo (Real - Predicho)")
    ax4.set_ylabel("Frecuencia")
    st.pyplot(fig4)

# ==========================================
# CONCLUSIONES TÉCNICAS Y TABLA DE DATOS
# ==========================================
with st.expander("📋 Ver Veredicto Estadístico y Resumen del Modelo OLS"):
    st.markdown(f"""
    **Veredicto de Ingeniería (Subtema 4.3):**
    * **Diferencia de medias:** El Tratamiento B produce en promedio **{diff_medias:.2f} ton/ha** más que el Tratamiento A.
    * **Intervalo de Confianza al 95%:** `[{ic_95[0]:.2f}, {ic_95[1]:.2f}] ton/ha`.
    * **Decisión:** Dado que el p-valor (`{p_valor:.2e}`) es menor a $\\alpha = 0.05$, se **rechaza la hipótesis nula ($H_0$)**. Se recomienda autorizar la inversión para escalar el Tratamiento B a todo el invernadero.
    """)
    st.text(modelo_ols.summary())

with st.expander("🗄️ Ver Datos Simulados (DuckDB)"):
    st.dataframe(df)