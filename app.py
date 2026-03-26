from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Estado global
data: pd.DataFrame | None = None
value_col: str = "ventas"


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def detect_value_column(df: pd.DataFrame) -> str:
    """Detecta la primera columna numérica que no sea 'producto'."""
    for col in df.columns:
        if col.lower() not in ("producto", "product", "item", "sku") \
                and pd.api.types.is_numeric_dtype(df[col]):
            return col
    raise ValueError("No se encontró una columna numérica válida en el CSV.")


def calcular_metricas(real: list, forecast: list) -> dict:
    """Calcula MAPE, MAPE', MSE, MAE y RMSE."""
    df_tmp = pd.DataFrame({"real": real, "forecast": forecast}).dropna()

    df_tmp["Error"]          = df_tmp["forecast"] - df_tmp["real"]
    df_tmp["Error_abs"]      = df_tmp["Error"].abs()
    df_tmp["MAPE"]            = df_tmp["Error_abs"] / df_tmp["real"].replace(0, 1)
    df_tmp["MAPE_prima"]      = df_tmp["Error_abs"] / df_tmp["forecast"].replace(0, 1)
    df_tmp["error_cuadrado"] = df_tmp["Error"] ** 2

    MSE = df_tmp["error_cuadrado"].mean()
    return {
        "MAPE":       round(float(df_tmp["MAPE"].mean()),       4),
        "MAPE_prima": round(float(df_tmp["MAPE_prima"].mean()), 4),
        "MSE":        round(float(MSE),                        4),
        "MAE":        round(float(df_tmp["Error_abs"].mean()), 4),
        "RMSE":       round(float(MSE ** 0.5),                 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MÉTODOS DE PRONÓSTICO
# ─────────────────────────────────────────────────────────────────────────────

def pronostico_promedio_movil(series: pd.Series, N: int) -> dict:
    """Promedio Móvil Simple."""
    df = pd.DataFrame({value_col: series.values})
    df["pronostico"] = df[value_col].rolling(window=N).mean().shift(1)

    real     = df[value_col].tolist()
    forecast = df["pronostico"].fillna("").tolist()
    forecast_clean = [v if v != "" else None for v in forecast]

    metricas = calcular_metricas(real, forecast_clean)
    next_val = round(float(series.tail(N).mean()), 2)

    return {
        "real":          real,
        "forecast":      forecast,
        "future":        [next_val],
        "future_labels": ["+1"],
        "next":          next_val,
        "errors":        metricas,
    }


def pronostico_suavizacion_doble(series: pd.Series, horizon: int = 6) -> dict:
    """
    Suavización Exponencial Doble (Holt's Linear Trend Method).
    Captura nivel y tendencia lineal. Ideal para series con tendencia pero sin estacionalidad.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        raise ImportError("Instala statsmodels: pip install statsmodels")

    s = series.reset_index(drop=True).astype(float)

    if len(s) < 4:
        raise ValueError(
            f"Se necesitan al menos 4 observaciones para Suavización Exponencial Doble. "
            f"Tu serie tiene {len(s)}."
        )

    model = ExponentialSmoothing(
        s,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    fitted  = model.fittedvalues.tolist()
    future  = model.forecast(horizon).tolist()

    # Primer valor ajustado suele ser idéntico al real; lo conservamos como None para consistencia
    forecast_full = [None] + fitted[1:]

    metricas = calcular_metricas(s.tolist(), fitted)

    return {
        "real":          s.tolist(),
        "forecast":      forecast_full,
        "future":        [round(v, 2) for v in future],
        "future_labels": [f"+{i+1}" for i in range(horizon)],
        "next":          round(future[0], 2),
        "errors":        metricas,
    }


def pronostico_winters(series: pd.Series, periodos_estacionales: int = 4,
                       horizon: int = 6) -> dict:
    """
    Método de Holt-Winters (Triple Exponential Smoothing).
    Captura nivel, tendencia y estacionalidad aditiva.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        raise ImportError("Instala statsmodels: pip install statsmodels")

    s = series.reset_index(drop=True).astype(float)

    min_obs = periodos_estacionales * 2
    if len(s) < min_obs:
        raise ValueError(
            f"Se necesitan al menos {min_obs} observaciones "
            f"para Winters con estacionalidad={periodos_estacionales}. "
            f"Tu serie tiene {len(s)}."
        )

    model = ExponentialSmoothing(
        s,
        trend="add",
        seasonal="add",
        seasonal_periods=periodos_estacionales,
        initialization_method="estimated",
    ).fit(optimized=True)

    fitted = model.fittedvalues.tolist()
    future = model.forecast(horizon).tolist()

    forecast_full = [None] + fitted[1:]

    metricas = calcular_metricas(s.tolist(), fitted)

    return {
        "real":          s.tolist(),
        "forecast":      forecast_full,
        "future":        [round(v, 2) for v in future],
        "future_labels": [f"+{i+1}" for i in range(horizon)],
        "next":          round(future[0], 2),
        "errors":        metricas,
    }


def pronostico_prophet(series: pd.Series, horizon: int = 6) -> dict:
    """
    Facebook/Meta Prophet.
    Modelo aditivo que descompone la serie en tendencia + estacionalidad + festivos.
    Robusto ante datos faltantes y cambios bruscos de tendencia.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Instala Prophet: pip install prophet   "
            "(requiere pystan; puede tardar unos minutos)"
        )

    s = series.reset_index(drop=True).astype(float)

    if len(s) < 4:
        raise ValueError(
            f"Se necesitan al menos 4 observaciones para Prophet. Tu serie tiene {len(s)}."
        )

    # Prophet requiere columnas 'ds' (fecha) e 'y' (valor)
    df_prophet = pd.DataFrame({
        "ds": pd.date_range(start="2020-01-01", periods=len(s), freq="MS"),
        "y":  s.values,
    })

    m = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,   # más rápido, sin intervalos de confianza
    )
    m.fit(df_prophet)

    # Histórico ajustado
    fitted_df = m.predict(df_prophet)
    fitted = fitted_df["yhat"].tolist()

    # Futuro
    future_df = m.make_future_dataframe(periods=horizon, freq="MS")
    pred_df   = m.predict(future_df)
    future_vals = pred_df["yhat"].tail(horizon).tolist()

    forecast_full = [None] + fitted[1:]

    metricas = calcular_metricas(s.tolist(), fitted)

    return {
        "real":          s.tolist(),
        "forecast":      forecast_full,
        "future":        [round(v, 2) for v in future_vals],
        "future_labels": [f"+{i+1}" for i in range(horizon)],
        "next":          round(future_vals[0], 2),
        "errors":        metricas,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _run_method(serie: pd.Series, method: str, params: dict) -> dict:
    """Ejecuta el método solicitado y retorna el resultado."""
    if method == "moving_average":
        n = int(params.get("n", 3))
        if n < 1:
            raise ValueError("N debe ser ≥ 1")
        if n >= len(serie):
            raise ValueError(f"N={n} es mayor o igual al número de datos ({len(serie)})")
        return pronostico_promedio_movil(serie, n)

    elif method == "double_exp":
        horizon = int(params.get("horizon", 6))
        return pronostico_suavizacion_doble(serie, horizon)

    elif method == "winters":
        seasonal = int(params.get("seasonal", 4))
        horizon  = int(params.get("horizon",  6))
        return pronostico_winters(serie, seasonal, horizon)

    elif method == "prophet":
        horizon = int(params.get("horizon", 6))
        return pronostico_prophet(serie, horizon)

    else:
        raise ValueError(f"Método desconocido: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Recibe el CSV y devuelve las columnas para que el usuario elija
    cuál es la columna de producto y cuál es la de valor.
    Si la detección automática es clara, también lo indica como sugerencia.
    """
    global data

    if "file" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()

        all_cols     = df.columns.tolist()
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        text_cols    = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

        # Sugerencias automáticas
        suggested_product = next(
            (c for c in all_cols if c in ("producto", "product", "item", "sku", "nombre", "name", "categoria", "category")),
            text_cols[0] if text_cols else None
        )
        suggested_value = next(
            (c for c in numeric_cols if c not in ("producto", "product", "item", "sku")),
            numeric_cols[0] if numeric_cols else None
        )

        # Guardar df en estado temporal (sin configurar todavía)
        data = df

        # Preview de las primeras filas
        preview = df.head(4).fillna("").astype(str).to_dict(orient="records")

        return jsonify({
            "message":           f"Archivo leído: {len(df)} filas, {len(all_cols)} columnas",
            "all_columns":       all_cols,
            "numeric_columns":   numeric_cols,
            "text_columns":      text_cols,
            "suggested_product": suggested_product,
            "suggested_value":   suggested_value,
            "total_rows":        len(df),
            "preview":           preview,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/configure", methods=["POST"])
def configure():
    """
    El usuario confirma qué columna es producto y cuál es valor.
    Body JSON:
    {
        "product_col": "nombre_producto",   // null si no hay columna de producto
        "value_col":   "ventas"
    }
    """
    global data, value_col

    if data is None:
        return jsonify({"error": "Primero carga un archivo CSV"}), 400

    body = request.get_json(force=True)
    p_col = body.get("product_col")
    v_col = body.get("value_col")

    if not v_col or v_col not in data.columns:
        return jsonify({"error": f"Columna de valor '{v_col}' no encontrada"}), 400

    try:
        df = data.copy()

        if p_col and p_col in df.columns:
            df = df.rename(columns={p_col: "producto"})
            products = sorted(df["producto"].dropna().unique().tolist())
        else:
            df["producto"] = "Serie única"
            products = ["Serie única"]

        value_col = v_col
        data = df

        return jsonify({
            "message":   f"{len(products)} producto(s) configurado(s)",
            "products":  products,
            "value_col": value_col,
            "total_rows": len(df),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Body JSON esperado:
    {
        "product":  "LaptopX1",
        "method":   "moving_average" | "double_exp" | "winters" | "prophet",
        "n":        3,
        "seasonal": 4,
        "horizon":  6
    }
    """
    global data, value_col

    if data is None:
        return jsonify({"error": "Primero carga un archivo CSV"}), 400

    body    = request.get_json(force=True)
    product = body.get("product")
    method  = body.get("method", "moving_average")

    if product and product != "Serie única":
        serie = data[data["producto"] == product][value_col].dropna().reset_index(drop=True)
    else:
        serie = data[value_col].dropna().reset_index(drop=True)

    if len(serie) < 2:
        return jsonify({"error": f"'{product}' no tiene suficientes datos"}), 400

    try:
        result = _run_method(serie, method, body)
        result["method"]  = method
        result["product"] = product
        result["n_data"]  = int(len(serie))
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """
    Compara todos los métodos disponibles para un producto.
    Body JSON:
    {
        "product":  "LaptopX1",
        "n":        3,
        "seasonal": 4,
        "horizon":  6
    }
    Retorna un dict con resultados por método (errores + próximo período).
    """
    global data, value_col

    if data is None:
        return jsonify({"error": "Primero carga un archivo CSV"}), 400

    body    = request.get_json(force=True)
    product = body.get("product")

    if product and product != "Serie única":
        serie = data[data["producto"] == product][value_col].dropna().reset_index(drop=True)
    else:
        serie = data[value_col].dropna().reset_index(drop=True)

    if len(serie) < 2:
        return jsonify({"error": f"'{product}' no tiene suficientes datos"}), 400

    methods = ["moving_average", "double_exp", "winters", "prophet"]
    results = {}

    for m in methods:
        try:
            r = _run_method(serie, m, body)
            results[m] = {
                "next":   r["next"],
                "errors": r["errors"],
                "future": r["future"],
                "future_labels": r["future_labels"],
            }
        except Exception as e:
            results[m] = {"error": str(e)}

    return jsonify({"product": product, "methods": results})


if __name__ == "__main__":
    app.run(debug=True)
