from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

data = None

# -----------------------------
# FUNCIÓN DE PRONÓSTICO
# -----------------------------
def pronosticar(N):

    global data

    df = data.copy()

    df["pronostico"] = df["ventas"].rolling(window=N).mean().shift(1)

    df["Error"] = df["pronostico"] - df["ventas"]
    df["Error_abs"] = df["Error"].abs()

    df["APE"] = df["Error_abs"] / df["ventas"].replace(0, 1)
    df["APE_prima"] = df["Error_abs"] / df["pronostico"].replace(0, 1)

    df["error_cuadrado"] = df["Error"] ** 2

    MAPE = df["APE"].mean()
    MAPE_prima = df["APE_prima"].mean()
    MSE = df["error_cuadrado"].mean()
    MAE = df["Error_abs"].mean()
    RMSE = MSE ** 0.5

    return {
        "real": df["ventas"].tolist(),
        "forecast": df["pronostico"].fillna("").tolist(),
        "errors": {
            "MAPE": round(MAPE, 4),
            "MAPE_prima": round(MAPE_prima, 4),
            "MSE": round(MSE, 4),
            "MAE": round(MAE, 4),
            "RMSE": round(RMSE, 4)
        }
    }

# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    global data

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    data = pd.read_csv(filepath)

    return jsonify({"message": "Archivo cargado correctamente"})

# -----------------------------
@app.route("/forecast", methods=["POST"])
def forecast():
    global data

    if data is None:
        return jsonify({"error": "Primero carga un archivo"})

    n = int(request.json["n"])

    result = pronosticar(n)

    return jsonify(result)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)