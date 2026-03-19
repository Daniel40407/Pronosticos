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
def moving_average(series, n):
    return series.rolling(window=n).mean()

# -----------------------------
def error_metrics(real, forecast):
    df = pd.DataFrame({"real": real, "forecast": forecast}).dropna()
    
    mae = (abs(df["real"] - df["forecast"])).mean()
    mse = ((df["real"] - df["forecast"])**2).mean()
    mape = (abs((df["real"] - df["forecast"]) / df["real"])).mean() * 100
    
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "MAPE": round(mape, 2)
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
        return jsonify({"error": "No hay datos cargados"})

    n = int(request.json["n"])
    results = {}

    for col in data.columns[1:]:
        forecast = moving_average(data[col], n)
        errors = error_metrics(data[col], forecast)

        results[col] = {
            "real": data[col].tolist(),
            "forecast": forecast.fillna("").tolist(),
            "errors": errors
        }

    return jsonify(results)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)