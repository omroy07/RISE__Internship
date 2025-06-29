from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)

CSV_FOLDER = r"D:\MLPROJECTS\MoneyVest1\companydataset"

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None

    if request.method == "POST":
        company = request.form.get("company", "").strip().upper()
        csv_path = os.path.join(CSV_FOLDER, f"{company}.csv")

        if not os.path.isfile(csv_path):
            error = f"CSV file for company '{company}' not found."
        else:
            try:
                df = pd.read_csv(csv_path)

                # Check if required column exists
                if "Close" not in df.columns:
                    error = "CSV file missing 'Close' column."
                elif len(df) < 10:
                    error = "Not enough data in CSV to make prediction."
                else:
                    # Simple prediction logic:
                    recent_avg = df['Close'].tail(5).mean()
                    prev_avg = df['Close'].iloc[-10:-5].mean()
                    prediction = 1 if recent_avg > prev_avg else 0

            except Exception as e:
                error = f"Error reading CSV: {e}"

    return render_template("index.html", error=error, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
