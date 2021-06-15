# USAGE
# Start the server:
# python run_server.py
# Submit a request via cURL:
# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

import dill
import flask
import pandas as pd

dill._dill._reverse_typemap['ClassType'] = type

# Инициализация Flask приложения и модели
app = flask.Flask(__name__)
model = None


def load_model(model_path):
    # загрузка модели
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to airline passenger satisfaction prediction process"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Инициализация словаря данных
    data = {"success": False}

    if flask.request.method == "POST":

        request_json = flask.request.get_data(as_text=True)

        df = pd.read_json(request_json)
        preds = model.predict(df)
        data["predictions"] = preds.tolist()
        # indicate that the request was a success
        data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    modelpath = "./models/xgbclf_pipeline.dill"
    load_model(modelpath)
    app.run()
