# USAGE
# Start the server:
# python run_server.py
# Submit a request via cURL:
# curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
# python simple_request.py

# import the necessary packages
import numpy as np
import dill
import flask
import pandas as pd

dill._dill._reverse_typemap['ClassType'] = type
# import cloudpickle

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to fraudelent prediction process"


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        id_, gender, customer_type, age, type_of_travel = '', '', '', '', ''
        class_, flight_distance, wifi_service = '', '', ''
        time_convenient, ease_of_online_booking = '', ''
        gate_location, food_and_drink, online_boarding = '', '', ''
        seat_comfort, inflight_entertainment, on_board_service = '', '', ''
        leg_room_service, baggage_handling, checkin_service = '', '', ''
        inflight_service, cleanliness, departure_delay, arrival_delay = '', '', '', ''

        request_json = flask.request.get_json()

        if request_json["id"]:
            id_ = request_json['id']
        if request_json["Gender"]:
            gender = request_json['Gender']
        if request_json["Customer Type"]:
            customer_type = request_json['Customer Type']
        if request_json["Age"]:
            age = request_json['Age']
        if request_json["Type of Travel"]:
            type_of_travel = request_json['Type of Travel']
        if request_json["Class"]:
            class_ = request_json['Class']
        if request_json["Flight Distance"]:
            flight_distance = request_json['Flight Distance']
        if request_json["Inflight wifi service"]:
            wifi_service = request_json['Inflight wifi service']
        if request_json["Departure/Arrival time convenient"]:
            time_convenient = request_json['Departure/Arrival time convenient']
        if request_json["Ease of Online booking"]:
            ease_of_online_booking = request_json['Ease of Online booking']
        if request_json["Gate location"]:
            gate_location = request_json['Gate location']
        if request_json["Food and drink"]:
            food_and_drink = request_json['Food and drink']
        if request_json["Online boarding"]:
            online_boarding = request_json['Online boarding']
        if request_json["Seat comfort"]:
            seat_comfort = request_json['Seat comfort']
        if request_json["Inflight entertainment"]:
            inflight_entertainment = request_json['Inflight entertainment']
        if request_json["On-board service"]:
            on_board_service = request_json['On-board service']
        if request_json["Leg room service"]:
            leg_room_service = request_json['Leg room service']
        if request_json["Baggage handling"]:
            baggage_handling = request_json['Baggage handling']
        if request_json["Checkin service"]:
            checkin_service = request_json['Checkin service']
        if request_json["Inflight service"]:
            inflight_service = request_json['Inflight service']
        if request_json["Cleanliness"]:
            cleanliness = request_json['Cleanliness']
        if request_json["Departure Delay in Minutes"]:
            departure_delay = request_json['Departure Delay in Minutes']
        if request_json["Arrival Delay in Minutes"]:
            arrival_delay = request_json['Arrival Delay in Minutes']

        preds = model.predict_proba(pd.DataFrame({"description": [description],
                                                  "company_profile": [company_profile],
                                                  "benefits": [benefits]}))
        data["predictions"] = preds[:, 1][0]
        data["description"] = description
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    modelpath = "./models/xgbclf_pipeline.dill"
    load_model(modelpath)
    app.run()
