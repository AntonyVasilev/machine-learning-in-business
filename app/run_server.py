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
    return "Welcome to airline passenger satisfaction prediction process"


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        # gender, customer_type, age, type_of_travel = '', '', '', ''
        # class_, flight_distance, wifi_service = '', '', ''
        # ease_of_online_booking, inflight_service = '', ''
        # food_and_drink, online_boarding, cleanliness = '', '', ''
        # seat_comfort, inflight_entertainment, on_board_service = '', '', ''
        # leg_room_service, baggage_handling, checkin_service = '', '', ''

        request_json = flask.request.get_json()
        # print(1)

        # if request_json["Gender"]:
        #     gender = request_json['Gender']
        # if request_json["Customer Type"]:
        #     customer_type = request_json['Customer Type']
        # if request_json["Age"]:
        #     age = request_json['Age']
        # if request_json["Type of Travel"]:
        #     type_of_travel = request_json['Type of Travel']
        # if request_json["Class"]:
        #     class_ = request_json['Class']
        # if request_json["Flight Distance"]:
        #     flight_distance = request_json['Flight Distance']
        # if request_json["Inflight wifi service"]:
        #     wifi_service = request_json['Inflight wifi service']
        # if request_json["Ease of Online booking"]:
        #     ease_of_online_booking = request_json['Ease of Online booking']
        # if request_json["Food and drink"]:
        #     food_and_drink = request_json['Food and drink']
        # if request_json["Online boarding"]:
        #     online_boarding = request_json['Online boarding']
        # if request_json["Seat comfort"]:
        #     seat_comfort = request_json['Seat comfort']
        # if request_json["Inflight entertainment"]:
        #     inflight_entertainment = request_json['Inflight entertainment']
        # if request_json["On-board service"]:
        #     on_board_service = request_json['On-board service']
        # if request_json["Leg room service"]:
        #     leg_room_service = request_json['Leg room service']
        # if request_json["Baggage handling"]:
        #     baggage_handling = request_json['Baggage handling']
        # if request_json["Checkin service"]:
        #     checkin_service = request_json['Checkin service']
        # if request_json["Inflight service"]:
        #     inflight_service = request_json['Inflight service']
        # if request_json["Cleanliness"]:
        #     cleanliness = request_json['Cleanliness']
        # print(1)

        # df = pd.DataFrame({"Gender": [gender],
        #                    "Customer Type": [customer_type],
        #                    "Age": [age],
        #                    "Type of Travel": [type_of_travel],
        #                    "Class": [class_],
        #                    "Flight Distance": [flight_distance],
        #                    "Inflight wifi service": [wifi_service],
        #                    "Ease of Online booking": [ease_of_online_booking],
        #                    "Food and drink": [food_and_drink],
        #                    "Online boarding": [online_boarding],
        #                    "Seat comfort": [seat_comfort],
        #                    "Inflight entertainment": [inflight_entertainment],
        #                    "On-board service": [on_board_service],
        #                    "Leg room service": [leg_room_service],
        #                    "Baggage handling": [baggage_handling],
        #                    "Checkin service": [checkin_service],
        #                    "Inflight service": [inflight_service],
        #                    "Cleanliness": [cleanliness]})

        df = pd.DataFrame(request_json)
        # print(1)
        preds = model.predict(df)
        # preds = model.predict(df)

        data["predictions"] = preds[0]
        # indicate that the request was a success
        data["success"] = True
        # print(1)
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
