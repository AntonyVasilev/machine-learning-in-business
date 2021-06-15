import json
from urllib import request
import numpy as np
import pandas as pd


def get_prediction(data):
    """
    Данная функция позволяет получить предсказание от модели через API
    """
    myurl = "http://127.0.0.1:5000/predict"
    req = request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = data.to_json()
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    response = request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']


if __name__ == '__main__':
    df = pd.read_csv('./datasets/test.csv', index_col=0)
    X = df.drop('satisfaction', axis=1)
    predictions = np.array(get_prediction(X.iloc[:10, :]))
    print(predictions[:10])
