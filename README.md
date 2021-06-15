# Machine Learning in business

### Итоговый проект по курсу "Машинное обучение в бизнесе"

Стек: sklearn, numpy, pandas, xgboost, API: flask  
Данные: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

Задача: Предсказать удовлетворенность пассажира сервисом авиакомпании по данным опроса. Бинарная классификация.

Преобразование признаков: QuantileTransformer, One Hot Encoding

Запуск:
1. Запусть Flask-приложение: app/run_server.py
2. Использовать функцию get_prediction из app/get_predictions.py для получения предсказаных значений

Функция get_prediction:
Принимает на вход pandas-датафрейм. Возможно подать на вход функции как одну строку, так и целый датафрейм.
Возвращает json-объект со списком предсказанных значений. Значений имеют тип 'str': 'satisfied'/'neutral or dissatisfied'.

Поддирректории:
1. 'app/datasets/' - содержит датасеты в формате .csv
2. 'app/models/' - содержит предобученную модель в формате .dill
3. 'app/notebooks/' - содержит блокноты Jupyter Noutbook с этапами создания модели классификации