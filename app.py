import gradio as gr
import tensorflow as tf
import joblib
import numpy as np
from datetime import datetime

current_time = datetime.today()
current_time_bank = str(current_time.day) + '.' + str(current_time.month) + '.' + str(current_time.year)
time_6_months_ago = str(current_time.day) + '.' + str(current_time.month-6) + '.' + str(current_time.year)
url_inf = "https://cbr.ru/hd_base/infl/?UniDbQuery.Posted=True&UniDbQuery.From=" + time_6_months_ago +"&UniDbQuery.To=" + current_time_bank
url_usd = 'https://www.cbr.ru/currency_base/dynamics/?UniDbQuery.Posted=True&UniDbQuery.so=1&UniDbQuery.mode=1&UniDbQuery.date_req1=' + time_6_months_ago + '&UniDbQuery.date_req2=' + current_time_bank +'&UniDbQuery.VAL_NM_RQ=R01235'


model   = tf.keras.models.load_model('rate_bidir_lstm.h5')
scaler_X = joblib.load('scaler_X_lstm.gz')
scaler_y = joblib.load('scaler_y_lstm.gz')

def predict(usd_4, usd_3, usd_2, usd_1,
            inf_4, inf_3, inf_2, inf_1,
            ks_4,  ks_3,  ks_2):
    x = np.array([[usd_4, usd_3, usd_2, usd_1,
                   inf_4, inf_3, inf_2, inf_1,
                   ks_4,  ks_3,  ks_2]], dtype=float)
    x = scaler_X.transform(x.reshape(-1,1)).reshape(1,11,1)
    pred = scaler_y.inverse_transform(model.predict(x))[0,0]
    return round(pred, 2)

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=f"{c}_{i}") for c, i in
            [('usd',4),('usd',3),('usd',2),('usd',1),
             ('inf',4),('inf',3),('inf',2),('inf',1),
             ('ks',4),('ks',3),('ks',2)]],
    outputs=gr.Number(label="Прогноз КС, %"),
    title="CBR Key-Rate Forecast",
    description=(
    "Введите значения USD за последние 4 месяца, 4 последних месяных CPI и 3 последних КС → получите прогноз по будущей ставке.\n\n"
    f"Текущий курс USD: {url_usd}\n\n"
    f"Текущие КС и инфляция: {url_inf}"
)
)
iface.launch()