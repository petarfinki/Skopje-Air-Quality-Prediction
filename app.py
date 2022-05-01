import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('measurements_Skopje.csv')

st.set_page_config('Skopje Air Quality Prediction App')

st.title('Predict Skopje air quality')

particles_list = ['PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2']
particles_choice = st.selectbox(label='Select particle to test on', options=particles_list)
temperature = st.number_input('Enter temperature in °C')
humidity = st.number_input('Enter humidity (g/m3)')
windspeed = st.number_input('Enter wind speed (km/h)')
cloudcover = st.number_input('Enter cloud cover (oktas)')

if particles_choice == 'PM10':
    df1 = df.drop(['time'], axis='columns')
    df2 = pd.get_dummies(df.conditions)
    df3 = pd.concat([df1, df2], axis=1)
    X = df3.drop(['conditions', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2'], axis='columns')
    y = df3.PM10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    linear_reg = LinearRegression()
    regr = RandomForestRegressor(n_estimators=200, random_state=1234)

    linear_reg.fit(X_train.values, y_train.values)
    regr.fit(X_train.values, y_train.values)

    linear_score_pm10 = linear_reg.score(X_test.values, y_test.values)
    column_data = X.columns.values
    pred_pm10 = linear_reg.predict(X_test)[0]


elif particles_choice == 'O3':
    df1 = df.drop(['time'], axis='columns')
    df2 = pd.get_dummies(df.conditions)
    df3 = pd.concat([df1, df2], axis=1)
    X = df3.drop(['conditions', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2'], axis='columns')
    y = df3.O3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    linear_reg = LinearRegression()
    regr = RandomForestRegressor(n_estimators=200, random_state=1234)

    linear_reg.fit(X_train.values, y_train.values)
    regr.fit(X_train.values, y_train.values)

    linear_score_O3 = linear_reg.score(X_test.values, y_test.values)
    column_data = X.columns.values
    pred_O3 = linear_reg.predict(X_test)[0]

elif particles_choice == 'NO2':
    df1 = df.drop(['time'], axis='columns')
    df2 = pd.get_dummies(df.conditions)
    df3 = pd.concat([df1, df2], axis=1)
    X = df3.drop(['conditions', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2'], axis='columns')
    y = df3.NO2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    linear_reg = LinearRegression()
    regr = RandomForestRegressor(n_estimators=200, random_state=1234)

    linear_reg.fit(X_train.values, y_train.values)
    regr.fit(X_train.values, y_train.values)

    linear_score_NO2 = linear_reg.score(X_test.values, y_test.values)
    column_data = X.columns.values
    pred_NO2 = linear_reg.predict(X_test)[0]

elif particles_choice == 'CO':
    df1 = df.drop(['time'], axis='columns')
    df2 = pd.get_dummies(df.conditions)
    df3 = pd.concat([df1, df2], axis=1)
    X = df3.drop(['conditions', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2'], axis='columns')
    y = df3.CO
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    linear_reg = LinearRegression()
    regr = RandomForestRegressor(n_estimators=200, random_state=1234)

    linear_reg.fit(X_train.values, y_train.values)
    regr.fit(X_train.values, y_train.values)

    linear_score_CO = linear_reg.score(X_test.values, y_test.values)
    column_data = X.columns.values
    pred_CO = linear_reg.predict(X_test)[0]

else:
    df1 = df.drop(['time'], axis='columns')
    df2 = pd.get_dummies(df.conditions)
    df3 = pd.concat([df1, df2], axis=1)
    X = df3.drop(['conditions', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2'], axis='columns')
    y = df3.SO2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    linear_reg = LinearRegression()
    regr = RandomForestRegressor(n_estimators=200, random_state=1234)

    linear_reg.fit(X_train.values, y_train.values)
    regr.fit(X_train.values, y_train.values)

    linear_score_SO2 = linear_reg.score(X_test.values, y_test.values)
    column_data = X.columns.values
    pred_SO2 = linear_reg.predict(X_test)[0]


def predict_price_linear(cestica, temperature, humidity, wind, cloud):
    try:
        cestica_index = particles_list.index(cestica)
    except ValueError:
        cestica_index = -1

    x = np.zeros(len(column_data))
    x[0] = temperature
    x[1] = humidity
    x[2] = wind
    x[3] = cloud

    return linear_reg.predict([x])[0]

def predict_price_forest(cestica, temperature, humidity, wind, cloud):
    try:
        cestica_index = particles_list.index(cestica)
    except ValueError:
        cestica_index = -1

    x = np.zeros(len(column_data))
    x[0] = temperature
    x[1] = humidity
    x[2] = wind
    x[3] = cloud

    return regr.predict([x])[0]
algorithms = ['Decision Tree Regression', 'Linear Regression']
select_algorithm = st.selectbox('Choose algorithm for prediction', algorithms)
if st.button('Predict'):
    if select_algorithm == 'Linear Regression':
        if particles_choice == 'PM10':
            predicted_price = st.subheader(predict_price_linear(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> μg/m3 </h5>", unsafe_allow_html=True)
            if predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 26:
                st.write('The quality is very good.')
            elif 25 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 51:
                st.write('The quality is good.')
            elif 50 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 91:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 90 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 181:
                st.write('The quality is bad.')
            elif predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) > 180:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')


        elif particles_choice == 'O3':
            predicted_price = st.subheader(predict_price_linear(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> g/Nm3 </h5>", unsafe_allow_html=True)
            if predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 61:
                st.write('The quality is very good.')
            elif 60 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 121:
                st.write('The quality is good.')
            elif 120 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 181:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 180 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 241:
                st.write('The quality is bad.')
            elif predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) > 240:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')

        elif particles_choice == 'NO2':
            predicted_price = st.subheader(
                predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover))
            st.markdown("<h5 style='text-align: left;'> mol/m² </h5>", unsafe_allow_html=True)
            if predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 51:
                st.write('The quality is very good.')
            elif 50 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 101:
                st.write('The quality is good.')
            elif 100 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 201:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 200 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 401:
                st.write('The quality is bad.')
            elif predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) > 400:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')

        elif particles_choice == 'CO':
            predicted_price = st.subheader(predict_price_linear(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> ppm </h5>", unsafe_allow_html=True)
            if predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 6:
                st.write('The quality is very good.')
            elif 5 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 7.6:
                st.write('The quality is good.')
            elif 7.5 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 11:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 10 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 21:
                st.write('The quality is bad.')
            elif predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) > 20:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')

        elif particles_choice == 'SO2':
            predicted_price = st.subheader(predict_price_linear(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> ppm </h5>", unsafe_allow_html=True)
            if predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 51:
                st.write('The quality is very good.')
            elif 50 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 101:
                st.write('The quality is good.')
            elif 100 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 351:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 350 < predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) < 501:
                st.write('The quality is bad.')
            elif predict_price_linear(particles_choice, temperature, humidity, windspeed, cloudcover) > 500:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')
        else:
            st.write('Mistake')
    elif select_algorithm == 'Decision Tree Regression':
        if particles_choice == 'PM10':
            predicted_price = st.subheader(predict_price_forest(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> μg/m3 </h5>", unsafe_allow_html=True)
            if predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 26:
                st.write('The quality is very good.')
            elif 25 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 51:
                st.write('The quality is good.')
            elif 50 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 91:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 90 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 181:
                st.write('The quality is bad.')
            elif predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) > 180:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')


        elif particles_choice == 'O3':
            predicted_price = st.subheader(predict_price_forest(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> g/Nm3 </h5>", unsafe_allow_html=True)
            if predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 61:
                st.write('The quality is very good.')
            elif 60 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 121:
                st.write('The quality is good.')
            elif 120 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 181:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 180 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 241:
                st.write('The quality is bad.')
            elif predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) > 240:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')

        elif particles_choice == 'NO2':
            predicted_price = st.subheader(
                predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover))
            st.markdown("<h5 style='text-align: left;'> mol/m² </h5>", unsafe_allow_html=True)
            if predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 51:
                st.write('The quality is very good.')
            elif 50 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 101:
                st.write('The quality is good.')
            elif 100 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 201:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 200 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 401:
                st.write('The quality is bad.')
            elif predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) > 400:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')

        elif particles_choice == 'CO':
            predicted_price = st.subheader(predict_price_forest(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> ppm </h5>", unsafe_allow_html=True)
            if predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 6:
                st.write('The quality is very good.')
            elif 5 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 7.6:
                st.write('The quality is good.')
            elif 7.5 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 11:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 10 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 21:
                st.write('The quality is bad.')
            elif predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) > 20:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')

        elif particles_choice == 'SO2':
            predicted_price = st.subheader(predict_price_forest(particles_choice, temperature, humidity, windspeed,
                                                                cloudcover))
            st.markdown("<h5 style='text-align: left;'> ppm </h5>", unsafe_allow_html=True)
            if predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 51:
                st.write('The quality is very good.')
            elif 50 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 101:
                st.write('The quality is good.')
            elif 100 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 351:
                st.write('The quality is mid-level(but potentially hazardous).')
            elif 350 < predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) < 501:
                st.write('The quality is bad.')
            elif predict_price_forest(particles_choice, temperature, humidity, windspeed, cloudcover) > 500:
                st.write('The quality is very bad.')
            else:
                st.write('Mistake.')
        else:
            st.write('Mistake')