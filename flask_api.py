# Import required python libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import traceback
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
# API definition
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("frontend.html")

@app.route('/predict', methods=['POST'])
def predict():
    with graph.as_default():
        payload = request.json
        try:
            model = load_model('pre_trained.hdf5')
            with open('feature_transformer.sav', 'rb') as filehandle:
                feature_transformer = pickle.load(filehandle)
            with open('target_transformer.sav', 'rb') as filehandle:
                target_transformer = pickle.load(filehandle)
            BuildingArea = int(payload['BuildingArea'])
            Rooms = int(payload['Rooms'])
            Postcode = int(payload['Postcode'])
            main_df = pd.read_csv('real_estate_data.csv')
            if Postcode not in main_df.Postcode.unique():
                return 'Data for Given Postal Code is not trained'
            if Rooms not in main_df.Rooms.unique():
                return 'Data for Given Number of Rooms is not trained'
            if BuildingArea <= 0:
                return 'Enter a valid build area'
            input_array = np.array([[BuildingArea], [Rooms], [Postcode]]).reshape(1,-1)
            df_temp = pd.DataFrame(input_array)
            df_temp.columns = ['BuildingArea', 'Rooms', 'Postcode']
            input_array = feature_transformer.transform(df_temp)
            if model:
                    prediction = target_transformer.inverse_transform(model.predict(input_array))
                    return str(prediction[0][0])
        except:
            return str(traceback.format_exc())

@app.route('/retrain', methods=['POST'])
def retrain():
    with graph.as_default():
        payload = request.json
        try:
            BuildingArea = int(payload['BuildingArea'])
            Rooms = int(payload['Rooms'])
            Postcode = int(payload['Postcode'])
            Price = int(payload['Price'])
            if BuildingArea <= 0:
                return 'Enter a valid build area'
            input_array = np.array([[Rooms], [BuildingArea], [Postcode], [Price]]).reshape(1,-1)
            df_temp = pd.DataFrame(input_array)
            df_temp.columns = ['Rooms', 'BuildingArea', 'Postcode', 'Price']
            main_df = pd.read_csv('real_estate_data.csv')
            dataframe = pd.concat([main_df,df_temp]).reset_index(drop=True)
            dataframe.dropna(inplace=True)
            dataframe.to_csv('real_estate_data.csv', index=False)
            feature_dataset = dataframe[['Rooms','BuildingArea','Postcode']]
            target_dataset = dataframe[['Price']]
            feature_transformer = make_column_transformer(
                    (['BuildingArea'], StandardScaler()),
                    (['Rooms', 'Postcode'], OneHotEncoder(categories="auto",drop="first"))
            )
            X = feature_transformer.fit_transform(feature_dataset)
            with open('feature_transformer.sav', 'wb') as filehandle:
                pickle.dump(feature_transformer, filehandle)
            target_transformer = StandardScaler()
            y = target_transformer.fit_transform(np.array(target_dataset).reshape(-1,1))            
            with open('target_transformer.sav', 'wb') as filehandle:
                pickle.dump(target_transformer, filehandle)
            # define the keras model
            model = Sequential()
            # Layer 1
            model.add(Dense(50, input_dim=X.shape[1], activation='relu'))
            # Dropout regularization is added to avoid overfitting
            model.add(Dropout(0.1))
            # Layer 2
            model.add(Dense(50, activation='relu'))
            # Dropout regularization is added to avoid overfitting
            model.add(Dropout(0.1))
            # Layer 3
            model.add(Dense(50, activation='relu'))
            # Output Layer
            model.add(Dense(1))
            #Compile the model
            model.compile(loss='mae', optimizer='adam', metrics=['mae'])
            model.fit(X, y, batch_size=32, epochs=50)
            model.save('pre_trained.hdf5')
            
            return "Model retrained with new data"
        except:
            return str(traceback.format_exc())

@app.route('/reset', methods=['POST'])
def reset():
    with graph.as_default():
        try:
            dataframe = pd.read_csv('real_estate_data_org.csv')
            dataframe.to_csv('real_estate_data.csv', index=False)
            feature_dataset = dataframe.iloc[:, 0:3]
            target_dataset = dataframe.iloc[:, 3]
            feature_transformer = make_column_transformer(
                    (['BuildingArea'], StandardScaler()),
                    (['Rooms', 'Postcode'], OneHotEncoder(categories="auto",drop="first"))
            )
            X = feature_transformer.fit_transform(feature_dataset)
            with open('feature_transformer.sav', 'wb') as filehandle:
                pickle.dump(feature_transformer, filehandle)
            target_transformer = StandardScaler()
            y = target_transformer.fit_transform(np.array(target_dataset).reshape(-1,1))            
            with open('target_transformer.sav', 'wb') as filehandle:
                pickle.dump(target_transformer, filehandle)
            # define the keras model
            model = Sequential()
            # Layer 1
            model.add(Dense(50, input_dim=X.shape[1], activation='relu'))
            # Dropout regularization is added to avoid overfitting
            model.add(Dropout(0.1))
            # Layer 2
            model.add(Dense(50, activation='relu'))
            # Dropout regularization is added to avoid overfitting
            model.add(Dropout(0.1))
            # Layer 3
            model.add(Dense(50, activation='relu'))
            # Output Layer
            model.add(Dense(1))
            #Compile the model
            model.compile(loss='mae', optimizer='adam', metrics=['mae'])
            model.fit(X, y, batch_size=32, epochs=50)
            model.save('pre_trained.hdf5')
            
            return "Model reset with original train data"
        except:
            return str(traceback.format_exc())
        
if __name__ == '__main__':
    global graph
    graph = tf.get_default_graph()
    global model 
    model = load_model('pre_trained.hdf5')
    with open('feature_transformer.sav', 'rb') as filehandle:
        global feature_transformer 
        feature_transformer = pickle.load(filehandle)
    with open('target_transformer.sav', 'rb') as filehandle:
        global target_transformer 
        target_transformer = pickle.load(filehandle)
    app.run(host='0.0.0.0', port=5802, debug=True)
