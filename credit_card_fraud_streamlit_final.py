import streamlit as st
import pandas as pd 
import pickle 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
 
st.title('Credit Card Fraud Detection')
st.write("This app uses 8 inputs to predict if the transaction is fraud or not ")
  
rf_model_path = './random_forest.pickle'
map_pickle = open(rf_model_path, 'rb')
loaded_model = pickle.load(map_pickle)
print("ML model loaded ......")
map_pickle.close()

# penguin_file = st.file_uploader('Upload your own data')

# if penguin_file is None:
#     print('penguin_file is Empty')
#     map_pickle = open(rf_model_path, 'rb')
#     loaded_model = pickle.load(map_pickle)
#     print("ML model loaded ......")
#     map_pickle.close()

# else:
#     print('penguin_file name', penguin_file.name)
#     penguin_df = pd.read_csv(penguin_file)
#     penguin_df = penguin_df.dropna()
#     penguin_df = penguin_df.drop_duplicates()
#     output = penguin_df['fraud']
#     features = penguin_df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
#                             'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
    
#     X = penguin_df.drop('fraud', axis=1)
#     y = penguin_df['fraud']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=13)
#     rf_model.fit(X_train, y_train)
    
#     pickle.dump(rf_model, open(rf_model_path, "wb"))
    
#     y_pred = rf_model.predict(X_test)
#     score = round(accuracy_score(y_pred, y_test), 2)
#     st.write('We trained a Random Forest model on these data,' 
#              ' it has a score of {}! Use the ' 
#              'inputs below to try out the model.'.format(score))
    
#     loaded_model = pickle.load(open(rf_model_path, "rb"))
    
#     features = pd.get_dummies(features)
#     output, credit_card_fraud_mapping = pd.factorize(output)
    
selectbox_options = {
  0: "No",
  1: "Yes"
}
with st.form('user_inputs'): 
  distance_from_home = st.number_input('Distance from home (KM)', min_value=0, key = 1)
  distance_from_last_transaction = st.number_input('Distance from last transaction (KM)', min_value=0, key = 2)
  median_purchase_price = st.number_input('Median Purchase Price (of card holder)', min_value=1, max_value=10000, key = 3)
  current_purchase_price = st.number_input('Current Transaction', min_value=0, max_value=10000, key = 8)
  repeat_retailer = st.selectbox('Repeat retailer', options=[0, 1], key = 4, format_func=lambda x: selectbox_options.get(x),)
  used_chip = st.selectbox ('Used chip', options=[0, 1], key = 5, format_func=lambda x: selectbox_options.get(x),)
  used_pin_number = st.selectbox ('Used pin number', options=[0, 1], key = 6, format_func=lambda x: selectbox_options.get(x),)
  online_order = st.selectbox('Online order', options=[0, 1], key = 7, format_func=lambda x: selectbox_options.get(x),)
  st.form_submit_button()


new_prediction = loaded_model.predict([[distance_from_home, distance_from_last_transaction, current_purchase_price/median_purchase_price, 
                               repeat_retailer, used_chip, used_pin_number, online_order]])
is_fraud = new_prediction[0] == 1
st.write('The credit card transaction fraud indicator : {} '.format(is_fraud))
print("prediction executed ......", is_fraud)