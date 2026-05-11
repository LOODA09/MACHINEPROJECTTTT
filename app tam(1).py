import streamlit as st
import pandas as pd
import pickle


st.title('Booking Status Prediction')
st.write("This app predicts the booking status of a customer based on different features.")
st.image('https://www.chardhamhotel.in/blog/wp-content/uploads/2024/05/Hotel-Booking-Tips.jpg', width=700)

car_parking_space = st.checkbox("Car Parking Space Available?")
car_parking_space = 1 if car_parking_space else 0

first_time_visitor = st.checkbox("First Time Visitor?")
first_time_visitor = 1 if first_time_visitor else 0


leadtime = st.number_input("Enter the Lead Time in days", min_value=0, max_value=500)
if leadtime <= 1:
    leadtime_category = 0
elif leadtime <= 7:
    leadtime_category = 1
elif leadtime <= 30:
    leadtime_category = 2
elif leadtime <= 365:
    leadtime_category = 3
else:
    leadtime_category = 4

average_price = st.slider('Average Price', 1, 180, 50)
special_requests = st.slider('Number of Special Requests', 0, 5, 1)


totalnights_category = st.selectbox(
    "Select Number of Nights Category",
    options=[
        ("Day Use (0 nights)", 0),
        ("Short Stay (1-3 nights)", 1),
        ("Week Stay (4-7 nights)", 2),
        ("Two Weeks Stay (8-14 nights)", 3),
        ("Long Stay (>14 nights)", 4)
    ],
    format_func=lambda x: x[0]
)[1]


day_name = st.selectbox(
    "Select Day of the Week",
    options=[("Monday", 0), ("Tuesday", 1), ("Wednesday", 2),
             ("Thursday", 3), ("Friday", 4), ("Saturday", 5), ("Sunday", 6)],
    format_func=lambda x: x[0]
)[1]

month = st.selectbox(
    "Select Month",
    options=[("January", 1), ("February", 2), ("March", 3), ("April", 4),
             ("May", 5), ("June", 6), ("July", 7), ("August", 8),
             ("September", 9), ("October", 10), ("November", 11), ("December", 12)],
    format_func=lambda x: x[0]
)[1]

year = st.selectbox("Select Year", options=[2015, 2016, 2017, 2018])
cancellation_ratio = st.slider("Cancellation Ratio", 0.0, 1.0, step=0.01)

encoded_features = []


selected_meal = st.selectbox("Select Type of Meal", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
if selected_meal == "Meal Plan 1":  
    encoded_features += [0, 0, 0]
else:
    for meal in ["Meal Plan 2", "Meal Plan 3", "Not Selected"]:
        encoded_features.append(1 if selected_meal == meal else 0)


selected_room = st.selectbox("Select Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
if selected_room == "Room_Type 1":  
    encoded_features += [0, 0, 0, 0, 0, 0]
else:
    for room in ["Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]:
        encoded_features.append(1 if selected_room == room else 0)


selected_segment = st.selectbox("Select Market Segment Type", ["Aviation", "Complementary", "Corporate", "Offline", "Online"])
if selected_segment == "Aviation":  
    encoded_features += [0, 0, 0, 0]
else:
    for segment in ["Complementary", "Corporate", "Offline", "Online"]:
        encoded_features.append(1 if selected_segment == segment else 0)


selected_group = st.selectbox("Select Number of Children and Adults", ["1", "2", "3", "4", "5", "Group"])
if selected_group == "1":  
    encoded_features += [0, 0, 0, 0, 0]
else:
    for group in ["2", "3", "4", "5", "Group"]:
        encoded_features.append(1 if selected_group == group else 0)


data = [[
    car_parking_space, leadtime_category, average_price, special_requests,
    totalnights_category, day_name, month, year,
    cancellation_ratio, first_time_visitor
] + encoded_features]


feature_names = [
    'car_parking_space', 'lead_time', 'average_price', 'special_requests',
    'number_of_total_nights', 'day_name', 'month', 'year', 'cancellation_ratio',
    'first_time_visitor',
    'type_of_meal_Meal Plan 2', 'type_of_meal_Meal Plan 3', 'type_of_meal_Not Selected',
    'room_type_Room_Type 2', 'room_type_Room_Type 3', 'room_type_Room_Type 4', 'room_type_Room_Type 5', 'room_type_Room_Type 6', 'room_type_Room_Type 7',
    'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline', 'market_segment_type_Online',
    'number_of_children_and_adults_2', 'number_of_children_and_adults_3', 'number_of_children_and_adults_4', 'number_of_children_and_adults_5', 'number_of_children_and_adults_Group'
]


input_df = pd.DataFrame(data, columns=feature_names)


scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

input_df[['average_price']] = scaler.transform(input_df[['average_price']])


if st.button("Predict Booking Status"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "Not Canceled" if prediction == 1 else "Canceled"
    st.write("**Prediction Result**")
    st.write(f"Booking Status: **{result}**")
    st.write(f"Probability of Not Being Canceled: **{probability:.2%}**")
    
    st.write("---")
    st.write("**Input Features Sent to Model:**")
    st.dataframe(input_df)
