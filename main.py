import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained machine learning model
model = pickle.load(open('rfc_resampled.pkl','rb'))
st.set_page_config(
    page_title="Employee Churn",
    page_icon=":smiley:",
)

def map_categorical_values(value):
    if value == 'Low':
        return 0
    elif value == 'Medium':
        return 1
    elif value == 'High':
        return 2
    else:
        return value  # for boolean values (True/False)

# Function to get user input and make prediction
def predict_employee_leave(satisfaction_level, last_evaluation, number_project, 
                           average_montly_hours, time_spend_company, 
                           work_accident, promotion_last_5years, salary):
    # Map categorical values to numerical values
    work_accident = int(work_accident)
    promotion_last_5years = int(promotion_last_5years)
    salary = map_categorical_values(salary)

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'satisfaction_level': [satisfaction_level],
        'last_evaluation': [last_evaluation],
        'number_project': [number_project],
        'average_montly_hours': [average_montly_hours],
        'time_spend_company': [time_spend_company],
        'Work_accident': [work_accident],
        'promotion_last_5years': [promotion_last_5years],
        'salary': [salary],
    })

    # Make prediction
    prediction = model.predict(input_data)

    return prediction[0]

# Streamlit app
def main():
    st.title('Employee Churn Prediction App')

    # User input for features
    satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    last_evaluation = st.slider('Last Evaluation', 0.0, 1.0, 0.5)
    number_project = st.slider('Number of Projects', 1, 10, 5)
    average_monthly_hours = st.slider('Average Monthly Hours', 50, 300, 150)
    time_spend_company = st.slider('Time Spent in Company (years)', 1, 10, 3)
    work_accident = st.checkbox('Work Accident')
    promotion_last_5years = st.checkbox('Promotion in Last 5 Years')
    salary = st.selectbox('Salary', ['Low', 'Medium', 'High'])

    # Make prediction
    if st.button('Predict'):
        result = predict_employee_leave(satisfaction_level, last_evaluation, number_project,
                                        average_monthly_hours, time_spend_company, 
                                        work_accident, promotion_last_5years, salary)
        st.success(f'The employee is likely to {"leave" if result == 1 else "stay"}.')

if __name__ == '__main__':
    main()
