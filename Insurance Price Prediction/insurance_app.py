import os
import time
import streamlit as st
import joblib
import pandas as pd

# Load the pretrained model
model = joblib.load(os.path.join(os.getcwd(), 'insurance_model.pkl'))

# Add some CSS styling for better appearance
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("# Insurance Predictions")
st.markdown("### Predict the cost of insurance based on your specifications")
st.markdown("#### Please fill in the details in the sidebar.")

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    
    occupation = st.sidebar.selectbox('Occupation', (
        'Nurse', 'Teacher', 'Photographer', 'Security', 'Cook',
        'Librarian', 'Contractor', 'Midwife', 'Resource Guard', 'Cocobod',
        'Agric Extension', 'Stenographer', 'Sales Officer', 'General Lab',
        'Sanitary Artisan', 'Civil Servant', 'Administrator',
        'Anesthesiologist', 'Nbssi', 'Telecom Agent', 'Plumbing',
        'Welding', 'Clerk', 'Engineer', 'Accountant', 'Security Guard',
        'Seamstress', 'Gardener', 'Optical Technician', 'Lecturer',
        'Officer', 'Chef', 'Matron', 'Trading'
    ))
    
    plan = st.sidebar.selectbox('Plan', ('Family Security Plan', 'Education', 'Flexi Child Education', 'Ultimate Life'))
    
    proposals = st.sidebar.date_input('Date of Proposal')
    
    st.sidebar.markdown("#### Financial Details")
    policy_value = st.sidebar.slider('Policy Value', 0.0, 5000.0, 1500.0)
    paid_premium = st.sidebar.slider('Paid Premium', 0.0, 5000.0, 2100.0)
    premium = st.sidebar.slider('Premium', 0.0, 5000.0, 1475.0)
    total_premium = st.sidebar.slider('Total Premium',0.0, 5000.0, 700.0)
    policy_duration_years = st.sidebar.slider('Policy Duration (Years)', 0.0, 50.0, 5.0)
    
    data = {
        'gender': gender,
        'occupation': occupation.upper(),
        'plan': plan.upper(),
        'proposals': proposals,
        'policy_value': policy_value,
        'paid_premium': paid_premium,
        'premium': premium,
        'total_premium': total_premium,
        'policy_duration_years': policy_duration_years,
        'policy_duration_months': policy_duration_years * 12   # To give us an approximate duration policy in months
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input
st.subheader('User Input features')
def highlight_max(s):
    '''
    Highlight the maximum value in each column.
    '''
    styles = []
    for col in s.index:
        if s[col] == s.max():
            styles.append('background-color: #f0ad4e; color: #fff; font-weight: bold')  # Highlight max value with a shade of orange
        else:
            styles.append('')
    return styles

styled_df = df.style.apply(highlight_max, axis=0)

# Apply additional styling to the dataframe
styled_df.set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#5bc0de'), ('color', 'white'), ('font-size', '17px'), ('font-weight', 'bold')]},  # Header styling
    {'selector': 'td', 'props': [('font-size', '15px')]},  # Cell content styling
])

# Display the styled dataframe
st.dataframe(styled_df)

if st.sidebar.button('Get Predictions'):
    with st.spinner('Calculating...'):
        time.sleep(1)  # Simulate the loading time

        # Prediction
        prediction = model.predict(df)

        # Display prediction
        st.subheader('Prediction')
        st.write(f'The predicted insurance cost is: GH₵{prediction[0]:.2f}')
        
        # Display a success message
        st.success('Prediction generated successfully!')

# Additional improvement: conditional formatting for user input display

