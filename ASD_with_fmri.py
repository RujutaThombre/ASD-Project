import streamlit as st
import joblib
import io
import ast
import difflib
import pandas as pd
from pymongo import MongoClient
import pickle
from Add_logo import add_logo
from Bio import SeqIO
import os
import numpy as np
from sklearn import preprocessing
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec


patient_pages = ["Home", "About", "Quiz", "FAQ"]

# Define page list for doctors
doctor_pages = ["Home", "About", "FAQ", "Screening Report", "fMRI Scanning Test"]

# Streamlit web application setup
st.set_page_config(
    page_title='Welcome to ASDCare!',
    layout='wide',
    page_icon='ASD1.png',
)
st.markdown(
    """
    <style>
    .stApp {
        background-color:  #F0F8FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

add_logo(logo_url='ASD1.png')

# Function to calculate sequence similarity
def calculate_similarity_percentage(sequence1, sequence2):
    matcher = difflib.SequenceMatcher(None, sequence1, sequence2)
    similarity_percentage = matcher.ratio() * 100
    return similarity_percentage


# Function for the screening model
def screening_model(dataset, user_sequence):
    max_similarity = 0

    for stored_sequence in dataset:
        similarity = calculate_similarity_percentage(stored_sequence, user_sequence)
        max_similarity = max(max_similarity, similarity)

    return max_similarity

def scan_for_markers(input_file_path, marker_directory):
    """
    Scan the input Fasta file for the presence of sequences from multiple markers
    located in the specified marker directory.

    Parameters:
    - input_file_path: The path to the input Fasta file.
    - marker_directory: The directory containing Fasta files for multiple markers.

    Returns:
    - A list of markers detected in the input file.
    """
    detected_markers = []
    
    for marker_file in os.listdir(marker_directory):
        marker_file_path = os.path.join(marker_directory, marker_file)
        
        with open(marker_file_path, "r") as marker_fasta:
            marker_records = set(record.seq for record in SeqIO.parse(marker_fasta, "fasta"))
        
        with open(input_file_path, "r") as input_fasta:
            input_sequences = set(record.seq for record in SeqIO.parse(input_fasta, "fasta"))
            
        if any(marker_sequence in input_sequences for marker_sequence in marker_records):
            detected_markers.append(marker_file)
    
    return detected_markers

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
dataset_path = 'ASD.csv'
dataset = pd.read_csv(dataset_path)['FASTA Sequence'].tolist()





mongo_url = "mongodb://localhost/"  

# Establish a connection to MongoDB
client = MongoClient(mongo_url)

# Choose a database (create it if it doesn't exist)
database_name = "db_ASD"  
db = client[database_name]
collection = db["test"]

# Load your machine learning model from a pickle file
with open('svm_classifier (1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def binary_encoding(value, positive_values=['always', 'usually', 'sometimes', 'yes']):
    try:
        # Convert value to lowercase
        lowercase_value = str(value).lower()
        # Check if the value is a numeric type or in positive_values
        if lowercase_value.isnumeric() or lowercase_value in positive_values:
            return 1
        else:
            return 0
    except ValueError:
        return 0

# Updated Prediction function
def predict(quiz_responses):
    # Map user responses to binary values using binary_encoding function
    binary_question_2 = binary_encoding(quiz_responses[1])
    binary_question_3 = binary_encoding(quiz_responses[2])
    binary_question_4 = binary_encoding(quiz_responses[3])
    binary_question_5 = binary_encoding(quiz_responses[4])
    binary_question_6 = binary_encoding(quiz_responses[5])
    binary_question_7 = binary_encoding(quiz_responses[6])
    binary_question_8 = binary_encoding(quiz_responses[7])
    binary_question_9 = binary_encoding(quiz_responses[8])
    binary_question_10 = binary_encoding(quiz_responses[9], positive_values=['yes'])
    binary_question_11 = binary_encoding(quiz_responses[10], positive_values=['yes'])
    binary_question_12 = binary_encoding(quiz_responses[11], positive_values=['yes'])

    # Calculate cumulative score
    cumulative_score = (
        binary_question_2 + binary_question_3 + binary_question_4 +
        binary_question_5 + binary_question_6 + binary_question_7 +
        binary_question_8 + binary_question_9 + binary_question_10 +
        binary_question_11 + binary_question_12 
    )

    color_low = "red"  
    color_high = "green"  
    
    # Normalize cumulative score to be between 0 and 1
    normalized_score = cumulative_score / 11.0  

    # Set the progress bar color based on cumulative score
    progress_bar_color = color_high 
    if cumulative_score > 5:
        progress_bar_color = color_high
        result_message = "The user has a high probability of ASD. Your Score is: " + str(cumulative_score)
    else:
        progress_bar_color = color_low
        result_message = "The user might not have ASD. Your Score is: " + str(cumulative_score)

    # Display the progress bar with inline HTML style
    st.markdown(
        f"""
        <div style="background: linear-gradient(to right, {color_low}, {color_high}); padding: 5px;">
            <div style="background-color: {progress_bar_color}; width: {normalized_score * 100}%; color: white; text-align: center;">
                {int(normalized_score * 100)}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display the result message
    st.markdown(f'<p class="medium-font">{result_message}</p>', unsafe_allow_html=True)

def navigate(step):
  st.session_state.current_question += step
  st.experimental_rerun()


# Load your dataset (replace 'your_dataset.csv' with the actual file path)
dataset_path = 'ASD.csv'
dataset = pd.read_csv(dataset_path)['FASTA Sequence'].tolist()

# Initialize session_state.logged_in if not defined
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.sidebar.image('ASD1.png', width=200)
st.sidebar.header("Navigation")


st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #ffffe0;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session_state.logged_in if not defined
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Initialize session_state.user_type if not defined
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# Define user types (patient and doctor)
user_types = ["Patient", "Doctor"]

page_list = patient_pages  # Default to patient pages
if st.session_state.user_type == "Doctor":
    page_list = doctor_pages

# Display only the dropdown without additional content
page = st.sidebar.selectbox("Select a Page", page_list)

# Define login credentials for patients and doctors
patient_credentials = {"patient123": "pass123"}
doctor_credentials = {"doctor123": "pass123"}

if page == "Home":    
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    user_type = st.selectbox("Select User Type", user_types)
    
    if st.button("Login"):
        if user_type == "Patient" and username in patient_credentials and password == patient_credentials[username]:
            st.success("Logged in as a Patient: " + username)
            st.session_state.logged_in = True
            st.session_state.user_type = "Patient"
        elif user_type == "Doctor" and username in doctor_credentials and password == doctor_credentials[username]:
            st.success("Logged in as a Doctor: " + username)
            st.session_state.logged_in = True
            st.session_state.user_type = "Doctor"
        else:
            st.error("Invalid credentials. Please try again.")

if st.session_state.logged_in:
    if st.session_state.user_type == "Patient":
        # Patient-specific content
        if page == "About":
            st.title("About us!")

        elif page == "Quiz":
          st.title("Autism Detection Quiz")

        # Define the quiz questions and possible answers
          questions = [
            "ID. Unique Patient ID:",
            "A1. Does your child avoid making eye contact when called by their name?",
            "A2. Do you encounter challenges in establishing eye contact with your child",
            "A3. Does your child abstain from employing pointing gestures to communicate desires or requests? (e.g., indicating a toy that is out of reach)",
            "A4. Is your child not inclined to point to share interests with you? (e.g., directing attention to an interesting sight)",
            "A5. Does your child abstain from engaging in pretend play? (e.g., caring for dolls or simulating phone conversations)",
            "A6. Does your child not track or follow your gaze or direction of attention?",
            "A7. In situations where you or a family member displays visible distress, does your child refrain from exhibiting signs of comfort? (e.g., through actions like stroking hair or offering hugs)",
            "A8. Would you characterize your child's initial words as atypical?",
            "A9. Does your child eschew the use of simple gestures? (e.g., waving goodbye)",
            "A10. Does your child engage in prolonged staring at objects with no discernible purpose?",
            "A11. Have you or your child experienced jaundice, or is there a current occurrence of jaundice in you or your child?",
            "What is your or your child's age?",
            "Please enter your gender:",
            "Please enter your ethinicity:",
            "Previous Autism Diagnosis:",
            "Country of residence:",
            "Have you used our application before?",
            "What was your result?",
            "Please enter your age description:",
            "What is your relation to the patient?"
        ]

          answer_options = ["Always", "Usually", "Sometimes", "Rarely", "Never"]

        # Initialize session state for quiz
          if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
            st.session_state.answers = [None] * len(questions)

        # Display the current question
          st.markdown(questions[st.session_state.current_question])

        # Display input or radio buttons based on the question type
          if questions[st.session_state.current_question] == "ID. Unique Patient ID:":
                 st.session_state.answers[st.session_state.current_question] = st.text_input("Enter your Unique Patient ID:")

          elif questions[st.session_state.current_question] == "A1. Does your child avoid making eye contact when called by their name?":
           st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A2. Do you encounter challenges in establishing eye contact with your child":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A3. Does your child abstain from employing pointing gestures to communicate desires or requests? (e.g., indicating a toy that is out of reach)":
           st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A4. Is your child not inclined to point to share interests with you? (e.g., directing attention to an interesting sight)":
             st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A5. Does your child abstain from engaging in pretend play? (e.g., caring for dolls or simulating phone conversations)":
             st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A6. Does your child not track or follow your gaze or direction of attention?":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A7. In situations where you or a family member displays visible distress, does your child refrain from exhibiting signs of comfort? (e.g., through actions like stroking hair or offering hugs)":
             st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=("Always", "Usually", "Sometimes", "Rarely", "Never"))

          elif questions[st.session_state.current_question] == "A8. Would you characterize your child's initial words as atypical?":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=('Yes', 'No'))

          elif questions[st.session_state.current_question] == "A9. Does your child eschew the use of simple gestures? (e.g., waving goodbye)":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer:',options=('Yes', 'No'))

          elif questions[st.session_state.current_question] == "A10. Does your child engage in prolonged staring at objects with no discernible purpose?":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Yes', 'No'))

          elif questions[st.session_state.current_question] == "A11. Have you or your child experienced jaundice, or is there a current occurrence of jaundice in you or your child?":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Yes', 'No'))

        #elif questions[st.session_state.current_question] == "Wohoo! well done with quiz.  Let's go ahead and get to know about a little more!":
            #st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Yay', 'Nay'))

          elif questions[st.session_state.current_question] == "What is your or your child's age?":
            st.session_state.answers[st.session_state.current_question] = st.text_input("Enter your Age:")

          elif questions[st.session_state.current_question] == "Please enter your gender:":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Female', 'Male','Transgender','Other'))

          elif questions[st.session_state.current_question] == "Please enter your ethinicity:":
            st.session_state.answers[st.session_state.current_question] = st.text_input("Enter your ethnicity:")

          elif questions[st.session_state.current_question] == "Previous Autism Diagnosis:":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Positive', 'Negative'))

          elif questions[st.session_state.current_question] == "Country of residence:":
            st.session_state.answers[st.session_state.current_question] = st.text_input("Enter your country of residence:")

          elif questions[st.session_state.current_question] == "Have you used our application before?":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Yes', 'No'))

          elif questions[st.session_state.current_question] == "What was your result?":
            st.session_state.answers[st.session_state.current_question] = st.radio(label='Select your answer',options=('Positive', 'Negative'))

          elif questions[st.session_state.current_question] == "Please enter your age description:":
            st.session_state.answers[st.session_state.current_question] = st.text_area("Enter your age description:")

          elif questions[st.session_state.current_question] == "What is your relation to the patient?":
            st.session_state.answers[st.session_state.current_question] = st.text_input("Enter you relation to the patient:")

          else:
            st.session_state.answers[st.session_state.current_question] = answer_options


        # Display navigation buttons with appropriate spacing
          col1, col2, col3 = st.columns(3)
          if st.session_state.current_question > 0:
            if col1.button("Back"):
                navigate(-1)

          if st.session_state.current_question < len(questions) - 1:
            if col3.button("Next"):
                navigate(1)
          else:
            # Display final page with a submit button
            st.markdown('<p class="medium-font">Quiz Complete</p>', unsafe_allow_html=True)

            # Display a submit button with proper spacing
            if st.button("Submit", key="submit"):

                # Process quiz responses
                result = predict(st.session_state.answers)
                st.success(result)
                # Store quiz responses in MongoDB
                quiz_responses = {
                    "ID": st.session_state.answers[0],
                    "A1_Score": st.session_state.answers[1],
                    "A2_Score": st.session_state.answers[2],
                    # ... (repeat for other answers)
                    "age_description": st.session_state.answers[19],
                    "relation": st.session_state.answers[20]
                }
                collection.insert_one(quiz_responses)
                st.success("Quiz response submitted and stored in MongoDB")

    elif st.session_state.user_type == "Doctor":
        # Doctor-specific content
        if page == "About":
            st.title("Welcome to ASDCare!")
            st.header("About Us")

        elif page == "Screening Report":
            st.title("Autism Screening App")

    # Get user input option
            screening_option = st.radio("Select screening option:", ["Enter DNA sequence", "Upload FNA file"])

            if screening_option == "Enter DNA sequence":
        # Get user input sequence
                user_input_sequence = st.text_area("Enter your DNA sequence:")

                if st.button("Screen for Autism"):
                    if not user_input_sequence:
                        st.warning("Please enter a DNA sequence.")
                    else:
                        similarity_percentage = screening_model(dataset, user_input_sequence)
                        st.success(f"Sequence Similarity: {similarity_percentage:.2f}%")

                # Assume a threshold for determining whether the user might have autism
                        autism_threshold = 80

                        if similarity_percentage >= autism_threshold:
                            st.warning("The user might have a certain percentage of autism.")
                        else:
                            st.info("The user is unlikely to have autism based on the sequence similarity.")

            elif screening_option == "Upload FNA file":
                uploaded_file = st.file_uploader("Upload FNA file", type=["fna"])

                if uploaded_file is not None:
            # Save the uploaded file temporarily
                    temp_fna_path = "temp_fna_file.fna"
                    with open(temp_fna_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())

            # Specify the marker directory
                    marker_directory = "C://Users//HP//Documents//ASDTESTicrtb//genefnas"

            # Scan for markers in the uploaded FNA file
                    detected_markers = scan_for_markers(temp_fna_path, marker_directory)

            # Display the detected markers
                    if detected_markers:
                        st.success("Presence of the following markers detected:")
                        for marker in detected_markers:
                            gene_name = os.path.splitext(marker)[0]  # Extract gene name without extension
                            st.write(f"- {gene_name}")
                    else:
                        st.info("No markers detected in the uploaded FNA file.")

        elif page == "fMRI Scanning Test":
            st.title("fMRI Scanning for ASD") 
            # Additional Streamlit code for making predictions
            st.title("ASD Prediction using fMRI Classifier")

            # Allow user to make predictions
            st.header("Make Predictions")

            # Load the trained fMRI model
            @st.cache(allow_output_mutation=True)
            def load_fmri_model():
                model_path = "C://Users//HP//Documents//ASDTESTicrtb//rf_model_4x4.joblib"
                model = joblib.load(model_path)
                return model

            fmri_model = load_fmri_model()

            # Upload CSV file containing fMRI data
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file is not None:
                # Read the CSV file into a DataFrame
                try:
                    df = pd.read_csv(uploaded_file, header=None)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    st.stop()

                # Check if the DataFrame has the expected shape (4x4)
                if df.shape == (4, 4):
                    # Standardize the features using a new scaler
                    # Note: You should replace this with the actual scaler from your training data
                    # If you don't have a saved scaler, you may need to standardize based on your training data statistics
                    new_scaler = preprocessing.StandardScaler()
                    user_matrix_standardized = new_scaler.fit_transform(df.values.flatten().reshape(1, -1))

                    # Make prediction using the loaded fMRI model
                    try:
                        prediction = fmri_model.predict(user_matrix_standardized)[0]

                        # Display the prediction
                        if prediction == 1:
                            st.markdown("<h3 style='text-align: left; color: black;'>Prediction: The input is classified as ASD, please consult a medical professional</h1>", unsafe_allow_html=True)
                            
                        else:
                            st.write("Prediction: The input is classified as non-ASD.")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                else:
                    st.error("Please upload a CSV file with a 4x4 matrix.")
elif page == "About":
    st.title("Welcome to ASDCare!")
    st.header("About Us")