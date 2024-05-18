# ASDCare: Early Detection of Autism Spectrum Disorder

## Overview

ASDCare is a machine learning-based application designed to predict Autism Spectrum Disorder (ASD) early using diverse datasets, including behavioral observations, neuroimaging, and genetic markers. The application achieves high accuracy, contributing to precision medicine in neurodevelopmental disorders.

## Features

- **Behavioral Data Analysis**: Utilizes behavioral observations to develop a robust predictive model.
- **Genetic Marker Screening**: Screens DNA sequences to identify genetic markers associated with ASD.
- **fMRI Analysis**: Employs fMRI data to predict ASD using advanced machine learning algorithms.
- **Personalized Interventions**: Provides personalized intervention plans based on individual characteristics and response patterns.
- **High Accuracy**: Achieves up to 87% accuracy in predicting ASD, outperforming existing methods.

## Installation

To run the ASDCare application, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/RujutaThombre/ASD-Project.git
    cd ASDCare
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the MongoDB database:**
    - Ensure MongoDB is installed and running on your machine.
    - Update the MongoDB connection URL in the script as needed.

4. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Usage

### For Patients

1. **Login**: Enter your credentials to log in as a patient.
2. **Quiz**: Complete the Autism Detection Quiz by answering the questions provided.
3. **Submit**: Submit your responses to receive a prediction of your ASD likelihood.

### For Doctors

1. **Login**: Enter your credentials to log in as a doctor.
2. **Screening Report**: Choose to either enter a DNA sequence or upload an FNA file to screen for genetic markers.
3. **fMRI Scanning Test**: Upload a CSV file containing fMRI data to make a prediction using the fMRI classifier.

## Contributors

- **Rujuta Thombre** - [RujutaThombre](https://github.com/RujutaThombre)
- **Viveka Patil** - [Viveka9Patil](https://github.com/Viveka9Patil)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Libraries and Frameworks**: Streamlit, pandas, scikit-learn, BioPython, Nilearn, MongoDB, Joblib.
- **Datasets**: Thanks to the providers of the behavioral, neuroimaging, and genetic marker datasets used in this study.

## Contact

For any questions or feedback, please contact [rujuta.thombre@gmail.com](mailto:rujuta.thombre@gmail.com).

