https://drugeffectivenesspredict-os562tblx8rhmhbeirh68z.streamlit.app/

# Drug Effectiveness Predictor

This project is a web application that predicts the effectiveness of a drug based on patient details and medication information. It uses a machine learning model trained on a dataset of drug reviews.

## Features

*   Predicts drug effectiveness score based on:
    *   Patient's age
    *   Medical condition
    *   Treatment duration
    *   Drug name
    *   Dosage
*   Simple and intuitive web interface built with Streamlit.
*   Provides a score and a qualitative assessment (High, Moderate, or Low effectiveness).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Drug_Effectiveness_pred.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Drug_Effectiveness_pred
    ```
3.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4.  Activate the virtual environment:
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Make sure you have the trained model (`drug_model.pkl`) and scaler (`scalar.pkl`) files in the project's root directory.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).
4.  Enter the patient and medication details in the web form and click "Predict Effectiveness".

## Model Training

The machine learning model was trained using a Jupyter Notebook (`Drug.ipynb`). The notebook details the following steps:

1.  **Data Loading and Preprocessing**: The dataset (`real_drug_dataset.csv`) is loaded, and categorical features are converted into a numerical format.
2.  **Exploratory Data Analysis (EDA)**: The notebook includes visualizations to understand the data distribution and correlations between features.
3.  **Feature Selection**: A Random Forest Regressor is used to identify the most important features for predicting the improvement score.
4.  **Model Training and Hyperparameter Tuning**: Several regression models are trained and evaluated. A Random Forest Regressor is selected and fine-tuned using `RandomizedSearchCV`.
5.  **Model Saving**: The best-performing model and the scaler are saved as `drug_model.pkl` and `scalar.pkl`.

## Technologies Used

*   **Python**: The core programming language.
*   **Streamlit**: For building the web application.
*   **Scikit-learn**: For machine learning and data preprocessing.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Jupyter Notebook**: For model development and experimentation.
