Mental Health Analyser

This project is a machine learning-based system designed to detect mental health conditions based on various user inputs. The system employs a Random Forest classifier to predict the likelihood of mental health issues such as anxiety and panic attacks. The project integrates a Flask API for prediction services and a Tkinter-based GUI application for user interaction.

Features
Machine Learning Model: Utilizes a Random Forest Classifier for accurate prediction of mental health conditions.
Flask API: A simple and efficient API built using Flask to handle prediction requests and serve responses.
Tkinter GUI: A user-friendly graphical interface built with Tkinter to collect user inputs such as age, CGPA, anxiety levels, panic attacks, and sleep schedule.
Technologies Used
Python: Core language for the project.
Scikit-learn: Library used for implementing the Random Forest model.
Flask: Micro web framework for building the API.
Tkinter: Python's standard GUI toolkit for creating the desktop application.
Random Forest Classifier: The machine learning model used for prediction.

How to Run
Clone the Repository:

bash
git clone https://github.com/your-username/mental-health-analyser.git
cd mental-health-analyser
Install Dependencies:

bash
pip install -r requirements.txt
Run the Flask API:

bash
python api.py
Run the Tkinter GUI Application:

bash
python gui.py
Use the Application:

Enter the required inputs such as age, CGPA, anxiety levels, panic attacks, and sleep schedule into the GUI.
Click the "Predict" button to get the prediction.


