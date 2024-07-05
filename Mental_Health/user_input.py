# user_input_with_gui.py

import numpy as np
from joblib import load
import tkinter as tk
from tkinter import messagebox


model = load('mental_health_model.joblib')


def predict_mental_health(age, gender, cgpa, anxiety, panic_attack, sleep_schedule):
    try:
        cgpa = np.mean([float(val) for val in cgpa.split('-')])
    except (ValueError, AttributeError):
        return 'Invalid input for CGPA', 0.0

    user_data = np.array([[age, cgpa, anxiety, panic_attack]])
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[0][1] 
    
   
    prediction_label = "Mentally Unstable At The Moment" if prediction[0] == 1 else "Having Mild Depression"

    return prediction_label, probability


def on_predict_button_click():
    user_age = float(entry_age.get())
    user_gender = int(entry_gender.get())
    user_cgpa = entry_cgpa.get()
    user_anxiety = int(entry_anxiety.get())
    user_panic_attack = int(entry_panic_attack.get())
    user_sleep_schedule = int(entry_sleep_schedule.get())

    result, probability = predict_mental_health(user_age, user_gender, user_cgpa, user_anxiety, user_panic_attack, user_sleep_schedule)
    result_text = f'Based on the model, you are: {result}\nConfidence: {probability * 100:.2f}%'
    messagebox.showinfo('Prediction Result', result_text)


window = tk.Tk()
window.title('Mental Health Predictor')

label_age = tk.Label(window, text='Enter your age:')
label_age.pack()

entry_age = tk.Entry(window)
entry_age.pack()

label_gender = tk.Label(window, text='Choose Your Gender (1 for Male, 0 for Female, 2 for Prefer Not To Say):')
label_gender.pack()

entry_gender = tk.Entry(window)
entry_gender.pack()

label_cgpa = tk.Label(window, text='Enter your CGPA (in the format X-Y):')
label_cgpa.pack()

entry_cgpa = tk.Entry(window)
entry_cgpa.pack()

label_anxiety = tk.Label(window, text='Do you have anxiety? (1 for Yes, 0 for No, 2 for sometimes):')
label_anxiety.pack()

entry_anxiety = tk.Entry(window)
entry_anxiety.pack()

label_panic_attack = tk.Label(window, text='Do you have panic attack? (1 for Yes, 0 for No, 2 for sometimes):')
label_panic_attack.pack()

entry_panic_attack = tk.Entry(window)
entry_panic_attack.pack()

label_sleep_schedule = tk.Label(window, text='How Is Your Sleep Schedule (1 for Regular, 0 for Poor, 2 for Irregular):')
label_sleep_schedule.pack()

entry_sleep_schedule = tk.Entry(window)
entry_sleep_schedule.pack()


predict_button = tk.Button(window, text='Predict', command=on_predict_button_click)
predict_button.pack()


window.mainloop()
