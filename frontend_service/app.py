import gradio as gr
import requests
import pandas as pd
import os

# Configuration for Service URLs
DATA_MANAGEMENT_SERVICE_URL = os.getenv("DATA_MANAGEMENT_SERVICE_URL", "http://data_management_service:8001")
DISEASE_PREDICTOR_SERVICE_URL = os.getenv("DISEASE_PREDICTOR_SERVICE_URL", "http://disease_predictor_service:8002")

def get_patients():
    """Fetches patient list from the data management service."""
    try:
        response = requests.get(f"{DATA_MANAGEMENT_SERVICE_URL}/patients/")
        response.raise_for_status()
        patients = response.json()
        if not patients:
            return pd.DataFrame(), "No patients found or empty response."
        # Convert to DataFrame for easier display in Gradio
        df = pd.DataFrame(patients)
        # Select and reorder columns for display if needed
        if not df.empty:
            df_display = df[['patient_id', 'date_of_birth', 'created_at', 'updated_at']]
            return df_display, "Patients loaded successfully."
        return pd.DataFrame(), "No patient data to display."
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Error fetching patients: {e}"
    except Exception as e:
        return pd.DataFrame(), f"An unexpected error occurred: {e}"

def get_health_records(patient_id: str):
    """Fetches health records for a specific patient."""
    if not patient_id or not patient_id.strip():
        return pd.DataFrame(), "Please enter a Patient ID."
    try:
        response = requests.get(f"{DATA_MANAGEMENT_SERVICE_URL}/health_records/patient/{patient_id.strip()}")
        response.raise_for_status()
        records = response.json()
        if not records:
            return pd.DataFrame(), f"No health records found for patient ID: {patient_id}"
        
        # Process records for display. The 'data' field is a dict.
        # We can flatten it or select key fields.
        processed_records = []
        for rec in records:
            base_info = {
                "record_id": rec.get("record_id"),
                "record_type": rec.get("record_type"),
                "record_date": rec.get("record_date"),
                "source": rec.get("source")
            }
            # Add fields from the nested 'data' dictionary
            if isinstance(rec.get("data"), dict):
                base_info.update(rec["data"])
            processed_records.append(base_info)
            
        df = pd.DataFrame(processed_records)
        return df, f"Health records for {patient_id} loaded."
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Error fetching health records: {e}"
    except Exception as e:
        return pd.DataFrame(), f"An unexpected error occurred while fetching records: {e}"

def predict_diabetes_from_inputs(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    """Submits features to the disease predictor service and returns prediction."""
    if not all([isinstance(val, (int, float)) for val in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]):
        return "Invalid input: All features must be numbers.", "", ""
        
    input_data = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": int(age)
    }
    try:
        response = requests.post(f"{DISEASE_PREDICTOR_SERVICE_URL}/predict/typ2_diabetes", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        
        outcome = "Positive for Diabetes" if prediction_result.get("predicted_outcome") == 1 else "Negative for Diabetes"
        prob_0 = prediction_result.get("probability_outcome_0", 0.0) * 100
        prob_1 = prediction_result.get("probability_outcome_1", 0.0) * 100
        
        status_msg = "Prediction successful."
        prediction_text = f"Prediction: {outcome}\nProbability (Negative): {prob_0:.2f}%\nProbability (Positive): {prob_1:.2f}%"
        return prediction_text, "", status_msg # Clear other outputs if successful

    except requests.exceptions.RequestException as e:
        err_msg = f"Error calling prediction service: {e}. Response: {e.response.text if e.response else 'No response'}"
        return "", "", err_msg # Update status message with error
    except Exception as e:
        err_msg = f"An unexpected error occurred during prediction: {e}"
        return "", "", err_msg

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Diabetes Risk Assessment & Patient Data Viewer")

    with gr.Tabs():
        with gr.TabItem("Diabetes Prediction"):
            gr.Markdown("Enter patient metrics below to predict Type 2 Diabetes risk.")
            with gr.Row():
                with gr.Column():
                    preg_input = gr.Number(label="Pregnancies", value=0)
                    glucose_input = gr.Number(label="Glucose (mg/dL)", value=120)
                    bp_input = gr.Number(label="Blood Pressure (mm Hg)", value=70)
                    skin_input = gr.Number(label="Skin Thickness (mm)", value=20)
                with gr.Column():
                    insulin_input = gr.Number(label="Insulin (mu U/ml)", value=80)
                    bmi_input = gr.Number(label="BMI (kg/m²)", value=25.0)
                    dpf_input = gr.Number(label="Diabetes Pedigree Function", value=0.5)
                    age_input = gr.Number(label="Age (years)", value=30)
            
            predict_btn = gr.Button("Get Prediction")
            prediction_status_message = gr.Textbox(label="Status", interactive=False, lines=2)
            prediction_output_text = gr.Textbox(label="Prediction Result", interactive=False, lines=3)
            
            predict_btn.click(
                fn=predict_diabetes_from_inputs,
                inputs=[
                    preg_input, glucose_input, bp_input, skin_input, 
                    insulin_input, bmi_input, dpf_input, age_input
                ],
                outputs=[prediction_output_text, gr.Textbox(), prediction_status_message] # Third output for status
            )

        with gr.TabItem("View Patients"):
            with gr.Row():
                load_patients_btn = gr.Button("Load All Patients")
            patients_status_message = gr.Textbox(label="Status", interactive=False)
            patients_df_output = gr.DataFrame(label="Patients List", interactive=False)
            
            load_patients_btn.click(
                fn=get_patients, 
                inputs=None, 
                outputs=[patients_df_output, patients_status_message]
            )

        with gr.TabItem("View Health Records by Patient ID"):
            with gr.Row():
                patient_id_input = gr.Textbox(label="Enter Patient ID to view their Health Records")
            with gr.Row():
                load_records_btn = gr.Button("Load Health Records")
            records_status_message = gr.Textbox(label="Status", interactive=False)
            records_df_output = gr.DataFrame(label="Health Records", interactive=False)

            load_records_btn.click(
                fn=get_health_records, 
                inputs=[patient_id_input], 
                outputs=[records_df_output, records_status_message]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)