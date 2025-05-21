import gradio as gr
import requests
import pandas as pd
import os
from datetime import date # For date validation

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
    try:
        # Ensure inputs are numbers before creating the payload
        payload_values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        if not all(isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.','',1).isdigit()) for val in payload_values):
            return "", "Invalid input: All features must be valid numbers."

        input_data = {
            "Pregnancies": int(float(pregnancies)),
            "Glucose": float(glucose),
            "BloodPressure": float(blood_pressure),
            "SkinThickness": float(skin_thickness),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": int(float(age))
        }
        response = requests.post(f"{DISEASE_PREDICTOR_SERVICE_URL}/predict/typ2_diabetes", json=input_data)
        response.raise_for_status()
        prediction_result = response.json()
        
        outcome = "Positive for Diabetes" if prediction_result.get("predicted_outcome") == 1 else "Negative for Diabetes"
        prob_0 = prediction_result.get("probability_outcome_0", 0.0) * 100
        prob_1 = prediction_result.get("probability_outcome_1", 0.0) * 100
        
        status_msg = "Prediction successful."
        prediction_text = f"Prediction: {outcome}\nProbability (Negative): {prob_0:.2f}%\nProbability (Positive): {prob_1:.2f}%"
        return prediction_text, status_msg

    except requests.exceptions.RequestException as e:
        err_msg = f"Error calling prediction service: {e}. Response: {e.response.text if e.response else 'No response'}"
        return "", err_msg
    except ValueError as e: # Catch errors from int/float conversion
        return "", f"Invalid input: Please ensure all feature values are numbers. Error: {e}"
    except Exception as e:
        err_msg = f"An unexpected error occurred during prediction: {e}"
        return "", err_msg

def create_new_patient(date_of_birth_str: str):
    """Creates a new patient record."""
    if not date_of_birth_str or not date_of_birth_str.strip():
        return "Please enter a Date of Birth."
    try:
        # Validate date format (basic validation)
        date.fromisoformat(date_of_birth_str.strip())
        payload = {"date_of_birth": date_of_birth_str.strip()}
        response = requests.post(f"{DATA_MANAGEMENT_SERVICE_URL}/patients/", json=payload)
        response.raise_for_status()
        created_patient = response.json()
        return f"Patient created successfully. ID: {created_patient.get('patient_id')}"
    except ValueError:
        return "Invalid Date of Birth format. Please use YYYY-MM-DD."
    except requests.exceptions.RequestException as e:
        return f"Error creating patient: {e}. Response: {e.response.text if e.response else 'No response'}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def create_new_health_record(patient_id, record_type, source, 
                             pregnancies, glucose, blood_pressure, skin_thickness, 
                             insulin, bmi, dpf, outcome):
    """Creates a new health record for a patient."""
    if not patient_id or not patient_id.strip():
        return "Patient ID is required."
    if not record_type or not record_type.strip():
        return "Record Type is required."

    # Helper to safely convert to float/int or return None if blank/invalid
    def safe_convert(value, target_type):
        if isinstance(value, str) and not value.strip(): return None
        if value is None: return None
        try:
            return target_type(value)
        except (ValueError, TypeError):
            return None # Or raise an error / return specific message
            
    data_payload = {
        "Pregnancies": safe_convert(pregnancies, int),
        "Glucose": safe_convert(glucose, float),
        "BloodPressure": safe_convert(blood_pressure, float), # Model might expect float
        "SkinThickness": safe_convert(skin_thickness, float),
        "Insulin": safe_convert(insulin, float),
        "BMI": safe_convert(bmi, float),
        "DiabetesPedigreeFunction": safe_convert(dpf, float),
        "Outcome": safe_convert(outcome, int)
    }
    # Remove keys with None values from data_payload to send only provided data
    data_payload = {k: v for k, v in data_payload.items() if v is not None}

    payload = {
        "patient_id": patient_id.strip(),
        "record_type": record_type.strip(),
        "source": source.strip() if source and source.strip() else None,
        "data": data_payload
    }
    
    try:
        response = requests.post(f"{DATA_MANAGEMENT_SERVICE_URL}/health_records/", json=payload)
        response.raise_for_status()
        created_record = response.json()
        return f"Health Record created successfully. ID: {created_record.get('record_id')}"
    except requests.exceptions.RequestException as e:
        return f"Error creating health record: {e}. Response: {e.response.text if e.response else 'No response'}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Diabetes Risk Assessment & Patient Data Viewer")

    with gr.Tabs():
        with gr.TabItem("Diabetes Prediction"):
            gr.Markdown("Enter patient metrics below to predict Type 2 Diabetes risk.")
            with gr.Row():
                with gr.Column():
                    preg_input = gr.Number(label="Pregnancies", value=0, minimum=0, precision=0)
                    glucose_input = gr.Number(label="Glucose (mg/dL)", value=120)
                    bp_input = gr.Number(label="Blood Pressure (mm Hg)", value=70)
                    skin_input = gr.Number(label="Skin Thickness (mm)", value=20)
                with gr.Column():
                    insulin_input = gr.Number(label="Insulin (mu U/ml)", value=80)
                    bmi_input = gr.Number(label="BMI (kg/m²)", value=25.0)
                    dpf_input = gr.Number(label="Diabetes Pedigree Function", value=0.5)
                    age_input = gr.Number(label="Age (years)", value=30, minimum=0, precision=0)
            
            predict_btn = gr.Button("Get Prediction")
            prediction_status_message = gr.Textbox(label="Status", interactive=False, lines=2)
            prediction_output_text = gr.Textbox(label="Prediction Result", interactive=False, lines=3)
            
            predict_btn.click(
                fn=predict_diabetes_from_inputs,
                inputs=[
                    preg_input, glucose_input, bp_input, skin_input, 
                    insulin_input, bmi_input, dpf_input, age_input
                ],
                outputs=[prediction_output_text, prediction_status_message]
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
                patient_id_input_view = gr.Textbox(label="Enter Patient ID to view their Health Records") # Renamed to avoid clash
            with gr.Row():
                load_records_btn = gr.Button("Load Health Records")
            records_status_message = gr.Textbox(label="Status", interactive=False)
            records_df_output = gr.DataFrame(label="Health Records", interactive=False)

            load_records_btn.click(
                fn=get_health_records, 
                inputs=[patient_id_input_view], 
                outputs=[records_df_output, records_status_message]
            )
        
        with gr.TabItem("Create New Patient"):
            gr.Markdown("Enter patient details to create a new patient record.")
            dob_input = gr.Textbox(label="Date of Birth (YYYY-MM-DD)", placeholder="e.g., 1990-01-30")
            create_patient_btn = gr.Button("Create Patient")
            create_patient_status = gr.Textbox(label="Status", interactive=False)

            create_patient_btn.click(
                fn=create_new_patient,
                inputs=[dob_input],
                outputs=[create_patient_status]
            )

        with gr.TabItem("Create New Health Record"):
            gr.Markdown("Enter details to create a new health record for an existing patient.")
            hr_patient_id_input = gr.Textbox(label="Patient ID (must exist)")
            hr_record_type_input = gr.Textbox(label="Record Type", placeholder="e.g., DiabetesScreeningData, LabResult")
            hr_source_input = gr.Textbox(label="Source (Optional)", placeholder="e.g., EHR, ManualEntry")
            
            gr.Markdown("### Health Record Data (Enter relevant values)")
            with gr.Row():
                with gr.Column():
                    hr_preg_input = gr.Number(label="Pregnancies", minimum=0, precision=0)
                    hr_glucose_input = gr.Number(label="Glucose")
                    hr_bp_input = gr.Number(label="Blood Pressure")
                    hr_skin_input = gr.Number(label="Skin Thickness")
                with gr.Column():
                    hr_insulin_input = gr.Number(label="Insulin")
                    hr_bmi_input = gr.Number(label="BMI")
                    hr_dpf_input = gr.Number(label="Diabetes Pedigree Function")
                    hr_outcome_input = gr.Number(label="Outcome (0 or 1)", minimum=0, maximum=1, precision=0)

            create_hr_btn = gr.Button("Create Health Record")
            create_hr_status = gr.Textbox(label="Status", interactive=False)

            create_hr_btn.click(
                fn=create_new_health_record,
                inputs=[
                    hr_patient_id_input, hr_record_type_input, hr_source_input,
                    hr_preg_input, hr_glucose_input, hr_bp_input, hr_skin_input,
                    hr_insulin_input, hr_bmi_input, hr_dpf_input, hr_outcome_input
                ],
                outputs=[create_hr_status]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)