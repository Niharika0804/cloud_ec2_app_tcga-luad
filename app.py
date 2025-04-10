from flask import Flask, request
import pandas as pd
import torch
from model.model_definition import CrossFusionTransformer
import os

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Directly define feature column lists
image_feature_cols = [f'pca_feature_{i}' for i in range(256)]
clinical_feature_cols = [
    'tobacco_smoking_quit_year', 'pack_years_smoked', 'days_to_diagnosis', 'age_at_diagnosis', 'days_to_year',
    'year_of_diagnosis', 'received_chemotherapy', 'received_radiation_therapy', 'last_contact_days_to_follow_up',
    'tobacco_smoking_status_Current Reformed Smoker for < or = 15 yrs',
    'tobacco_smoking_status_Current Reformed Smoker for > 15 yrs', 'tobacco_smoking_status_Current Smoker',
    'tobacco_smoking_status_Lifelong Non-Smoker', 'tobacco_smoking_status_Not Reported',
    'tobacco_smoking_status_Unknown', 'ajcc_pathologic_stage_Stage I', 'ajcc_pathologic_stage_Stage IA',
    'ajcc_pathologic_stage_Stage IB', 'ajcc_pathologic_stage_Stage IIA', 'ajcc_pathologic_stage_Stage IIB',
    'ajcc_pathologic_stage_Stage IIIA', 'ajcc_pathologic_stage_None', 'laterality_Left', 'laterality_Right',
    'laterality_None', 'tissue_or_organ_of_origin_Lower lobe, lung', 'tissue_or_organ_of_origin_Not Reported',
    'tissue_or_organ_of_origin_Ovary', 'tissue_or_organ_of_origin_Upper lobe, lung',
    'primary_diagnosis_Adenocarcinoma with mixed subtypes', 'primary_diagnosis_Adenocarcinoma, NOS',
    'primary_diagnosis_Granulosa cell tumor, malignant', 'primary_diagnosis_Mucinous adenocarcinoma',
    'primary_diagnosis_Not Reported', 'primary_diagnosis_Papillary adenocarcinoma, NOS', 'prior_malignancy_no',
    'prior_malignancy_yes', 'prior_malignancy_None', 'ajcc_pathologic_t_T1', 'ajcc_pathologic_t_T1a',
    'ajcc_pathologic_t_T1b', 'ajcc_pathologic_t_T2', 'ajcc_pathologic_t_T2a', 'ajcc_pathologic_t_T3',
    'ajcc_pathologic_t_None', 'ajcc_pathologic_n_N0', 'ajcc_pathologic_n_N1', 'ajcc_pathologic_n_N2',
    'ajcc_pathologic_n_None', 'ajcc_pathologic_m_M0', 'ajcc_pathologic_m_MX', 'ajcc_pathologic_m_None',
    'morphology_8140/3', 'morphology_8255/3', 'morphology_8260/3', 'morphology_8480/3', 'morphology_8620/3',
    'morphology_Not Reported', 'residual_disease_R0', 'residual_disease_None', 'icd_10_code_C34.1',
    'icd_10_code_C34.3', 'icd_10_code_C34.30', 'icd_10_code_None', 'therapeutic_agents_Carboplatin, Paclitaxel',
    'therapeutic_agents_Cisplatin, Docetaxel',
    'therapeutic_agents_Erlotinib Hydrochloride, Carboplatin, Paclitaxel, Unknown, Docetaxel',
    'therapeutic_agents_Pemetrexed Disodium, Cisplatin, Bevacizumab',
    'therapeutic_agents_Pemetrexed, Erlotinib, Cisplatin', 'therapeutic_agents_None',
    'ethnicity_not hispanic or latino', 'ethnicity_not reported', 'gender_female', 'gender_male',
    'race_black or african american', 'race_white', 'country_of_residence_at_enrollment_United States'
]
biospecimen_feature_cols = [
    'portion_1_1_number', 'portion_2_1_number', 'num_primary_tumor_samples', 'num_recurrent_tumor_samples',
    'num_normal_samples', 'has_primary_tumor_descriptor', 'has_recurrent_tumor_descriptor', 'num_sample_types',
    'has_tumor_dna', 'has_tumor_rna', 'sample_1_type_Blood Derived Normal', 'sample_1_type_Primary Tumor',
    'sample_1_type_Solid Tissue Normal', 'sample_1_tumor_descriptor_Not Applicable',
    'sample_1_tumor_descriptor_Primary', 'sample_1_composition_Not Reported', 'sample_1_tissue_type_Normal',
    'sample_1_tissue_type_Tumor', 'portion_1_1_is_ffpe_False', 'analyte_1_1_1_type_DNA', 'analyte_1_1_1_type_RNA',
    'analyte_1_1_1_type_Repli-G (Qiagen) DNA', 'analyte_1_1_1_type_Total RNA',
    'sample_2_type_Blood Derived Normal', 'sample_2_type_Primary Tumor', 'sample_2_type_Solid Tissue Normal',
    'sample_2_tumor_descriptor_Not Applicable', 'sample_2_tumor_descriptor_Primary',
    'sample_2_composition_Not Reported', 'sample_2_tissue_type_Normal', 'sample_2_tissue_type_Tumor',
    'portion_2_1_is_ffpe_False', 'analyte_2_1_1_type_DNA', 'analyte_2_1_1_type_RNA',
    'analyte_2_1_1_type_Repli-G (Qiagen) DNA', 'analyte_2_1_1_type_Total RNA', 'analyte_2_1_2_type_DNA',
    'analyte_2_1_2_type_RNA', 'analyte_2_1_2_type_Repli-G (Qiagen) DNA',
    'analyte_2_1_2_type_Repli-G X (Qiagen) DNA', 'sample_3_type_Blood Derived Normal',
    'sample_3_type_Primary Tumor', 'sample_3_type_Solid Tissue Normal', 'sample_3_tumor_descriptor_Not Applicable',
    'sample_3_tumor_descriptor_Primary', 'sample_3_tissue_type_Normal', 'sample_3_tissue_type_Tumor',
    'analyte_1_1_2_type_DNA', 'analyte_1_1_2_type_RNA', 'analyte_1_1_2_type_Repli-G (Qiagen) DNA'
]

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossFusionTransformer(image_dim=len(image_feature_cols), clinical_dim=len(clinical_feature_cols), biospecimen_dim=len(biospecimen_feature_cols)).to(device)
try:
    model.load_state_dict(torch.load('model/model.pth', map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
<html>
<head>
    <title>Multimodal Prediction Service</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        div {
            margin-bottom: 15px;
        }
        label {
            display: inline-block;
            margin-right: 10px;
            font-weight: bold;
            color: #555;
        }
        input[type="file"] {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        input[type="radio"] {
            margin-right: 5px;
        }
        #combined_data_upload {
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        input[type="submit"] {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        h2 {
            color: #28a745;
            margin-top: 20px;
            text-align: center;
        }
        p[style*="color: red;"] {
            color: #dc3545;
            margin-top: 10px;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Multimodal Prediction Service</h1>

    <form method="POST" action="/predict" enctype="multipart/form-data">
        <div>
            <input type="radio" id="combined_data" name="upload_option" value="combined_data" checked>
            <label for="combined_data">Upload Preprocessed Combined Data CSV</label><br>
        </div>

        <div id="combined_data_upload">
            <label for="preprocessed_combined">Preprocessed Combined Clinical, Biospecimen, and Image Feature CSV:</label>
            <input type="file" name="preprocessed_combined" required><br><br>
        </div>

        <input type="submit" value="Predict">
    </form>

    <script>
        // Immediately show only the combined data upload section and check its radio button
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('combined_data').checked = true;
            document.getElementById('combined_data_upload').style.display = 'block';
            const threeFilesDiv = document.getElementById('three_files_upload');
            if (threeFilesDiv) threeFilesDiv.style.display = 'none';
            const twoFilesDiv = document.getElementById('two_files_upload');
            if (twoFilesDiv) twoFilesDiv.style.display = 'none';
        });
    </script>

</body>
</html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Prediction output: 1 (alive)"

    preprocessed_combined_file = request.files.get('preprocessed_combined')

    if preprocessed_combined_file:
        combined_path = os.path.join(UPLOAD_FOLDER, preprocessed_combined_file.filename)
        try:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            preprocessed_combined_file.save(combined_path)
        except Exception as e:
            return f'Error saving preprocessed combined file: {e}'

        try:
            df_preprocessed = pd.read_csv(combined_path)
            image_features = df_preprocessed[image_feature_cols].values
            clinical_features = df_preprocessed[clinical_feature_cols].values
            biospecimen_features = df_preprocessed[biospecimen_feature_cols].values

            image_tensor = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0).to(device) if image_features.size > 0 else None
            clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0).to(device) if clinical_features.size > 0 else None
            biospecimen_tensor = torch.tensor(biospecimen_features, dtype=torch.float32).unsqueeze(0).to(device) if biospecimen_features.size > 0 else None

            with torch.no_grad():
                output = model(image_tensor, clinical_tensor, biospecimen_tensor)
                prediction = output.squeeze().item()
            return f"Prediction: {round(prediction, 4)}"

        except KeyError as e:
            return f"Error: Column '{e}' not found in the uploaded CSV file. Please ensure the CSV contains all the necessary preprocessed columns."
        except Exception as e:
            return f'Error processing preprocessed combined data: {e}'
    else:
        return "Please upload a preprocessed CSV file."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)