import pandas as pd
import numpy as np
from joblib import load  # For loading the preprocessor
import os
import torch

def create_biospecimen_features_final(df):
    feature_df = pd.DataFrame(df['submitter_id'].unique(), columns=['submitter_id'])
    feature_df = feature_df.set_index('submitter_id')
    df = df.set_index('submitter_id')

    for index in feature_df.index:
        patient_data = df.loc[index]
        if isinstance(patient_data, pd.Series):
            patient_data = pd.DataFrame([patient_data])

        # Counts of different sample types
        feature_df.loc[index, 'num_primary_tumor_samples'] = sum(x == 'Primary Tumor' for x in patient_data[['sample_1_type', 'sample_2_type', 'sample_3_type']].fillna('').values.flatten())
        feature_df.loc[index, 'num_recurrent_tumor_samples'] = sum(x == 'Recurrent Tumor' for x in patient_data[['sample_1_type', 'sample_2_type', 'sample_3_type']].fillna('').values.flatten())
        feature_df.loc[index, 'num_normal_samples'] = sum(x in ['Solid Tissue Normal', 'Blood Derived Normal'] for x in patient_data[['sample_1_type', 'sample_2_type', 'sample_3_type']].fillna('').values.flatten())

        # Presence of specific tumor descriptors (can be kept as binary indicators)
        primary_tumor_present = any(x == 'Primary' for x in patient_data[['sample_1_tumor_descriptor', 'sample_2_tumor_descriptor', 'sample_3_tumor_descriptor']].fillna('').values.flatten())
        recurrent_tumor_present = any(x == 'Recurrence' for x in patient_data[['sample_1_tumor_descriptor', 'sample_2_tumor_descriptor', 'sample_3_tumor_descriptor']].fillna('').values.flatten())
        feature_df.loc[index, 'has_primary_tumor_descriptor'] = int(primary_tumor_present)
        feature_df.loc[index, 'has_recurrent_tumor_descriptor'] = int(recurrent_tumor_present)

        # Count of different sample types (overall)
        sample_types = pd.unique(patient_data[['sample_1_type', 'sample_2_type', 'sample_3_type']].fillna('').values.flatten())
        feature_df.loc[index, 'num_sample_types'] = len([st for st in sample_types if st != ''])

        # Presence of specific analytes in tumor samples (keep as is for now)
        def has_analyte_in_tumor(analyte, patient_data):
            for i in range(1, 4):
                tumor_desc_col = f'sample_{i}_tumor_descriptor'
                analyte_cols = [col for col in patient_data.columns if col.startswith(f'analyte_{i}_') and 'type' in col]
                if tumor_desc_col in patient_data.columns and any(patient_data[tumor_desc_col].fillna('').values == 'Primary'):
                    for analyte_col in analyte_cols:
                        if analyte_col in patient_data.columns and analyte in str(patient_data[analyte_col].fillna('').values):
                            return 1
            return 0

        feature_df.loc[index, 'has_tumor_dna'] = has_analyte_in_tumor('DNA', patient_data)
        feature_df.loc[index, 'has_tumor_rna'] = has_analyte_in_tumor('RNA', patient_data)

    feature_df = feature_df.reset_index()
    return feature_df

def preprocess_clinical_biospecimen_data(clinical_path, biospecimen_path=None, feature_cols_path='model/feature_cols.pkl', preprocessor_path='model/preprocessor.joblib'):
    """
    Preprocesses the clinical and biospecimen DataFrames (single patient or combined) for inference
    using the loaded preprocessor.

    Args:
        clinical_path (str): Path to the clinical CSV file (single row or combined).
        biospecimen_path (str, optional): Path to the biospecimen CSV file (only if separate). Defaults to None.
        feature_cols_path (str, optional): Path to the saved feature columns list. Defaults to 'model/feature_cols.pkl'.
        preprocessor_path (str, optional): Path to the saved preprocessor. Defaults to 'model/preprocessor.joblib'.

    Returns:
        torch.Tensor: The preprocessed merged data as a PyTorch tensor (single row) or None on error.
    """
    try:
        preprocessor = load(preprocessor_path)
    except Exception as e:
        print(f"Error loading the preprocessor from {preprocessor_path}: {e}")
        return None

    try:
        df_clinical = pd.read_csv(clinical_path)
        if biospecimen_path:
            df_biospecimen = pd.read_csv(biospecimen_path)
        else:
            df_biospecimen = None
    except Exception as e:
        print(f"Error reading input CSV files: {e}")
        return None

    # --- Generate Biospecimen Features if separate file provided ---
    if df_biospecimen is not None:
        try:
            df_biospecimen_features = create_biospecimen_features_final(df_biospecimen.copy())
        except Exception as e:
            print(f"Error generating biospecimen features: {e}")
            return None

        # --- Merge DataFrames (assuming 'submitter_id' is the common key) ---
        try:
            df_merged = pd.merge(df_clinical, df_biospecimen_features, on='submitter_id', how='inner')
            if df_merged.empty:
                print("Error: Merged DataFrame is empty. Check 'submitter_id' column in input files.")
                return None
        except Exception as e:
            print(f"Error merging clinical and biospecimen data: {e}")
            return None
    else:
        df_merged = df_clinical.copy() # If combined file, use it directly

    try:
        with open(feature_cols_path, 'rb') as f:
            feature_cols_dict = load(f)
            clinical_cols = feature_cols_dict.get('clinical_cols', [])
            biospecimen_cols = feature_cols_dict.get('biospecimen_cols', [])
            all_relevant_cols = clinical_cols + biospecimen_cols
    except Exception as e:
        print(f"Error loading feature columns from {feature_cols_path}: {e}")
        return None

    # --- Select relevant columns ---
    if not all(col in df_merged.columns for col in all_relevant_cols):
        print(f"Error: Missing expected columns in merged/combined DataFrame. Expected: {all_relevant_cols}, Found: {df_merged.columns.tolist()}")
        return None

    df_processed = df_merged[all_relevant_cols].copy()

    # --- Apply the loaded preprocessor ---
    try:
        processed_features = preprocessor.transform(df_processed)
        print("Clinical and biospecimen data preprocessed using the loaded preprocessor.")
        return torch.tensor(processed_features, dtype=torch.float32)
    except Exception as e:
        print(f"Error applying the preprocessor: {e}")
        return None

def process_preprocessed_clinical_biospecimen(csv_path, feature_cols_path='model/feature_cols.pkl', preprocessor_path='model/preprocessor.joblib'):
    """
    Loads and preprocesses the already combined clinical and biospecimen CSV using the loaded preprocessor.
    This function now essentially calls the main preprocessing function.
    """
    return preprocess_clinical_biospecimen_data(csv_path, None, feature_cols_path, preprocessor_path)

if __name__ == "__main__":
    # --- Simulate saving the feature columns ---
    os.makedirs('model', exist_ok=True)
    from joblib import dump
    feature_cols = {
        'image_feature_cols': [f'pca_feature_{i}' for i in range(256)], # Example image features
        'clinical_cols': ['clinical_feature_1', 'categorical_clin_1', 'clinical_feature_2'],
        'biospecimen_cols': ['bio_feature_1', 'num_primary_tumor_samples', 'categorical_bio_1']
    }
    dump(feature_cols, 'model/feature_cols.pkl')

    # --- Simulate saving a dummy preprocessor ---
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    numerical_features = ['clinical_feature_1', 'clinical_feature_2', 'bio_feature_1', 'num_primary_tumor_samples']
    categorical_features = ['categorical_clin_1', 'categorical_bio_1']
    preprocessor_dummy = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])
    # Fit on some dummy data (replace with your actual training data loading)
    dummy_df = pd.DataFrame(np.random.rand(10, 6), columns=numerical_features + categorical_features)
    preprocessor_dummy.fit(dummy_df)
    dump(preprocessor_dummy, 'model/preprocessor.joblib')

    # --- Simulate input data ---
    clinical_data_string = """submitter_id,clinical_feature_1,categorical_clin_1,clinical_feature_2
TCGA-XX-XXXX,1.2,A,5.6
"""
    biospecimen_data_string = """submitter_id,bio_feature_1,num_primary_tumor_samples,categorical_bio_1,sample_1_type
TCGA-XX-XXXX,0.8,2,X,Primary Tumor
"""
    combined_data_string = """clinical_feature_1,categorical_clin_1,clinical_feature_2,bio_feature_1,num_primary_tumor_samples,categorical_bio_1
1.2,A,5.6,0.8,2,X
"""

    from io import StringIO
    df_clinical_single = pd.read_csv(StringIO(clinical_data_string))
    df_biospecimen_single = pd.read_csv(StringIO(biospecimen_data_string))
    df_combined_single = pd.read_csv(StringIO(combined_data_string))

    # --- Preprocess the single patient data (as if called from Flask) ---
    preprocessed_data_from_files = preprocess_clinical_biospecimen_data(
        StringIO(clinical_data_string),
        StringIO(biospecimen_data_string)
    )

    if preprocessed_data_from_files is not None:
        print("\nPreprocessed data from files (using loaded preprocessor):")
        print(preprocessed_data_from_files)

    # --- Preprocess the already combined data (as if called from Flask) ---
    preprocessed_combined_data = process_preprocessed_clinical_biospecimen(
        StringIO(combined_data_string)
    )

    if preprocessed_combined_data is not None:
        print("\nPreprocessed combined data (using loaded preprocessor):")
        print(preprocessed_combined_data)