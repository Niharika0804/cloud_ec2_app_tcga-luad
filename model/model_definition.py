import torch
import torch.nn as nn

class CrossFusionTransformer(nn.Module):
    def __init__(self, image_dim, clinical_dim, biospecimen_dim, nhead=4, num_encoder_layers=1,
                 image_proj_dim=256, clinical_proj_dim=128, biospecimen_proj_dim=64):
        super(CrossFusionTransformer, self).__init__()

        self.image_proj = nn.Linear(image_dim, image_proj_dim)
        self.image_norm = nn.BatchNorm1d(image_proj_dim)
        self.image_transformer = nn.TransformerEncoderLayer(d_model=image_proj_dim, nhead=nhead)

        self.clinical_proj = nn.Linear(clinical_dim, clinical_proj_dim)
        self.clinical_norm = nn.BatchNorm1d(clinical_proj_dim)
        self.clinical_transformer = nn.TransformerEncoderLayer(d_model=clinical_proj_dim, nhead=nhead)

        self.biospecimen_proj = nn.Linear(biospecimen_dim, biospecimen_proj_dim)
        self.biospecimen_norm = nn.BatchNorm1d(biospecimen_proj_dim)
        self.biospecimen_transformer = nn.TransformerEncoderLayer(d_model=biospecimen_proj_dim, nhead=nhead)

        self.fusion_fc = nn.Linear(image_proj_dim + clinical_proj_dim + biospecimen_proj_dim, 256)
        self.fusion_norm = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, image_features, clinical_features=None, biospecimen_features=None):
        image_features = self.dropout(torch.relu(self.image_norm(self.image_proj(image_features))))
        image_features = self.image_transformer(image_features.unsqueeze(1)).squeeze(1)

        if clinical_features is not None:
            clinical_features = self.dropout(torch.relu(self.clinical_norm(self.clinical_proj(clinical_features))))
            clinical_features = self.clinical_transformer(clinical_features.unsqueeze(1)).squeeze(1)

        if biospecimen_features is not None:
            biospecimen_features = self.dropout(torch.relu(self.biospecimen_norm(self.biospecimen_proj(biospecimen_features))))
            biospecimen_features = self.biospecimen_transformer(biospecimen_features.unsqueeze(1)).squeeze(1)

        fused_features = image_features
        if clinical_features is not None:
            fused_features = torch.cat((fused_features, clinical_features), dim=1)
        if biospecimen_features is not None:
            fused_features = torch.cat((fused_features, biospecimen_features), dim=1)

        fused_features = self.dropout(torch.relu(self.fusion_norm(self.fusion_fc(fused_features))))
        output = self.output_layer(fused_features)
        return output

if __name__ == '__main__':
    # Example usage based on the training script:
    image_dim = 256  # Number of PCA features from DICOM images
    clinical_dim = 123 # Replace with the actual number of clinical features after preprocessing (e.g., after one-hot encoding)
    biospecimen_dim = 45 # Replace with the actual number of biospecimen features after preprocessing

    model = CrossFusionTransformer(image_dim, clinical_dim, biospecimen_dim)
    print(model)

    # Example dummy inputs
    dummy_image_features = torch.randn(1, image_dim)
    dummy_clinical_features = torch.randn(1, clinical_dim)
    dummy_biospecimen_features = torch.randn(1, biospecimen_dim)

    output = model(dummy_image_features, dummy_clinical_features, dummy_biospecimen_features)
    print("Output shape:", output.shape)
    print("Example output:", output)
