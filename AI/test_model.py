import torch
import numpy as np
from src.ai.transition_params_model import TransitionParamsVAE

checkpoint = torch.load("models/params_vae.pth", map_location='cpu')

model = TransitionParamsVAE(
    input_dim=28,
    hidden_dim=512,
    latent_dim=128,
    output_dim=38
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

fake_input = torch.randn(1, 28)
params = model.predict(fake_input)
params_dict = model.get_params_dict(params)

print("Paramètres retournés par le modèle:")
for k, v in params_dict.items():
    print(f"  {k}: {v:.3f}")