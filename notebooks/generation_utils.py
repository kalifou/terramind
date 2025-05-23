import torch
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference

def large_tile_generation(model, input, params=None):
    if params is None:
        # Set tiled inference parameters
        tiled_inference_parameters = TiledInferenceParameters(
            h_crop=224, h_stride=200, w_crop=224, w_stride=200,
            average_patches=True,  # Average pixel values across chips
            batch_size=8,
            verbose=True
        )

    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, dtype=torch.float, device='cpu')
    if len(input.shape) == 3:
        input = input.unsqueeze(0)  # Add batch dim

    # Define output channels: S1RTC + DEM  + LULC + NDVI
    num_channels = 2 + 1 + 10 + 1

# Define model forward for tiled inference
    def model_forward(x):
        # Run chained generation for all output modalities
        generated = model(x)

        # TerraTorch tiled inference expects a tensor output from model forward. Concatenate all generations along channel dimension
        out = torch.concat([
            generated['S1RTC'],
            generated['DEM'],
            generated['LULC'],
            generated['NDVI']
        ], dim=1)

        return out

pred = tiled_inference(model_forward, input, num_channels, tiled_inference_parameters)
pred = pred.squeeze(0)  # Remove batch dim