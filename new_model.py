import torch
import torch.nn as nn


class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.gamma = 0.5
        self.cte = 2
        self.contrast_factor = 1.5  # Example contrast adjustment factor

    def forward(self, pixel_tensor):
        # Normalize image to [0, 1] range
        if torch.max(pixel_tensor) > 1:
            pixel_tensor = pixel_tensor / 255.0

        # Apply gamma compression and tonemapping using PyTorch functions
        pixel_tensor = self.gamma_compression_torch(pixel_tensor)
        pixel_tensor = self.tonemap_torch(pixel_tensor)
        # Denormalize to [0, 255] and convert to uint8
        pixel_tensor = torch.clip(pixel_tensor, 0, 1) * 255
        pixel_tensor = pixel_tensor.to(torch.uint8)

        return pixel_tensor

    def gamma_compression_torch(image):
        """Converts from linear to gamma space."""
        return torch.clamp(image, min=1e-8) ** (1.0 / 2.2)

    def tonemap_torch(image):
        """Simple S-curved global tonemap."""
        return (3 * (image**2)) - (2 * (image**3))



# Initialize the new model
model = CustomModel()

# Test with a dummy image
dummy_img = torch.rand(1, 3, 1728, 2304)  # Batch of 1 RGB image (1728x2304)
output = model(dummy_img)

# Print output to verify shape and type
print(output.shape)  # Should be (1, 3, 1728, 2304)
print(output.dtype)  # Should be torch.uint8

import torch.utils.mobile_optimizer as mobile_optimizer

# Trace the model
traced_model = torch.jit.trace(model, dummy_img)

# Set the path same as in your previous code
remoteDesktopPath = "/home/ganeshmaudghalza/Documents/CVProject/CVColabFiles/simple_model.pt"

# Optimize for mobile
traced_model_optimized = mobile_optimizer.optimize_for_mobile(traced_model)

# Save the optimized model
traced_model_optimized._save_for_lite_interpreter(remoteDesktopPath)
