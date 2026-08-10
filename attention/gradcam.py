import torch
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Forward hook
        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        # Backward hook
        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, class_index=None):

        self.model.eval()

        # Clear previous gradients
        self.model.zero_grad()

        # Forward pass
        output = self.model(image)

        # If no class is provided, use predicted class
        if class_index is None:
            class_index = output.argmax(dim=1).item()

        # Select target class score
        score = output[:, class_index]

        # Backward pass
        score.backward()

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted activation maps
        cam = (weights * activations).sum(dim=1)

        # Remove negative values
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()

        if cam.max() != 0:
            cam = cam / cam.max()

        return cam, output

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()