"""Grad-CAM helper for torchvision ResNet50 and EfficientNetB0 models."""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), logits.detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def get_target_layer(model, model_name):
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError("Grad-CAM is configured for resnet50 and efficientnet_b0, not custom_cnn.")
