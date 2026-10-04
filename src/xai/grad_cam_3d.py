"""
3D Grad-CAM for R3D-18. Hooks the last residual stage (layer4) to get
class-activation maps over (time, height, width), then weights channels
by the gradient of the EF output w.r.t. those activations -- the
standard Grad-CAM recipe, extended to a spatiotemporal feature map
instead of a 2D one.
"""
import torch
import torch.nn.functional as F


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (1, 3, T, H, W). Returns a (T', H', W') importance map
        at the resolution of the hooked layer (coarser than the input,
        since R3D-18 downsamples time and space through the stages)."""
        self.model.zero_grad()
        out = self.model(clip)
        ef_pred = out["ef"]
        ef_pred.backward()

        # Global-average-pool the gradients over space+time to get one
        # importance weight per channel (standard Grad-CAM weighting).
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, T', H', W')
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)  # (T', H', W')
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu()