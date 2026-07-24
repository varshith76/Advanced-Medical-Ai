import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2

class MedicalGradCAM:
    def __init__(self):
        # 1. Load a pre-trained DenseNet121 architecture (Favored for Chest X-Rays)
        # Using weights argument to ensure compliance with modern torchvision versions
        try:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        except Exception:
            self.model = models.densenet121(pretrained=True)
            
        self.model.eval()

        # 🔥 CRITICAL FIX: Recursively turn off all inplace operations to prevent autograd errors
        for module in self.model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

        # 2. Target the final convolutional layer of DenseNet121
        self.target_layer = self.model.features.norm5
        
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _forward_hook(self, module, input, output):
        # Safely clone the tensor to isolate it from inplace mutations
        self.activations = output.detach().clone()

    def _backward_hook(self, module, grad_input, grad_output):
        # Safely clone the tensor gradients to clear out the backward hook view block
        self.gradients = grad_output[0].detach().clone()

    def _register_hooks(self):
        # Register hooks with clean references
        h1 = self.target_layer.register_forward_hook(self._forward_hook)
        h2 = self.target_layer.register_full_backward_hook(self._backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate(self, image_tensor):
        # Clear out previous registers
        self.gradients = None
        self.activations = None

        # Pass input forward through network
        output = self.model(image_tensor)
        
        # Binary classification mapping for demonstration (0: Normal, 1: Pneumonia)
        probabilities = torch.softmax(output, dim=1)
        confidence, class_idx = torch.max(probabilities, dim=1)
        
        class_idx = class_idx.item()
        confidence = confidence.item()
        diagnosis = "Pneumonia Detected" if class_idx == 1 else "Normal Lung Baseline"

        # Backward pass target tracking
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # Compute Grad-CAM Heatmap matrix arrays
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Linear combination of channels
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.clamp(cam, min=0) # Apply ReLU to CAM
        
        # Normalize into a range between 0 and 1
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
            
        return diagnosis, confidence, cam_np

# Setup standardized diagnostic pipeline transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Single instance deployment execution mapping
_cam_engine = MedicalGradCAM()

def generate_gradcam(image_bytes: bytes):
    """
    Core entrypoint module for Streamlit/FastAPI UI architectures.
    Accepts raw images files bytes, processes predictions, and returns mapping outputs.
    """
    # 1. Transform raw binary data stream into standard PIL format image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = pil_image.size
    
    # 2. Extract image tensor map inputs
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # 3. Process Grad-CAM evaluation matrices
    diagnosis, confidence, cam_mask = _cam_engine.generate(input_tensor)
    
    # 4. Use OpenCV to format and overlay the heatmap onto original source image
    cam_mask_resized = cv2.resize(cam_mask, (orig_w, orig_h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Merge original image and heatmap matrix together
    original_np = np.array(pil_image)
    overlayed_image = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    
    # 5. Compress the overlayed diagnostic output image array back into a binary stream
    output_pil = Image.fromarray(overlayed_image)
    buffer = io.BytesIO()
    output_pil.save(buffer, format="JPEG")
    heatmap_bytes = buffer.getvalue()
    
    return diagnosis, confidence, heatmap_bytes