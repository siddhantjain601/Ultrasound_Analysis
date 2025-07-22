from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import io
import base64

# ========= FastAPI App & CORS ==========
app = FastAPI()

# Adjust the `allow_origins` to your actual frontend URL/port in development and production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add or edit as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Model Imports and Loading ==========
from resnet import resnet18  # Update path if needed.

# Example class order; use your training order
class_names = ['benign', 'malignant', 'normal']
num_classes = len(class_names)

# Classification (ResNet)
resnet_model = resnet18(num_classes=num_classes)
resnet_model.load_state_dict(torch.load("resnet.pth", map_location="cpu"))
resnet_model.eval()

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Adjust to match model training
])

# Segmentation (Attention UNet)
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block(in_c[0]+out_c, out_c)
    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.b1 = conv_block(256, 512)
        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        b1 = self.b1(p3)
        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        output = self.output(d3)
        return output

seg_model = attention_unet()
seg_model.load_state_dict(torch.load('best_attention_unet_busi.pth', map_location="cpu"))
seg_model.eval()

# For grayscale input to UNet
seg_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
threshold = 0.5

# ========= API Endpoint ==========
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        # ---- Classification (ResNet) ----
        image_rgb = Image.open(io.BytesIO(image_data)).convert('RGB')
        clf_input = clf_transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            clf_output = resnet_model(clf_input)
            pred_class_idx = clf_output.argmax(1).item()
            pred_classname = class_names[pred_class_idx]

        if pred_classname == 'normal':
            return {
                "prediction": pred_classname,
                "segmentation": None
            }
        
        # ---- Segmentation (Attention UNet) ----
        image_gray = Image.open(io.BytesIO(image_data)).convert('L')
        original_size = image_gray.size  # (width, height)
        seg_input = seg_transform(image_gray).unsqueeze(0)
        with torch.no_grad():
            output = seg_model(seg_input)
            mask = torch.sigmoid(output).cpu().numpy()[0][0]
            mask = (mask > threshold).astype(np.uint8) * 255

        predicted_mask = Image.fromarray(mask).resize(original_size, Image.NEAREST)

        buf = io.BytesIO()
        predicted_mask.save(buf, format='PNG')
        byte_mask = buf.getvalue()
        mask_base64 = base64.b64encode(byte_mask).decode('utf-8')

        return {
            "prediction": pred_classname,
            "segmentation": mask_base64
        }
    except Exception as e:
        return {
            "error": str(e)
        }
