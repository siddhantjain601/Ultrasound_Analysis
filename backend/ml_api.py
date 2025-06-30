from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import io
import cv2

app = FastAPI()


import torch
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
        # Change input channels from 3 to 1 for grayscale images
        self.e1 = encoder_block(1, 64)  # Modified line
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.b1 = conv_block(256, 512)
        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # Add final activation
        # self.final_activation = nn.Sigmoid()

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


model = attention_unet()
model.load_state_dict(torch.load('attention_unet_busi.pth', map_location=torch.device('cpu')))
model.eval()

# Configuration
threshold = 0.5
device = torch.device('cpu')

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data)).convert('L')  # Grayscale
        original_size = original_image.size  # (width, height)

        # Transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Prepare input tensor
        image_tensor = transform(original_image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
        
        # Process output
        mask = torch.sigmoid(output).cpu().numpy()[0][0]  # Remove batch and channel dim
        mask = (mask > threshold).astype(np.uint8) * 255

        # Create overlay
        predicted_mask = Image.fromarray(mask).resize(original_size, Image.NEAREST)
        overlay = Image.blend(
            original_image.convert('RGBA'),
            predicted_mask.convert('RGBA'),
            alpha=0.3
        )

        # After mask computation
        predicted_mask = Image.fromarray(mask).resize(original_size, Image.NEAREST)

        # Convert to bytes
        buf = io.BytesIO()
        predicted_mask.save(buf, format='PNG')
        return Response(content=buf.getvalue(), media_type="image/png")


    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)
