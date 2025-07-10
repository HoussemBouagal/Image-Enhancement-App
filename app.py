from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import json
import io
from model_definitions import ESPCN, ConditionalUNet, add_noise

app = Flask(__name__)

# Configure upload folder using relative path
app.config["UPLOAD_FOLDER"] = os.path.join(app.instance_path, 'uploads')
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure models folder using relative path
MODELS_FOLDER = os.path.join(app.root_path, 'models')
os.makedirs(MODELS_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models using relative paths
espcn = ESPCN(upscale_factor=2).to(device)
espcn.load_state_dict(torch.load(
    os.path.join(MODELS_FOLDER, 'espcn_final.pth'),
    map_location=device
))
espcn.eval()

unet = ConditionalUNet().to(device)
unet.load_state_dict(torch.load(
    os.path.join(MODELS_FOLDER, 'unet_final.pth'),
    map_location=device
))
unet.eval()

# Image transformations
to_tensor = transforms.ToTensor()
resize_40 = transforms.Resize((40, 40), interpolation=Image.BICUBIC)
resize_160 = transforms.Resize((160, 160), interpolation=Image.BICUBIC)

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    img_np = tensor.permute(1, 2, 0).numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)

def crop_image(image, crop_data):
    """Crop image based on crop data from cropper.js"""
    left = crop_data['x']
    top = crop_data['y']
    width = crop_data['width']
    height = crop_data['height']
    
    # Ensure coordinates are within image bounds
    img_width, img_height = image.size
    left = max(0, min(left, img_width - 1))
    top = max(0, min(top, img_height - 1))
    width = min(width, img_width - left)
    height = min(height, img_height - top)
    
    return image.crop((left, top, left + width, top + height))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        crop_data_json = request.form.get("cropData")
        
        if not file:
            return jsonify({"error": "No image file provided"}), 400
            
        filename = file.filename
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(img_path)

        try:
            original_image = Image.open(img_path).convert("RGB")
            
            # Apply cropping if crop data is provided
            if crop_data_json:
                crop_data = json.loads(crop_data_json)
                image = crop_image(original_image, crop_data)
            else:
                image = original_image

            # Generate LR image
            lr_image = resize_40(image)
            lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)

            with torch.no_grad():
                sr_80 = espcn(lr_tensor)

                # UNet diffusion
                noisy_hr_tensor = to_tensor(resize_160(image)).unsqueeze(0).to(device)
                noisy_hr = add_noise(noisy_hr_tensor, noise_level=0.1)
                sr_160 = unet(noisy_hr, sr_80).clamp(0, 1)

            # Save results
            lr_up = lr_image.resize((160, 160), Image.BICUBIC)
            lr_up.save(os.path.join(app.config["UPLOAD_FOLDER"], "lr.png"))
            tensor_to_pil(sr_80).save(os.path.join(app.config["UPLOAD_FOLDER"], "sr80.png"))
            tensor_to_pil(sr_160).save(os.path.join(app.config["UPLOAD_FOLDER"], "sr160.png"))
            
            # Save the original image if it was cropped (for comparison)
            if crop_data_json:
                original_image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            message = "Images processed successfully."

        except Exception as e:
            print("❌ Error:", e)
            message = f"❌ An error occurred during processing: {str(e)}"
            return render_template("index.html", input_image=False, error_message=message)

        return render_template("index.html", input_image=True, success_message=message, filename=filename)

    return render_template("index.html", input_image=False)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)