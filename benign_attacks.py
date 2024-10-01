import os
import random
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import torchvision.transforms.functional as F

def jpeg_compression(img):
    if isinstance(img, torch.Tensor):
        img = F.to_pil_image(img)
    img.save("temp.jpg", "JPEG", quality=50)
    compressed_img = Image.open("temp.jpg")
    os.remove("temp.jpg")  # Clean up the temporary file
    return compressed_img

def ensure_tensor(img):
    if isinstance(img, Image.Image):
        return transforms.ToTensor()(img)
    return img

def ensure_pil(img):
    if isinstance(img, torch.Tensor):
        return transforms.ToPILImage()(img)
    return img

def apply_transformations(input_dir, output_dir, image_size=(128, 128), crop_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform_list = [
        ("crop_400", transforms.RandomCrop(crop_size)),
        ("resize_128", transforms.Resize(image_size)),
        ("noise", lambda img: ensure_tensor(img) + torch.randn_like(ensure_tensor(img)) * 0.05),
        ("grayscale", transforms.Grayscale()),
        ("jpeg_compression", transforms.Lambda(lambda img: jpeg_compression(ensure_pil(img)))),
    ]

    for transform_name, transform in tqdm(transform_list, desc="Transformations"):
        transform_dir = os.path.join(output_dir, transform_name)
        if not os.path.exists(transform_dir):
            os.makedirs(transform_dir)

        for filename in tqdm(os.listdir(input_dir), desc=f"Processing {transform_name}"):
            if filename.endswith((".jpg", ".png")):
                img_path = os.path.join(input_dir, filename)
                image = Image.open(img_path).convert("RGB")
                
                transformed_image = transform(image)
                
                # Ensure the result is a PIL Image before saving
                transformed_image = ensure_pil(transformed_image)
                transformed_image.save(os.path.join(transform_dir, filename))

if __name__ == "__main__":


    hidden_input = "watermarked_images"
    hidden_output = "attacked_images"

    apply_transformations(hidden_input, hidden_output)
  
