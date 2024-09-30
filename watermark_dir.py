import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import os
import argparse
import time
import random

from torchvision import transforms
from PIL import Image
import io
from text_zipper import TextZipper
from itertools import islice
from functools import partial

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoFeatureExtractor,
)
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset
import json
from scipy.special import betainc, gammaincc
from datasets import load_dataset
from modulations import MODULATIONS
import io
import os
from torch.utils.data import DataLoader, random_split, Subset, Sampler




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--watermark_encoder_model",
    type=str,
    default='models/Hide-R/bzhenc.pth',
    help="Path to the watermarking encoder model.")
parser.add_argument(
    "--watermark_power",
    type=float,
    default=1.0,
    help="Watermark power.")
parser.add_argument(
    "--watermark_decoder_model",
    type=str,
    default='models/Hide-R/bzhdec.pth',
    help="Path to the watermarking decoder model.")
parser.add_argument(
    '--language_model', '-L',
    type=str,
    default="facebook/opt-125m",
    help='Language model for text compression'
)
parser.add_argument(
    '--key', '-k', type=int, default=42,
    help='Watermarking key'
)
parser.add_argument(
    '--modulation', '-m', default='cyclic',
    choices = list(sorted(MODULATIONS.keys())),
    help='Message modulation scheme'
)

parser.add_argument(
    '--image-size', '-s', type=int, default=128,
    help='Watermark size',
)
parser.add_argument(
    "--adapter_path", type= str,
    default = None
)
parser.add_argument(
    "--input_directory", "-i",default = None,type= str ,help="Directory containing the images to be watermarked.",
)


args = parser.parse_args()

# prepare dataset
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            # transforms.ToTensor(),
            transforms.Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std),
        )
        

    def forward(self, x):
        with torch.no_grad():
            x = self.preprocess(x)
        return x

def compute_psnr(A,B):
    A = np.asarray(A)
    
    B = np.asarray(B)

    assert(A.size == B.size)

    size = np.prod(A.shape)
    mse = np.sum((A - B)**2) / size
    return 10*np.log10(255*255 / mse)
    
def pfa(c, D, double=False):
    pfa = 0.5 * betainc((D - 1) * 0.5, 0.5, 1.0 - c**2)
    if double:
        return 2.0 * pfa
    else:
        if c < 0.0:
            pfa = 1.0 - pfa
    return pfa

def resize_and_pad_to_numpy(img, target=256):
    size = img.size
    ar = size[0] / size[1]
    if ar > 1:
        resize = (target, int(target / ar))
    else:
        resize = (int(target * ar), target)
    img = img.resize(resize, Image.LANCZOS).convert('RGB')
       
    # pad if needed, centering to avoid border artifacts if we can
    img = np.asarray(img)
    pw = (target - resize[0]) // 2
    ph = (target - resize[1]) // 2
    padding = ((ph, target - resize[1] - ph), (pw, target - resize[0] - pw), (0,0))
    img = np.pad(img, padding, mode="reflect")
    return img, ar

def embed(img, msg, power):
    msg = msg / np.sqrt(np.dot(msg, msg))
    msg = torch.tensor(msg, dtype=torch.float32).unsqueeze(0).to(device)

    imgo, ar = resize_and_pad_to_numpy(img, target=args.image_size)
    imgt = transform_imnet(imgo).unsqueeze(0).to(device)

    with torch.no_grad():
        t0 = time.time()
        imgw = watermark_encoder(imgt, msg)

    # resize and remove padding
    y = (imgw - imgt).cpu().numpy() * 0.5 + 0.5
    mimg = Image.fromarray((y[0].transpose((1,2,0)) * 255.0).clip(0,255.0).astype(np.uint8))
    size = max(img.size)
    mimg = mimg.resize((size, size), Image.LANCZOS)
    mimg = np.asarray(mimg)
    pw = (size - img.size[0]) // 2
    ph = (size - img.size[1]) // 2
    mimg = mimg[ph:ph+img.size[1],pw:pw+img.size[0]]

    mimg = (mimg / 255.0 - 0.5)

    # add to original
    y = np.asarray(img) + mimg * power * 255.0
       
    imgw = Image.fromarray(y.clip(0,255.0).astype(np.uint8))
    return imgw

def detect(img):
    # isotropic downscale
    img, ar = resize_and_pad_to_numpy(img, target=args.image_size)
    img = transform_imnet(img).unsqueeze(0).to(device)

    with torch.no_grad():
        dec = watermark_decoder(img)[0].cpu().numpy()
        # torchscript output is unnormalized to make it easier to whiten
        dec = dec / np.sqrt(np.dot(dec, dec))

    return dec

convert_tensor = transforms.ToTensor()

def text_encode(text):
    # encode
    bitstream = io.BytesIO()
    H = tz.encode(bitstream, text)
    data = bitstream.getvalue()
    return data, H

def text_decode(data):
    bitstream = io.BytesIO(data)        
    # decode
    decoded_text = tz.decode(bitstream, max_length=10)

    return decoded_text.split("\n")[0]

def watermark_sample(args, caption, image_file, watermark_power):
    results = {}
    c = 0

    # Compress caption with LLM
    data, nb_bits = text_encode(caption)

    # Modulate to vector on unit hypersphere in 256D
    modulator = MODULATIONS[args.modulation](args.key)
    w = modulator.encode(data)

    # Open input image
    pil_image = image_file.convert('RGB')

    # Embed in reference
    img_w = embed(pil_image, w, watermark_power)
    #save and open back img_w
    img_w.save("watermarked_image.png")
    img_w = Image.open("watermarked_image.png")
    psnr = compute_psnr(img_w, pil_image)
    results["psnr"] = psnr
    wr = detect(img_w)
    decoded_data = modulator.decode_plain(wr)
    text = text_decode(decoded_data)
    text = ' '.join(text.split(" ")[0:-1])
    caption = ' '.join(caption.split(" ")[0:-1])

    if text != caption:
        print("WRONG CAPTION")
        print("ORIGINAL CAPTION : ",caption, "DECODED : ",text)

    return img_w,w

def process_dataset(modulation, key, dataset = None, watermark_power = 1.0, dataloader = None,OUTPUT_DIR = "watermarked_images"):
    results_dict = {}
    c=0
    os.makedirs(f"{OUTPUT_DIR}",exist_ok=True)
    for og_image_file, images in tqdm(dataloader,total=len(dataloader)):
        c+=1 
        caption = dataset[str(og_image_file[0])]
        caption = caption.replace('\n', '').replace('\t', '')
        
        image_file = f"{OUTPUT_DIR}/{og_image_file[0]}_watermark_pow_{watermark_power}.png"
        # key = og_image_file[0]
        img_w,w = watermark_sample(args, caption, images[0], watermark_power)

        if img_w is None:
            continue
        img_w.save(image_file)
    return results_dict

def main():
    seed = 42 
    print ("Seed : ",seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_path = "emu_blip2_captions_test_set_short_captions.json"

    if args.input_directory is None:
        print("args.input_directory is None")
        print("using facebook/emu_edit_test_set_generations as default")
        # Prepare your dataset
        dataset = load_dataset("facebook/emu_edit_test_set_generations")
        test_dataset = dataset["test"]

    def custom_collate_emu_batch(batch):
        # Initialize tensors lists for pixel_values, input_ids, and attention_masks

        idxs = []
        images = []
        pil_images = []

        inputs_list = []
        for sample in batch:
            idxs.append(sample['idx'])
            image = sample['image'].convert('RGB')
            pil_images.append(image)
            # images.append(image)

        return idxs, pil_images
    
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_emu_batch,
                    num_workers=16, pin_memory=True, drop_last=False)

    with open(data_path,"r") as file:
        data = json.load(file)
    captions = list(data.values())
    captions = [caption.replace('\n', '').replace('\t', '') for caption in captions]
    
    results = process_dataset(args.modulation,int(args.key), dataset= data,watermark_power=args.watermark_power,dataloader = dataloader)

if __name__ == '__main__':

    print(f'Loading watermark encoder from {args.watermark_encoder_model}...')
    watermark_encoder = torch.jit.load(args.watermark_encoder_model).to(device)
    watermark_encoder.eval()

    print(f'Loading watermark decoder from {args.watermark_decoder_model}...')
    watermark_decoder = torch.jit.load(args.watermark_decoder_model).to(device)
    watermark_decoder.eval()
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    print(f'Loading LLM for text compression from {args.language_model}')
    tz = TextZipper(modelname = args.language_model,adapter_path=args.adapter_path)
    tz.model.to(device)

    
    main()
