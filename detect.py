import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from modulations import MODULATIONS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse
import sys
import os
import json
from text_zipper import TextZipper
import io
from watermark_dir import *
import numpy as np
import pandas as pd
from scipy.special import betainc, gammaincc

from scipy.stats import pearsonr
import json
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watermark_encoder_model",
        type=str,
        default='models/Hide-R/encoder.pth',
        help="Path to the watermarking encoder model.")
    parser.add_argument(
        "--watermark_power",
        type=float,
        default=1.0,
        help="Watermark power.")
    parser.add_argument(
        "--watermark_decoder_model",
        type=str,
        default='models/Hide-R/decoder.pth',
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
        default= "models/finetuned_llm",
    )
    parser.add_argument(
        "--output_file","-d", type= str, help="Output file for the results"
    )
    parser.add_argument(
        "--input_directory", "-i",type= str ,help="Directory containing the images to be watermarked.",
    )


    args = parser.parse_args()
    return args



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

def text_encode(text):
    # encode
    bitstream = io.BytesIO()
    H = tz.encode(bitstream, text)
    data = bitstream.getvalue()

    return data, H

def pfa_min(rho_0, M):
    if rho_0 * M < 1e-3:
        # upper bound in case of numerical precision error
        rho = rho_0 * M
    elif 1 - rho_0 == 1:
        # approximation in case of numerical precision error
        rho = 1 - np.exp(-rho_0*M)
    else:
        # exact formula
        rho = 1 - (1 - rho_0)**M
    return rho

def fisher(pfas):
    k = len(pfas)
    spfas = np.asarray(pfas)
    X22 = -np.sum(np.log(pfas))
    # cdf of Chi-squared of degree 2*k for x=2.0*X22
    gpfa = gammaincc(k, X22)
    return gpfa

def pfa(c, D, double=False):
    pfa = 0.5 * betainc((D - 1) * 0.5, 0.5, 1.0 - c**2)
    if double:
        return 2.0 * pfa
    else:
        if c < 0.0:
            pfa = 1.0 - pfa
    return pfa


def text_decode(data,max_length=10):
    bitstream = io.BytesIO(data)
        
    # decode
    decoded_text = tz.decode(bitstream, max_length=10)

    return decoded_text

def detect(img):
    # isotropic downscale
    img, ar = resize_and_pad_to_numpy(img, target=args.image_size)
    img = transform_imnet(img).unsqueeze(0).to(device)

    with torch.no_grad():
        dec = watermark_decoder(img)[0].cpu().numpy()  
    return dec

def detect_watermark(image_path,key=None):
    if key is not None:
        modulator = MODULATIONS[args.modulation](key)
        print(key)
    else :
        modulator = MODULATIONS[args.modulation](args.key)

    # Open input image
    img_w = Image.open(image_path).convert('RGB')
    wr = detect(img_w)
    decoded_data = modulator.decode(wr)
    text = text_decode(decoded_data,max_length=None)
    
    return text, wr


def computing_rho(wr,key=None,orthogonalize = False):

        D = 256
        if key is not None:
            modulator = MODULATIONS[args.modulation](key)
        else :
            modulator = MODULATIONS[args.modulation](args.key)
        decoded_data = modulator.decode(wr)
        M = float(2**(modulator.N*8))
    
        W = modulator.encode(decoded_data, return_components=True)

        # normalize the components
        B = [b / torch.sqrt(b.dot(b)) for b in W]
        # compute cosine similarity with all components
        C = [wr.dot(b) for b in B]
        # compute individual p-values for single test
        R = [pfa(c, D) for c in C]
        # compute individual p-values for the min over M tests
        A = [pfa_min(r, M) for r in R]
        # aggregate the individual p-values with Fisher method
        rho = fisher(A)
        return rho,C,R,A #yes, it is rho, not rho_0

def detect_watermark_in_directory(directory,captions_json=None,coco=False):
    results = {}

    for root, dirs, files in os.walk(directory):
        for filename in tqdm(files,total=len(files)):
            if (filename.endswith(".jpg") or filename.endswith(".png")):
                image_path = os.path.join(root, filename)
                if "checkpoint" in filename:
                    continue

                ID = filename.split("_")[0]
                k = int(ID) #this works to change the key for each sample => better security 
                # k=42
                caption_decoded, wr = detect_watermark(image_path,key=k)
               
                # Extract the ID from the filename
                if ID in results:
                    results[ID]["captions"].append(caption_decoded)
                else:
                    results[ID] = {"captions": [caption_decoded]}


                if captions_json is not None:
                    with open(captions_json) as f:
                        captions = json.load(f)

                    caption = captions[ID]
                    rho,C,R,A = computing_rho(wr,key=k)

                    if "rho" in results[ID]:
                        results[ID]["rho"].append(rho)
                    else:
                        results[ID]["rho"] = [rho]
                    if "C" in results[ID]:
                        results[ID]["C"].extend(C)
                    else:
                        results[ID]["C"] = C
                    if "R" in results[ID]:
                        results[ID]["R"].extend(R)
                    else:
                        results[ID]["R"] = R
                    if "A" in results[ID]:
                        results[ID]["A"].extend(A)
                    else:
                        results[ID]["A"] = A

                    caption_decoded = ' '.join(caption_decoded.split(" ")[0:-1])
                    caption = ' '.join(caption.split(" ")[0:-1])
                    hamming_distance = sum(c1 != c2 for c1, c2 in zip(caption, caption_decoded))

                    if "hamming_distance" in results[ID]:
                        results[ID]["hamming_distance"].append(hamming_distance)
                    else:
                        results[ID]["hamming_distance"] = [hamming_distance]

                if captions_json is None and "checkpoint" not in filename:
                    try :
                        key_id = filename.split(".")[0]
                        rho = computing_rho(wr,key=int(key_id))

                        if "pfas" in results[ID]:
                            results[ID]["pfas"].append(rho)
                        else:
                            results[ID]["pfas"] = [rho]

                    except MemoryError: continue
    return results

def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    if isinstance(o, np.int32):
        return int(o)
    # Add more checks if needed
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def compute_mrr(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
    
    mrr_values = []
    for key, value in data.items():

        hamming_distances = [value["hamming_distance"] == [0]]
        # print(hamming_distances)
        mrr_values.extend([hd  for hd in hamming_distances])
    mrr_values = np.array(mrr_values)
    mrr = np.sum(mrr_values)/len(mrr_values)
    print("Message Recovery Rate :",mrr/len(mrr_values))
    return mrr

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Loading LLM for text compression from {args.language_model}')
    tz = TextZipper(modelname = args.language_model,adapter_path=args.adapter_path)
    tz.model.to(device)

    print(f'Loading watermark decoder from {args.watermark_decoder_model}...')
    watermark_decoder = torch.jit.load(args.watermark_decoder_model).to(device)
    watermark_decoder.eval()
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    # Specify the directory containing the images
    input_directory = Path(args.input_directory)
    output_file = Path(args.output_file)

    captions_source = "emu_blip2_captions_test_set_short_captions.json"

    # Detect watermarks
    results = detect_watermark_in_directory(str(input_directory), captions_json=captions_source)

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, default=default_converter)
    
    print(f"Results saved to {output_file}")
    mrr = compute_mrr(output_file)
