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
        default='/home/gevennou/hidden_models/bzhenc.pth',
        help="Path to the watermarking encoder model.")
    parser.add_argument(
        "--watermark_power",
        type=float,
        default=1.0,
        help="Watermark power.")
    parser.add_argument(
        "--watermark_decoder_model",
        type=str,
        default='/home/gevennou/hidden_models/bzhdec.pth',
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
        '--modulation', '-m', default='cyclicorth',
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
    # print(H[0])
    # print("=== text '%s' (H=%.02f effective %d bits) ===" % (text, H, size * 8))

    #print("bitstream = ", bitstream.getvalue())

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
#    print("bitstream = ", bitstream.getvalue())
        
    # decode
    decoded_text = tz.decode(bitstream, max_length=10)

    return decoded_text#.split("\n")[0]

def detect(img):
    # isotropic downscale
    img, ar = resize_and_pad_to_numpy(img, target=args.image_size)
    img = transform_imnet(img).unsqueeze(0).to(device)

    with torch.no_grad():
        dec = watermark_decoder(img)[0].cpu().numpy()
        # torchscript output is unnormalized to make it easier to whiten
        # dec = dec / np.sqrt(np.dot(dec, dec))
    # print(dec.shape)    
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
    
    # print("DECODED : ", text)
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
        sigma = 1- np.prod(1-np.asarray_chkfinite(A))
        return rho,sigma,C,R,A #yes, it is rho, not rho_0

def encoding_caption_for_comparison(caption,key=None):
    # Encode the caption to be able to compare it with the decoded caption
    data = text_encode(caption)

    ##add the encoding step to return the vector representing the captino
    # Modulate to vector on unit hypersphere in 256D
    if key is not None:
        modulator = MODULATIONS[args.modulation](key)
        print(key)
    else :
        modulator = MODULATIONS[args.modulation](args.key)
    # modulator = MODULATIONS[args.modulation](args.key)
    # print("KEY : ",args.key)
        # Set the dimension
    w = modulator.encode(data)
    dim = 256

    # Generate a random vector
    random_vector = torch.randn(dim)

    # Normalize the vector to have a norm of 1
    normalized_vector = random_vector / torch.norm(random_vector)
    # data_ = text_decode(data)
    test = modulator.decode_plain(normalized_vector)
    encoded_vector = modulator.encode(test)
    print("tada",torch.dot(torch.from_numpy(encoded_vector),normalized_vector))
    exit()

    test_ = text_decode(test,max_length=None)
    return w

def detect_watermark_in_directory(directory,captions_json=None,coco=False):
    results = {}
    errors = 0
    pfas = 0
    print(directory)
    print(os.listdir(directory))
    for root, dirs, files in os.walk(directory):
        print(root,dirs,files)
        for filename in tqdm(files,total=len(files)):

            print(filename)
            if (filename.endswith(".jpg") or filename.endswith(".png")):
                image_path = os.path.join(root, filename)
                if "checkpoint" in filename:
                    continue
                if coco :
                    ID = filename.replace("_watermarked.jpg",".jpg")
                    key_id = ID.split(".")[0]
                    k = int(key_id)
                else :
                    ID = filename.split("_")[0]
                    # k = int(ID)
                    k=42
                caption_decoded, wr = detect_watermark(image_path,key=k)
               
                # Extract the ID from the filename
                if ID in results:
                    results[ID]["captions"].append(caption_decoded)
                else:
                    results[ID] = {"captions": [caption_decoded]}


                if captions_json is not None:
                    with open(captions_json) as f:
                        captions = json.load(f)
                    # print(captions.keys())
                    caption = captions[ID]
                    print("GROUND TRUTH : ",caption , "||| DECODED : ", caption_decoded)
                    rho,sigma,C,R,A = computing_rho(wr,key=k)

                    if "rho" in results[ID]:
                        results[ID]["rho"].append(rho)
                    else:
                        results[ID]["rho"] = [rho]
                    if "sigma" in results[ID]:
                        results[ID]["sigma"].append(sigma)
                    else:
                        results[ID]["sigma"] = [sigma]
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
                    #     results[ID]["correlation"].append(correlation)
                    # else :
                    #     results[ID]["correlation"] = [correlation]
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
                        # print(f"Caption : {caption} \n Decoded : {caption_decoded} \n Hamming distance : {hamming_distance}")
    return results
def plot_pfa_range(results):
    import matplotlib.pyplot as plt

    # Load the JSON data
    file_path = 'results_watermarking_pow1_generateds_10tok_emu_512.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Load the task nature for each idx
    task_file_path = 'task_idx_emu.json'
    with open(task_file_path, 'r') as task_file:
        task_data = json.load(task_file)

    # Remove task entries if ID not in data
    task_data = {key: value for key, value in task_data.items() if key in data}
    # Extract PFA values and corresponding Hamming distances
    pfas = []
    hamming_distances = []
    task_types = []
    
    for key, value in data.items():
        pfas.extend(value['pfas'])
        hamming_distances.extend(value['hamming_distance'])
        for i in range(len(value['pfas'])):
            task_types.extend([task_data[key]])
    print(len(pfas),len(task_types))
    # Convert lists to numpy arrays for easier manipulation
    pfas = np.array(pfas)
    hamming_distances = np.array(hamming_distances)
    task_types = np.array(task_types)

    # Define new bins with finer granularity including more intermediate ranges
    bins = [0, 1e-50, 1e-40, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1]
    bin_labels = ['0-1e-50', '1e-50-1e-40', '1e-40-1e-30', '1e-30-1e-25', '1e-25-1e-20', '1e-20-1e-15', '1e-15-1e-10', '1e-10-1e-8', '1e-8-1e-6', '1e-6-1e-4', '1e-4-1e-3', '1e-3-1e-2', '1e-2-1']
    pfa_bins = np.digitize(pfas, bins)

    # Compute the frequency for each bin for different thresholds
    def compute_frequency(threshold_mask, pfa_bins, bin_count):
        frequency_per_bin = []
        for i in range(1, bin_count):
            bin_mask = pfa_bins == i
            bin_hamming_greater_than_threshold = threshold_mask[bin_mask]
            frequency = np.sum(bin_hamming_greater_than_threshold) / np.sum(bin_mask) if np.sum(bin_mask) > 0 else 0
            frequency_per_bin.append(frequency)
        return frequency_per_bin

    # Calculate the frequency of Hamming distances greater than 1, 2, 5, and 10 with respect to PFA for each task type
    task_types_unique = np.unique(task_types)
    print(len(task_types))
    task_frequency_gt_1 = {}
    task_frequency_gt_2 = {}
    task_frequency_gt_5 = {}
    task_frequency_gt_10 = {}

    for task_type in task_types_unique:
        task_mask = task_types == task_type
        task_hamming_greater_than_one = hamming_distances > 1
        task_hamming_greater_than_two = hamming_distances > 2
        task_hamming_greater_than_five = hamming_distances > 5
        task_hamming_greater_than_ten = hamming_distances > 10

        task_frequency_gt_1[task_type] = compute_frequency(task_hamming_greater_than_one & task_mask, pfa_bins, len(bins))
        task_frequency_gt_2[task_type] = compute_frequency(task_hamming_greater_than_two & task_mask, pfa_bins, len(bins))
        task_frequency_gt_5[task_type] = compute_frequency(task_hamming_greater_than_five & task_mask, pfa_bins, len(bins))
        task_frequency_gt_10[task_type] = compute_frequency(task_hamming_greater_than_ten & task_mask, pfa_bins, len(bins))

    # Plotting the curves for different Hamming distance thresholds with intermediate ranges for each task type
    plt.figure(figsize=(14, 10))
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h', 'H', 'x', '+', 'D', 'd']
    for i, task_type in enumerate(task_types_unique):
        marker = markers[i % len(markers)]
        # plt.plot(bin_labels, task_frequency_gt_1[task_type], marker=marker, linestyle='-', label=f'Hamming Distance > 1 ({task_type})', linewidth=2)
        # plt.plot(bin_labels, task_frequency_gt_2[task_type], marker=marker, linestyle='-', label=f'Hamming Distance > 2 ({task_type})', linewidth=2)
        # plt.plot(bin_labels, task_frequency_gt_5[task_type], marker=marker, linestyle='-', label=f'Hamming Distance > 5 ({task_type})', linewidth=2)
        plt.plot(bin_labels, task_frequency_gt_10[task_type], marker=marker, linestyle='-', label=f'Hamming Distance > 10 ({task_type})', linewidth=2)

    plt.xlabel('PFA Range', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('Frequency of Hamming Distance Thresholds with respect to PFA Ranges for each Task Type', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("error_wrt_pfa_task_type.svg")

def plot_results(results):
    import matplotlib.pyplot as plt
    pfas = []
    for ID in results:
        pfas.extend(results[ID]["pfas"])
    #add the hamming distances to the plot, when they are no hamming distance, it means they were no error
    hamming_distances = []
    for ID in results:
        if "hamming_distance" in results[ID]:
            hamming_distances.extend(results[ID]["hamming_distance"])
        else:
            hamming_distances.append(0)
    # plt.hist(hamming_distances, bins=100)
    # # plt.show()
    # plt.savefig("hamming_distances.svg")
    #logscale for x axis

    # Calculate CDF
    sorted_pfas = np.sort(pfas)
    cdf = np.arange(1, len(sorted_pfas) + 1) / len(sorted_pfas)

    # Plotting the CDF with min and max values highlighted
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_pfas, cdf, linestyle='-', label='CDF')
    plt.xscale('log')

    # Highlight min and max values
    min_pfa = sorted_pfas[0]
    max_pfa = sorted_pfas[-1]
    plt.axvline(min_pfa, color='green', linestyle='--', label=f'Min PFA: {min_pfa:.2e}')
    plt.axvline(max_pfa, color='red', linestyle='--', label=f'Max PFA: {max_pfa:.2e}')

    plt.xlabel('Probability of False Alarm (log scale)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title('CDF of Probability of False Alarm with Min and Max Values', fontsize=16)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("pfass_cdf_min_max.svg")

def plot_cdf_types(file_path, task_file_path):
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Load the task nature for each idx
    with open(task_file_path, 'r') as task_file:
        task_data = json.load(task_file)

    # Remove task entries if ID not in data
    task_data = {key: value for key, value in task_data.items() if key in data}

    # Extract PFA values and corresponding Hamming distances
    pfas = []
    hamming_distances = []
    task_types = []

    for key, value in data.items():
        pfas.extend(value['pfas'])
        hamming_distances.extend(value['hamming_distance'])
        for i in range(len(value['pfas'])):
            task_types.extend([task_data[key]])

    # Convert lists to numpy arrays for easier manipulation
    pfas = np.array(pfas)
    hamming_distances = np.array(hamming_distances)
    task_types = np.array(task_types)

    unique_tasks = np.unique(task_types)

    plt.figure(figsize=(12, 8))

    for task in unique_tasks:
        task_mask = task_types == task
        task_pfas = pfas[task_mask]

        # Calculate CDF
        sorted_pfas = np.sort(task_pfas)
        cdf = np.arange(1, len(sorted_pfas) + 1) / len(sorted_pfas)

        # Plotting the CDF
        plt.plot(sorted_pfas, cdf, linestyle='-', label=f'CDF for {task}')

    plt.xscale('log')
    plt.xlabel('Probability of False Alarm (log scale)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title('CDF of Probability of False Alarm by Task Type', fontsize=16)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("pfas_cdf_by_task_type.svg")
    print("plot done")

def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    if isinstance(o, np.int32):
        return int(o)
    # Add more checks if needed
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

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

    # Determine if it's a COCO directory based on the path
    is_coco = False
    if is_coco:
        captions_source = "/home/gevennou/clark/scripts/coco_val2017_2k_captions.json"
    else :
        captions_source = "/home/gevennou/clark/scripts/emu_blip2_captions_test_set_short_captions.json"

    # Detect watermarks
    results = detect_watermark_in_directory(str(input_directory), captions_json=captions_source, coco=is_coco)

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, default=default_converter)

    print(f"Results saved to {output_file}")
