# SWIFT: Semantic Watermarking for Image Forgery Thwarting

Official implementation of the paper "SWIFT: Semantic Watermarking for Image Forgery Thwarting", accepted at WIFS 2024.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Watermarking](#watermarking)
  - [Detection](#detection)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This repository contains the implementation of SWIFT, a semantic watermarking technique designed to thwart image forgery. The project includes tools for both watermarking images and detecting potential forgeries.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Gautier29/swift_watermarking.git
   cd swift_watermarking
   ```

2. Install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Watermarking

To apply the SWIFT watermark the Emu Edit test set :

```bash
./watermark.sh 
```

### Detection

To extract watermark and compare with ground truth from "emu_blip2_captions_test_set_short_captions.json" run:

```bash
./detect.sh [options]
```

This script can be used on both original watermarked images and potentially attacked images for further study.

## Project Structure

- `lattices/`: Contains modulation code for comparison purpose
- `models/`: Includes finetuned optimization models
- `arithmeticcoding.py`: Implements arithmetic coding algorithms
- `benign_attacks.py`: Simulates benign attacks for testing
- `detect.py`: Core detection and MRR computation logic
- `finetune_llm.py`: Fine-tuning utilities
- `modulations.py`: Modulation techniques for watermarking
- `text_zipper.py`: Text compression utilities
- `watermark_dir.py`: Directory-based watermarking tools

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---

For more information, please refer to our paper: [SWIFT: Semantic Watermarking for Image Forgery Thwarting](https://arxiv.org/abs/2407.18995)

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{swift2024,
  title={SWIFT: Semantic Watermarking for Image Forgery Thwarting},
  author={Gautier Evennou and Vivien Chappelier and Ewa Kijak and Teddy Furon},
  booktitle={Proceedings of the IEEE International Workshop on Information Forensics and Security (WIFS)},
  year={2024}
}
