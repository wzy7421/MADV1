
# MAD-QWT: Informatics-Enhanced Perception in Autonomous Driving

This repository includes the official PyTorch implementation for performing multimodal inference and training with the **MAD framework**, as detailed in our paper:  
*"Integrating Multi-Q Gabor Wavelet and Semantic Modeling for Informatics-Enhanced Perception in Autonomous Driving".*  
**MAD** is a robust drivable area segmentation and occupancy-aware risk assessment model that systematically couples monocular vision with Q-factor Wavelet Transform (QWT)-enhanced vibration signals to maintain stability under complex and degraded visual conditions.

---

## Requirements

- **Python 3.8+**
- **PyTorch 1.10+** (CUDA supported)
- **OpenCV**
- **SciPy**
- **PyWavelets** (for QWT processing)

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation
Please download the original SpaceNet, DeepGlobe, and Cityscapes datasets from their official sources and place them into the `data/` directory.

Run the preprocessing script to format the labels for drivable area extraction:

```bash
python src/data_utils.py --dataset_path ./data/
```

### 2. Download Pre-trained Weights
To enable immediate inference and ensure full reproducibility, we provide the pre-trained weights.  
Download the `mad_best.pth` from [Insert Google Drive / Baidu Pan Link Here]  
Place the downloaded file into the `weights/` directory.

### 3. Execute Inference
For evaluation on the test set and generating occupancy grid results, run:

```bash
python src/main.py --mode test --weights weights/mad_best.pth
```

Enjoy accurate drivable area detection and physically-consistent risk assessment!

---

## Experimental Results

Qualitative comparison of drivable area estimation and occupancy detection under complex driving scenarios:

![Experimental Results (1)](assets/Experimental%20results%20(1).jpg)
![Experimental Results (2)](assets/Experimental%20results%20(2).jpg)
![Experimental Results (4)](assets/Experimental%20results%20(4).jpg)
![Algorithm Architecture](assets/Algorithm%20Architecture.jpg)
---

## License

This project is licensed under the MIT License. Feel free to use it in both open-source and commercial applications.

---

## Citation

If you find our work or this repository helpful, please consider citing our paper:

```bibtex
@article{Wang2026MAD,
  title={Integrating Multi-Q Gabor Wavelet and Semantic Modeling for Informatics-Enhanced Perception in Autonomous Driving},
  author={Wang, Zhenyu and Wang, Jianmin},
  journal={Advanced Engineering Informatics},
  year={2026}
}
```

---

## Acknowledgment 🌟

We would like to express our sincere gratitude to the open-source community, particularly the creators of the **TwinLiteNet** and **ENet** models for their pioneering work in extremely lightweight perception architectures. Their contributions have profoundly impacted the community and provided invaluable baselines for our research. We also thank the creators of the **SpaceNet**, **DeepGlobe**, and **Cityscapes** datasets for providing the foundational data for our evaluations.

---

## Project Structure

The project is organized as follows:

```bash
├── data/                          # Directory for datasets
│   ├── SpaceNet/                  # SpaceNet dataset
│   ├── DeepGlobe/                 # DeepGlobe dataset
│   └── Cityscapes/                # Cityscapes dataset
├── images/                        # Sample images and demo outputs
├── models/                        # Network architectures (DeepLabV3+, Faster R-CNN)
├── weights/                       # Directory for pre-trained .pth weights
├── src/                           # Source code
│   ├── main.py                    # Main script for training and evaluation
│   ├── qwt_core.py                # Multi-Q Gabor Wavelet Transform & SRAP/MinIP projection
│   └── data_utils.py              # Dataset loading and preprocessing scripts
├── requirements.txt               # Python dependencies
├── LICENSE                        # License file
└── README.md                      # Project documentation
```

---

Feel free to modify and adapt this template as needed for your project. The structure includes essential steps for data preparation, model execution, and licensing, and ensures a clean setup for new users.
