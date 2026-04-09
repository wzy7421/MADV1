
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

### 2. Pre-trained Weights (Demo)
Due to current NDA restrictions prior to official publication, full-scale weights cannot be released yet. However, to enable immediate inference verification, we have included a subset-trained PyTorch weight (best_demo.pt) and an exported ONNX model (test.onnx) inside the model/ directory.

### 3. Execute Inference

For evaluation on the test set and generating occupancy grid results, run:

```bash
python main.py --mode test --weights model/best_demo.pt
```



## Experimental Results

Qualitative comparison of drivable area estimation and occupancy detection under complex driving scenarios:

![Overall Algorithm Framework Diagram](assets/Overall%20Algorithm%20Framework%20Diagram.jpg)
![Algorithm Architecture](assets/Algorithm%20Architecture.jpg)
![Experimental Results (1)](assets/Experimental%20results%20(1).jpg)
![Experimental Results (2)](assets/Experimental%20results%20(2).jpg)
![Experimental Results (6)](assets/Experimental%20results%20(6).jpg)
![Experimental Results (7)](assets/Experimental%20results%20(7).jpg)
![Experimental Results (4)](assets/Experimental%20results%20(4).jpg)
![Experimental Results (8)](assets/Experimental%20results%20(8).jpg)
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
├── datasets/                # Place downloaded SpaceNet/DeepGlobe datasets here
├── assets/                  # Diagram assets and results
├── images/                  # Sample images and demo outputs
├── model/                   # Network architectures and Pre-trained Weights
│   ├── faster_rcnn.py       # Detection architecture
│   ├── best.pt              # Pre-trained PyTorch weights
│   └── test.onnx            # Exported ONNX model
├── main.py                  # Main script for training and evaluation
├── qwt_core.py              # Multi-Q Gabor Wavelet Transform & SRAP/MinIP projection
├── data_utils.py            # Dataset loading and preprocessing scripts
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

---
