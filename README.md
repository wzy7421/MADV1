# MAD-QWT: Informatics-Enhanced Perception in Autonomous Driving

This repository includes the official PyTorch implementation for performing multimodal inference and training with the **MAD framework**, as detailed in our paper: *"Integrating Multi-Q Gabor Wavelet and Semantic Modeling for Informatics-Enhanced Perception in Autonomous Driving"*. 

MAD is a robust drivable area segmentation and occupancy-aware risk assessment model that systematically couples monocular vision with Q-factor Wavelet Transform (QWT) enhanced vibration signals to maintain stability under complex and degraded visual conditions.

## Acknowledgment 🌟

We would like to express our sincere gratitude to the open-source community, particularly the creators of the **TwinLiteNet** and **ENet** models for their pioneering work in extremely lightweight perception architectures. Their contributions have profoundly impacted the community and provided invaluable baselines for our research. We also thank the creators of the SpaceNet, DeepGlobe, and Cityscapes datasets for providing the foundational data for our evaluations.

## Project Structure

The project is organized as follows:

```text
├── data/
│   ├── SpaceNet/
│   ├── DeepGlobe/
│   └── Cityscapes/
├── images/                  # Sample images and demo outputs
├── models/                  # Network architectures (DeepLabV3+, Faster R-CNN)
├── weights/                 # Directory for pre-trained .pth weights
├── src/
│   ├── main.py              # Main script for training and evaluation
│   ├── qwt_core.py          # Multi-Q Gabor Wavelet Transform & SRAP/MinIP projection
│   └── data_utils.py        # Dataset loading and preprocessing scripts
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md


## Requirements
Python 3.8+
PyTorch 1.10+ (CUDA supported)
OpenCV
SciPy / PyWavelets (for QWT processing)
Install all dependencies via:
