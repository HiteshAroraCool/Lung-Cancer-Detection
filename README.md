# Lung Cancer Detection

A deep learning project for detecting and classifying lung cancer from CT scan images using MobileNet architecture.

## Project Overview

This project implements a multi-class lung cancer detection system using deep learning techniques, based on the [Lung Cancer Detection dataset from Kaggle](https://www.kaggle.com/code/bcscuwe1/lung-cancer-detection-muilt-classification).

## Features

- Multi-class lung cancer classification
- Transfer learning using MobileNet architecture
- Data augmentation for improved model performance
- Advanced training features:
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
- Detailed performance visualization

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - opencv-python
  - scikit-learn
  - tensorflow
  - keras

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Lung-Cancer-Detection.git
cd Lung-Cancer-Detection
```

2. Create and activate virtual environment:
```bash
python -m venv cnn_venv
.\cnn_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The model uses MobileNet as the base architecture with additional layers:
- Global Average Pooling
- Dense layers with dropout for regularization
- Softmax output layer for classification

## Project Structure

```
Lung-Cancer-Detection/
├── requirements.txt    # Project dependencies
├── setup.py           # Package setup configuration
├── .gitignore         # Git ignore rules
└── src/              # Source code directory
    └── model.py      # Model implementation
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## Acknowledgments

- Original Kaggle notebook by bcscuwe1
- NIH for the dataset
- Contributors and maintainers
