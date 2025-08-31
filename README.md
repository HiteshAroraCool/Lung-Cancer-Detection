# Lung Cancer Detection

<video width="100%" controls>
  <source src="Running.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

A deep learning project for detecting and classifying lung cancer from CT scan images using MobileNet architecture.

## Project Overview

This project implements a multi-class lung cancer detection system using deep learning techniques.

## Features

- Multi-class lung cancer classification
- Transfer learning using MobileNet architecture
- Data augmentation for improved model performance
- Advanced training features:
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
- Detailed performance visualization
- Model artifacts saving and loading
- Prediction pipeline for easy inference

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
git clone https://github.com/HiteshAroraCool/Lung-Cancer-Detection.git
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
├── README.md          # Project documentation
├── requirements.txt   # Project dependencies
├── artifacts/        # Saved model files
│   └── final_model.keras  # Trained model
├── checkpoints/      # Training checkpoints
│   └── best_model.weights.h5
├── dataset/         # Dataset directory
├── logs/           # Training logs
├── src/            # Source code directory
│   └── pipeline/   # Training and prediction pipelines
└── notebooks/      # Jupyter notebooks for experimentation
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## Acknowledgments

- NIH for the dataset