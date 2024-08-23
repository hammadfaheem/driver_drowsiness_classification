# Driver Drowsiness Detection

## Project Overview

Driver drowsiness is a major factor in road accidents, and detecting it early can prevent potentially life-threatening situations. This project leverages deep learning techniques to develop a model that can identify signs of drowsiness from facial images of drivers. By processing images and classifying them into "Drowsy" and "Non-Drowsy" categories, this model aims to enhance road safety by providing an automated detection mechanism.

## Dataset

The project uses the **Driver Drowsiness Dataset (DDD)** obtained from Kaggle. This dataset contains images labeled as either "Drowsy" or "Non-Drowsy," which are used to train the machine learning model. The dataset is divided into three subsets:
- **Training Set**: Used to train the model.
- **Validation Set**: Used to tune model parameters and prevent overfitting.
- **Test Set**: Used to evaluate the performance of the final model.

https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

## Data Preparation

### Step 1: Setting Up the Environment
To begin, the Kaggle API is configured to download the dataset. This involves installing the `kaggle` Python package and setting up authentication using a Kaggle API key.

### Step 2: Dataset Extraction
After downloading, the dataset is extracted and organized into appropriate folders. The images are divided into the following categories:
- **Drowsy**
- **Non-Drowsy**

### Step 3: Creating Train, Validation, and Test Folders
Using the `split_folders` library, the images are split into training, validation, and test sets with an 80-10-10 split ratio. This ensures that the model is trained on a diverse set of images and evaluated on unseen data.

## Model Architecture

### Convolutional Neural Network (CNN)
The core of the project is a CNN model built using TensorFlow and Keras. CNNs are particularly effective for image classification tasks due to their ability to capture spatial hierarchies in images.

The architecture of the CNN in this project includes:
- **Convolutional Layers**: These layers extract features from the input images.
- **Max-Pooling Layers**: These layers reduce the dimensionality of the feature maps, making the computation more efficient while retaining essential features.
- **Dropout Layers**: These layers help in preventing overfitting by randomly setting a fraction of input units to zero during training.
- **Fully Connected Layers**: These layers perform the final classification.

### Model Compilation and Training
The model is compiled using the `Adam` optimizer and the `categorical_crossentropy` loss function, which is standard for multi-class classification problems. The training process includes monitoring validation accuracy to prevent overfitting.

## Evaluation

After training, the model's performance is evaluated on the test set. Key metrics such as accuracy, precision, recall, and F1-score are calculated to gauge the model's effectiveness in detecting drowsiness.

## Results

The project successfully develops a CNN model capable of distinguishing between drowsy and non-drowsy drivers. The model achieves a significant level of accuracy, demonstrating its potential for real-world applications in driver monitoring systems.

## Future Work

Future enhancements could include:
- **Integration with real-time video streams**: Deploying the model in a real-time environment where it continuously monitors drivers' facial cues.
- **Incorporating other modalities**: Adding additional data sources like head position, eye closure rate, or yawning detection to improve accuracy.
- **Optimizing the model for mobile deployment**: Compressing the model so that it can run efficiently on mobile or embedded systems.

## How to Use

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- Kaggle API

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Kaggle API by placing your `kaggle.json` file in the appropriate directory:
   ```bash
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. Run the notebook to train and evaluate the model.

### Running the Model
Execute the notebook cells sequentially to:
- Download and prepare the dataset.
- Build and train the CNN model.
- Evaluate the model's performance.

