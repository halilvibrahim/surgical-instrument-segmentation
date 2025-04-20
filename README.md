# Surgical Instrumentation Segmentation

This project performs semantic segmentation of surgical instruments in endoscopic videos using a U-Net model with a ResNet-34 encoder pretrained on ImageNet.

## Model

We use the [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch) implementation of U-Net with a ResNet-34 backbone. The model is trained to segment the following 10 classes:

| Label | Class Name         |
|-------|--------------------|
| 0     | Background         |
| 1     | Instrument Shaft   |
| 2     | Instrument Wrist   |
| 3     | Instrument Body    |
| 4     | Suture Needle      |
| 5     | Thread             |
| 6     | Suction Instrument |
| 7     | Needle Holder      |
| 8     | Clamp              |
| 9     | Catheter           |

## 📁 Dataset

You can access the official training and test sets from the SAR-RARP50 dataset:

- **Train set:** [SAR-RARP50 Train Set](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529)
- **Test set:** [SAR-RARP50 Test Set](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_test_set/24932499)

For more information, visit the [official Synapse page](https://www.synapse.org/#!Synapse:syn27618412/wiki/616881).

## Requirements

Install required packages:

```bash
pip install torch torchvision opencv-python matplotlib tqdm scikit-learn segmentation-models-pytorch


## Folder structure

Surgical Instrumentation Segmentation/
├── dataloader.py
├── train.py
├── checkpoints/                # Model weights will be saved here
├── visualizations/             # Prediction visualizations will be saved here
├── train_set/
│   ├── video_01/
│   ├── ...
│   └── video_40/
├── test_set/
│   ├── video_41/
│   ├── ...
│   └── video_50/
├── extract_frames.py          # (You should have this script to extract frames)



## Usage


Step 1: Prepare Training Data
Place your training videos inside train_set folder as:

train_set/
├── video_01/
├── video_02/
├── ...
└── video_40/


Then run:
cd train_set
python3 extract_frames.py


Step 2: Prepare Test Data
Put your test videos into test_set as:

test_set/
├── video_41/
├── video_42/
├── ...
└── video_50/


Then run:
cd ../test_set
python3 extract_frames.py

Step 3: Train the Model
Run the training script from the main project directory:

cd ../Surgical Instrumentation Segmentation
python3 train.py


The model will save the best weights (based on validation loss) to:

checkpoints/best_model.pth


Step 4: Evaluation and Visualization
After training, the model is evaluated on the test set, and mean IoU is printed. Predictions for each sample are visualized and saved to:

visualizations/
Output Example
Each test image will be visualized as:

Original Input

Ground Truth Mask (color-coded)

Predicted Mask (color-coded)

