## Welcome to Language of Images: Classifying marketing images with transformers and vision language models

This is the public repository for the IJRM article Language of images: Classifying marketing images with transformers and vision language models (https://doi.org/10.1016/j.ijresmar.2026.01.001).

Here you can find our easy to implement script to apply state-of-the-art image classification for your own dataset as well as all code for replication.

All our scripts are using the Python programming language. If you do not have any version of Python installed on your system, please follow the upcoming steps. If you run into any errors using the code later, please check whether you have matching versions of all python packages installed (see section 1.2).

### 1 Requirements

#### 1.1 Python Installation

We recommend Python 3.10 to 3.12. If you do not have Python installed, download it from https://www.python.org/downloads/ or install it with your system package manager.

Verify your installation:

```
python3 --version
pip3 --version
```

Create and activate a virtual environment (recommended):

```
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate the virtual environment later:

```
deactivate
```

#### 1.2 Required Python Packages

You can install these packages using "pip install PACKAGE_NAME" or add them to a requirements.txt file.

If you can, we recommend you install the CUDA enabled version of torch (using your Nvidia GPU for speed acceleration over CPU computation).

- torch>=2.5.1
- tensorflow>=2.19.0
- transformers>=4.48.2
- torchvision>=0.19.0+cu124
- openai>=1.70.0
- pandas>=2.2.2
- keras>=3.9.2


### 2 Apply image classification on your dataset

#### 2.1 Preparing your dataset

For application of image classification you need 1) image files and 2) annotated/labeled entries in a .csv or .xlsx file for training and evaluation.

1. Store all images you want to work with in one subfolder (e.g., 'Application_Images').

2. Your dataset should contain the following columns:
- image_path: Path to the image file
- label: Correct class/category of the corresponding image

We recommend storing image paths relative to your dataset file (so the project is easy to move to another machine).

Example folder structure:

```
Example_Dataset/
	Application_Images/
		img_001.jpg
		img_002.jpg
	labels.csv
```

Example labels.csv:

```
image_path,label
Application_Images/img_001.jpg,joy
Application_Images/img_002.jpg,trust
```

To get a better understanding see the "Example_Dataset" here in this Github repository.

This toy example for emotion classification showcases how files should be structured. You can train the model on any classes/categories of your choosing, just make sure to provide enough training images per class (at least 4, more better) and roughly balanced images per class (e.g., 120 images of class A, 90 images of class B etc.).


#### 2.2 Running the classification

Run the script and pass your dataset file. The `--dataset` argument accepts a .csv or .xlsx file. Use `--output_path` to choose where the trained model and predictions should be saved.

```
python plug_play_image_classification.py --dataset Example_Dataset/emotion_data.csv --output_path model_output/test_run1
```

What you will get in the output path:

- A trained model file
- A predictions file with labels per image
- A short metrics summary (e.g., accuracy, F1)

**TBD: We will upload the scripts soon, after we have completed all testing.**

Thank you for your patience.
Max, Mark, Jochen, and Keno