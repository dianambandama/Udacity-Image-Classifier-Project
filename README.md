# AI Programming with Python Project

Image Classification (102 Flower Categories) using PyTorch Models
This repository contains the project code for **Udacity's AI Programming with Python Nanodegree program**. The project involves building an image classifier using PyTorch and converting it into command-line applications: train.py and predict.py. The image classifier is designed to recognize 102 different species of flowers from the provided dataset.

## Project Overview
### Image Classifier Development
  - Notebook: Image Classifier Project.ipynb
    - Used VGG16 or DenseNet121 from torchvision.models pretrained models.
    - Loaded a pre-trained network and defined a new, untrained feed-forward network as a classifier.
    - Utilized ReLU activations and dropout.
    - Trained the classifier layers using backpropagation.
    - Tracked loss and accuracy on the validation set to determine the best hyperparameters.
### Command-Line Applications
- train.py: Trains the image classifier.
- predict.py: Predicts flower species from an input image.
### Command-Line Applications
-   train.py: Trains the image classifier.The following arguments are available:
-   data_dir: Provide data directory. (Mandatory, Type: str)
 -  save_dir: Provide saving directory. (Optional, Type: str)
 -  arch: Use vgg16 (default) or densenet121. (Optional, Type: str)
 -  learning_rate: Learning rate. Default value: 0.001. (Optional, Type: float)
 -  Hidden_units: Hidden units in the classifier. Default value: 512. (Optional, Type: int)
-   epochs: Number of epochs. Default value: 5. (Optional, Type: int)
-   gpu: Option to use GPU (CUDA). (Optional, Type: str)
    predict.py: Predicts flower species from an input image. The following arguments are available:
-   age_dir: Provide the path to the image. (Mandatory, Type: str)
-   ad_dir: Provide the path to the checkpoint. (Mandatory, Type: str)
-   p_k: Top K most likely classes. (Optional, Type: int) -   tegory_names: Mapping of categories to real names. Provide the JSON file name. (Optional, Type: str)
 -  gpu: Option to use GPU (CUDA). (Optional, Type: str)
  ### Dataset 
  The dataset contains 102 flower categories. Each category has a set of images for training, validation, and testing.

  ## Usage
#### Training the Model
  Run the following command to train the model:
  - python train.py data_dir --arch vgg16 --epochs 5 --gpu cuda
### Making Predictions
Run the following command to predict flower species:
- python predict.py image_path checkpoint_path --top_k 5 --gpu cuda
## License
- This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Udacity for providing the dataset and project guidelines.
- PyTorch for the deep learning framework.


