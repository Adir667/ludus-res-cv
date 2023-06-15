# Ludus CV Response Time

This project aims to demonstrate the concept of replacing sensors, such as an accelerometer, in the smart paddle and providing response time analysis using computer vision tools.

## Getting Started

To get started with this project, follow the instructions provided in the Model training Jupiter notebook file included in this repository. You can alternatively use the pre-trained model file provided by us.

Make sure to install any necessary dependencies before running the project.

## Dataset

This repository includes a prepared dataset consisting of just over 300 images that focus on two annotations - the paddle and the trigger. The dataset is focused on the object in indoor lighting conditions.

You can use Roboflow's augmentation option to improve performance. The model provided in this repository is augmented with horizontal and vertical flips.

The images and videos were taken by our team, and we split the dataset with a ratio of 70%, 18%, and 12% for training, validation, and testing, respectively.

You can access the dataset here: https://app.roboflow.com/inh/trigger-0ubzh/7

## Model Training

To train the model, follow the instructions provided in the Model training Jupiter notebook file included in this repository. You can also find information about the reasoning of YOLOv5 algorithms in the TFGD document.

## API

The API documentation for interacting with the model can be found here: https://app.swaggerhub.com/apis/Adir667/Ludus-res/3.0.0

Source code can be found under /api folder.
The API was built with Swagger and Flask framworks

## Documents

This repository also hold the official educational documentation of this project (TFGD and advisory reports). These can be found under /docs folder.


## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
