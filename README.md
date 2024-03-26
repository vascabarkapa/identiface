# Identiface

Identiface is a Python project that uses computer vision techniques to detect and analyze faces in real-time. It utilizes pre-trained models for face detection, age estimation, and gender classification to provide information about the detected faces.

## Prerequisites

To run this project, you need to have the following dependencies installed:

- Python 3
- OpenCV (cv2)
- NumPy

## Installation

1. Clone the project repository:

```
git clone https://github.com/vascabarkapa/identiface.git
```

2. Change into the project directory:

```
cd identiface
```

3. Install the required Python packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Make sure your webcam is connected to your computer.

2. Run the following command to start the Identiface program:

```
python identiface.py
```

3. The program will open a new window showing the live video feed from your webcam. It will detect faces in the video and display the estimated age and gender information for each detected face.

4. Press 'q' on your keyboard or close the window to exit the program.

## Configuration

The project uses pre-trained models for face detection, age estimation, and gender classification. The paths to these models are defined in the `identiface.py` file. If you have different model files or want to use custom models, you can modify the paths accordingly.

```python
# Models
FACE_PROTO = "face_detector/opencv_face_detector.pbtxt"
FACE_MODEL = "face_detector/opencv_face_detector_uint8.pb"
AGE_PROTO = "age/age_deploy.prototxt"
AGE_MODEL = "age/age_net.caffemodel"
GENDER_PROTO = "gender/gender_deploy.prototxt"
GENDER_MODEL = "gender/gender_net.caffemodel"
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.