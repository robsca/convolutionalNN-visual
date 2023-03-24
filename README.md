
### SETUP

```bash
virtualenv venv -p python3
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### DESCRIPTION
It's been deployed here: https://robsca-convolutionalnn-visual-main-qnoqf7.streamlit.app

This application is a simple demonstration of how a convolutional neural network (CNN) can be used to classify images. The CNN is trained on the MNIST dataset of handwritten digits. The user can draw a digit on the canvas and the CNN will classify the digit. The user can also see the CNN's predictions for each digit.

### RUN
```bash
streamlit run main.py
```

