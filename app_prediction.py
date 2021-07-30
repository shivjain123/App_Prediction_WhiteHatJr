import numpy as np
import os
import ssl
from sklearn.datasets import fetch_openml as fl
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acs
from PIL import Image

if (not os.environ.get('PYTHONHTTPSVERIFIED', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

x, y = fl('mnist_784', version=1, return_X_y=True)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n_classes = len(classes)

x_train, x_test, y_train, y_test = tts(x, y, train_size=7500, test_size=2500, random_state=42)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

lr = LogisticRegression(solver='saga', multi_class='multinomial')

lr.fit(x_train_scaled, y_train)

y_pred = lr.predict(x_test_scaled)
ac_score = acs(y_pred, y_test)

print(f"The Accuracy Score by Logistic Regression is {ac_score}.")

def get_prediction(image):
    img_pil = Image.open(image)
    image_bw = img_pil.convert('L').resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw, pixel_filter)
    img_inverted = np.clip(image_bw - min_pixel, 0, 255)
    max_pixel = np.max(image_bw)
    image_bw_resized_inverted_scaled = np.asarray(img_inverted)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
    test_pred = lr.predict(test_sample)
    return test_pred