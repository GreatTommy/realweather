import time
import os
from .constants import *
from .weatherapi import *
import numpy as np


def get_live_image(img_url, delay=5):
    img_data = b""
    while not img_data:
        img_data = requests.get(img_url).content
        if not img_data:
            time.sleep(delay)
    return img_data


def get_live_weather(city):
    coords = api_geocoding(city)
    return api_current_weather(coords)

def get_live_weather_accu(city):
    # location = api_location_accu(city)
    location = accu_api_locations[city]
    return api_current_weather_accu(location)


def convert_weather_to_onehot(id):
    return np.array([elem == id for elem in id_list]).astype(int)


def format_to_nn_input(obj, exclusion_list=None):
    if exclusion_list is None:
        exclusion_list = []
    res = np.array([])
    for key, value in obj.items():
        if key in exclusion_list:
            continue
        if key == "weather":
            res = np.append(res, convert_weather_to_onehot(value))
        else:
            res = np.append(res, value)
    return res.astype(np.float32)


def process_image(img, location, size=224):
    img = img.crop(image_crops[location])
    img = img.resize((size, size))
    return np.asarray(img)


def clean_dataset(location):
    for file in os.listdir(error_dir):
      	if file.endswith(".txt"):
            os.remove(os.path.join(images_dir[location], file.replace(".txt", ".jpg")))
            os.remove(os.path.join(data_dir[location], file.replace(".txt", ".json")))
            os.remove(os.path.join(data_dir_accu[location], file.replace(".txt", ".json")))
            os.remove(os.path.join(error_dir, file))
            print(f"Exception {file} treated")
