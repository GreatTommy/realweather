from datetime import datetime
import json
import os
from .constants import *
from .utils import *
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler
from math import ceil
from sklearn.preprocessing import StandardScaler


def get_data(location, save_path=None):
    img_data = get_live_image(imgs_urls[location])
    ts = datetime.now().timestamp().__int__()
    if save_path:
        img_name = f"{ts}_{location}.jpg"
        with open(os.path.join(save_path, "images", img_name), "wb") as handler:
            handler.write(img_data)
    city = location.split("_")[0]
    weather = get_live_weather(city)
    if save_path:
        js = json.dumps(weather)
        file_name = f"{ts}_{location}.json"
        with open(os.path.join(save_path, "weather_data", file_name), "w") as handler:
            handler.write(js)
    weather = get_live_weather_accu(city)
    if save_path:
        js = json.dumps(weather)
        file_name = f"{ts}_{location}.json"
        with open(os.path.join(save_path, "weather_data_accu", file_name), "w") as handler:
            handler.write(js)
    return ts, img_data, weather


def convert_weather_data_format(obj):
    return {
        "dt": obj["dt"],
        "sunrise": obj["sys"]["sunrise"],
        "sunset": obj["sys"]["sunset"],
        "temp": obj["main"]["temp"],
        "feels_like": obj["main"]["feels_like"],
        "pressure": obj["main"]["pressure"],
        "humidity": obj["main"]["humidity"],
        "dew_point": 0, # None,
        "uvi": 0, # None,
        "clouds": obj["clouds"]["all"],
        "visibility": obj["visibility"],
        "wind_speed": obj["wind"]["speed"],
        "wind_deg": obj["wind"]["deg"],
        "weather": obj["weather"],
    }


def format_weather_forecast(obj):
    if "coord" in obj:
        obj = convert_weather_data_format(obj)
    if "pop" in obj:
        pop = obj["pop"]
    else:
        pop = 1 if obj["weather"][0]["id"] < 700 else 0
    return {
        "dist_sunrise": obj["dt"] - obj["sunrise"],
        "dist_sunset": obj["dt"] - obj["sunset"],
        "temp": obj["temp"],
        "feels_like": obj["feels_like"],
        "pressure": obj["pressure"],
        "humidity": obj["humidity"],
        "dew_point": obj["dew_point"],
        "uvi": obj["uvi"],
        "clouds": obj["clouds"],
        "visibility": obj["visibility"],
        "wind_speed": obj["wind_speed"],
        "wind_deg": obj["wind_deg"],
        "pop": pop,
        "weather": obj["weather"][0]["id"],
    }
  
def format_weather_forecast_accu(obj):
    return {
        "temp": obj["Temperature"]["Metric"]["Value"],
        "pressure": obj["Pressure"]["Metric"]["Value"],
        "humidity": obj["RelativeHumidity"],
        "dew_point": obj["DewPoint"]["Metric"]["Value"],
        "uvi": obj["UVIndex"],
        "clouds": obj["CloudCover"],
        "visibility": obj["Visibility"]["Metric"]["Value"],
        "ceiling": obj["Ceiling"]["Metric"]["Value"],
        "precipitation": obj["HasPrecipitation"] * 1,
        "weather": obj["WeatherIcon"],
    }
  

def open_weather_data(location, scale_data=True, exclusion_list=None):
    dataset = []
    for file in sorted(os.listdir(data_dir[location])):
        if file.endswith(".json"):
            with open(os.path.join(data_dir[location], file)) as json_file:
                data = json.load(json_file)
            data = format_to_nn_input(format_weather_forecast(data), exclusion_list)
            dataset.append(data)
    if scale_data:
        return scale_weather_data(np.asarray(dataset))
    return np.asarray(dataset)
  
def open_weather_data_accu(location, scale_data=True, exclusion_list=None):
    dataset = []
    for file in sorted(os.listdir(data_dir_accu[location])):
        if file.endswith(".json"):
            with open(os.path.join(data_dir_accu[location], file)) as json_file:
                data = json.load(json_file)
            data = format_to_nn_input(format_weather_forecast_accu(data), exclusion_list)
            dataset.append(data)
    if scale_data:
        return scale_weather_data(np.asarray(dataset))
    return np.asarray(dataset)


def open_images(location):
    dataset = []
    for file in sorted(os.listdir(images_dir[location])):
        if file.endswith(".jpg"):
            try:
                img = Image.open(os.path.join(images_dir[location], file))
                arr = process_image(img, location)
                img.close()
            except Exception as e:
                print(f"Exception with image {file}")
                with open(
                    os.path.join(error_dir, file.replace(".jpg", ".txt")), "w"
                ) as f:
                    f.write(str(e))
                continue
            arr = (arr.astype(np.float32) / 255)
            dataset.append(arr)
    return np.moveaxis(np.asarray(dataset), -1, 1)


class RealWeatherDataset(Dataset):
    def __init__(self, images, data, transform=None, device=None):
        self.images = torch.from_numpy(images)
        self.data = torch.from_numpy(data)
        self.transform = transform
        if device:
            self.images = self.images.to(device)
            self.data = self.data.to(device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        data = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, data


class FastDataLoader:
    def __init__(
        self, dataset, batch_size=64, shuffle=False, sampler=None, drop_last=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if sampler is None:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        self.sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        self.idx_iterator = iter(self.sampler)
        return self

    def __next__(self):
        idx = next(self.idx_iterator)
        return self.dataset[idx]

    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        else:
            return ceil(length / self.batch_size)

def scale_weather_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

