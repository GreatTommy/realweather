import itertools
import time
from .constants import *
import requests
from datetime import datetime
from random import randrange


def api_request(uri, delay=10):
    sc = 0
    while sc != 200:
        response = requests.get(uri)
        sc = response.status_code
        if sc != 200:
            time.sleep(delay)
    return response.json()


def api_current_weather(coords):
    key = owm_api_key
    # updates every 10 min 06 sec
    lat, lon = coords
    exclude = "minutely,hourly,daily,alerts"
    uri = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={exclude}&appid={key}&units=metric"
    obj = api_request(uri)
    return obj["current"]


def api_geocoding(city):
    key = owm_api_key
    uri = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={key}"
    obj = api_request(uri)[0]
    return obj["lat"], obj["lon"]


def api_weather_forecast(coords):
    key = owm_api_key
    lat, lon = coords
    exclude = "minutely,alerts"
    uri = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={exclude}&appid={key}&units=metric"
    obj = api_request(uri)
    sun_times = [
        {"dt": day["dt"], "sunrise": day["sunrise"], "sunset": day["sunset"]}
        for day in obj["daily"]
    ]
    for hour, day in itertools.product(obj["hourly"], sun_times):
        if (
            datetime.fromtimestamp(hour["dt"]).date()
            >= datetime.fromtimestamp(day["dt"]).date()
        ):
            hour["sunrise"] = day["sunrise"]
            hour["sunset"] = day["sunset"]
    return obj["hourly"]

def api_location_accu(city):
    key = accu_api_keys[randrange(len(accu_api_keys))]
    uri = f"http://dataservice.accuweather.com/locations/v1/cities/search?apikey={key}&q={city}"
    obj = api_request(uri)[0]
    return obj["Key"]

def api_current_weather_accu(location):
    key = accu_api_keys[randrange(len(accu_api_keys))]
    # updates every 10 min 06 sec
    uri = f"http://dataservice.accuweather.com/currentconditions/v1/{location}?apikey={key}&details=true"
    return api_request(uri)[0]