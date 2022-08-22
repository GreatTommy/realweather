import time
from datetime import datetime
import sys
from .data import *


def data_collector(locations, delay_mins=15, n_hours=None, log=True):
    n_iters = int(n_hours * 60 / delay_mins) if n_hours else sys.maxsize
    for it in range(n_iters):
        if log:
            ts = datetime.now().timestamp().__int__()
            print(f"{ts} > Iter {it + 1}/{n_iters}")
        for location in locations:
            if log:
                print(f"{ts} > {location}")
            ts, _, _ = get_data(location, save_path=f"meteo_{location}")
            if log:
                print(f"{ts} > {location} done")
        if log:
            print(f"{ts} > Done")
        time.sleep(delay_mins * 60)
