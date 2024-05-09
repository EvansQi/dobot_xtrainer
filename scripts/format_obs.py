import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def save_frame(
    folder: str,
    timestamp: int,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs["control"] = action  # add action to obs

    # make folder if it doesn't exist
    # folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder + str(timestamp) + ".pkl"
    print(recorded_file)

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

def save_action(recorded_file,action: np.ndarray):
    with open(recorded_file, "ab") as f:
        pickle.dump(action, f)


if __name__ == "__main__":
    save_path = "/home/dobot/gello/data/0202_172214/trace.pkl"
    # test write
    # act = [1,2,3,4,5,6]
    # for i in range(5):
    #     save_action(save_path,act)
    #     print(i)

    # test read
    with open(save_path, 'rb') as file:
        while True:
            try:
                loaded_data = pickle.load(file)

                # 打印加载的对象
                print(loaded_data)
            except EOFError:
                break



