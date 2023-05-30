from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from PIL import Image


def get_luck_score(luck_df):
    inverse_scores = [f"Q91_{i}" for i in [9, 15]]
    luck_df.loc[:, inverse_scores] = 7 - luck_df.loc[:, inverse_scores]
    luck_vec = np.array([1, 1, 1, 1, 1, 0, 1, 0, -1, 1, 1, 1, 0, 1, -1])
    return pd.DataFrame({"bigl": luck_vec @ luck_df.values.T})


def get_gambling_score(gambling_df):
    gambling_exp_vec = np.zeros(gambling_df.columns.size)
    gambling_exp_vec[[0, 5, 10, 15]] = 1
    illusion_of_control_vec = np.zeros(gambling_df.columns.size)
    illusion_of_control_vec[[2, 7, 12, 17]] = 1
    predictive_control_vec = np.zeros(gambling_df.columns.size)
    predictive_control_vec[[3, 8, 13, 18, 21, 22]] = 1
    inability_to_stop_vec = np.zeros(gambling_df.columns.size)
    inability_to_stop_vec[[1, 6, 11, 16, 20]] = 1
    interpretive_bias_vec = np.zeros(gambling_df.columns.size)
    interpretive_bias_vec[[4, 9, 14, 19]] = 1
    return pd.DataFrame({"gambling_expectancies": gambling_exp_vec @ gambling_df.values.T,
                         "illusion_of_control": illusion_of_control_vec @ gambling_df.values.T,
                         "predictive_control": predictive_control_vec @ gambling_df.values.T,
                         "inability_to_stop_gambling": inability_to_stop_vec @ gambling_df.values.T,
                         "interpretive_bias": interpretive_bias_vec @ gambling_df.values.T})


def get_rei_score(rei_df):
    inverse_scores = [f"questionnaire 1_{i}" for i in
                      [1, 2, 3, 4, 5, 11, 13, 15, 17, 19, 21, 26, 28, 30, 34, 36, 37, 38]]
    rei_df.loc[:, inverse_scores] = 6 - rei_df.loc[:, inverse_scores]
    rational_ability_vec = np.zeros(rei_df.columns.size)
    rational_ability_vec[[0, 3, 7, 12, 13, 16, 24, 26, 29, 38]] = 1
    rational_engagement_vec = np.zeros(rei_df.columns.size)
    rational_engagement_vec[[1, 5, 9, 15, 19, 25, 27, 31, 32, 39]] = 1
    experimental_ability_vec = np.zeros(rei_df.columns.size)
    experimental_ability_vec[[2, 4, 17, 18, 20, 33, 34, 35, 36, 37]] = 1
    experimental_engagement_vec = np.zeros(rei_df.columns.size)
    experimental_engagement_vec[[6, 8, 10, 11, 14, 21, 22, 23, 28, 30]] = 1
    return pd.DataFrame({"rational_ability": rational_ability_vec @ rei_df.values.T,
                         "rational_engagement": rational_engagement_vec @ rei_df.values.T,
                         "experimental_ability": experimental_ability_vec @ rei_df.values.T,
                         "experimental_engagement": experimental_engagement_vec @ rei_df.values.T})


def process_monetary_questionnaires(df):
    luck_df = df[[f"Q91_{k}" for k in range(1, 16)]].replace(
        ["Strongly disagree", "Slightly disagree", "Somewhat disagree", "Somewhat agree", "Slightly agree",
         "Strongly agree"], [1, 2, 3, 4, 5, 6])
    gambling_df = df[[f"Q79_{k}" for k in range(1, 24)]].replace(
        ['Strongly Disagree', 'Moderately Disagree', 'Mildly Disagree', 'Neither Agree nor Disagree',
         'Mildly Agree', 'Moderately Agree', 'Strongly Agree'], list(range(1, 8)))
    rei_df = df[[f"questionnaire 1_{k}" for k in range(1, 41)]].replace(
        ['Completely True\n5\n', '4', '3', '2', 'Completely False\n\n1\n'],
        [5, 4, 3, 2, 1])
    luck_scores = get_luck_score(luck_df)
    gambling_scores = get_gambling_score(gambling_df)
    rei_scores = get_rei_score(rei_df)
    return pd.concat([df, luck_scores, gambling_scores, rei_scores], axis=1)


def load_data(filename: str):
    print(f"==================== {filename} ====================")
    path = RAW_DATA_PATH / filename
    df: pd.DataFrame = pd.read_csv(path)
    df = df.iloc[2:, :]
    df = df.apply(lambda s: pd.to_numeric(s, errors='ignore'))
    df = df[df["PROLIFIC_PID"].str.len() == 24]
    # df = df.drop_duplicates(subset='PROLIFIC_PID', keep="first")
    df = df[df['Progress'] == 100]
    print(f"number of subjects before preprocessing: {len(df)}")
    df = df[df["attention check 1_1"].isin([np.nan])]
    df = df[(df["attention check 1_2"].str.lower().str.strip() == "yes")]
    print(f"number of subjects after removing attention check fails: {len(df)}")

    x_columns = [name for name in df.columns if name.endswith("_x")]
    y_columns = [name for name in df.columns if name.endswith("_y")]

    df.loc[:, 'x'] = df[x_columns].bfill(axis=1).iloc[:, 0]
    df.loc[:, 'y'] = df[y_columns].bfill(axis=1).iloc[:, 0]

    relevant_columns = ["PROLIFIC_PID", "age", "gender", "education level", "pcChoice", "pcResult", "x", "y",
                        "pcChoiceImage", "Duration (in seconds)"]
    column_renaming = ["id", "age", "gender", "education", "pc_choice", "pc_result", "x", "y", "pc_choice_image",
                       "duration"]

    if filename == "run1_monetary.csv" or filename == "run1_shapes.csv":
        df = df[pd.to_numeric(df['age'], errors='coerce') < 100]
        df = process_monetary_questionnaires(df)
        relevant_columns += ["bigl", "gambling_expectancies", "illusion_of_control", "predictive_control",
                             "inability_to_stop_gambling", "interpretive_bias", "rational_ability",
                             "rational_engagement", "experimental_ability", "experimental_engagement"]

    df = df[relevant_columns]
    df = df.rename(columns={relevant_columns[i]: column_renaming[i] for i in range(len(column_renaming))})
    df["pc_choice_x"] = 0
    df["pc_choice_y"] = 0
    df["dist"] = 0
    df = df.dropna(subset=column_renaming)
    print(f"number of subjects after removing submissions with missing data: {len(df)}")

    mean_duration, std_duration = df['duration'].mean(), df['duration'].std()
    df = df[df['duration'] > mean_duration - 2.5 * std_duration]
    df = df[df['duration'] < mean_duration + 2.5 * std_duration]
    mean_duration = df['duration'].mean()
    df = df.drop(columns=["duration"])
    print(f"number of subjects after removing submissions that exceed duration thresholds: {len(df)}")
    if "run1_shapes" in filename or "gamblers" in filename:
        df["x"] *= 500 / 550
        df["y"] *= 500 / 550
    radius = (EXP2_IMAGE_SIZE if "run2" in filename else EXP1_IMAGE_SIZE) // 2
    in_circ_idx = (((df["x"].astype(int) - radius) ** 2) + ((df["y"].astype(int) - radius) ** 2)) <= (radius ** 2)
    df = df.loc[in_circ_idx, :]
    print(f"number of subjects after removing submissions that did not choose within the circle: {len(df)}")
    print(f"Average duration: {mean_duration}\n")
    return df.reset_index(drop=True)


def get_image_from_url(url, key):
    if key not in images:
        images[key] = Image.open(BytesIO(requests.get(url).content))
        images[key].save(IMAGE_PATH / (str(key) + ".png"))

        def onclick(event):
            global CHOICE_LOCATIONS
            CHOICE_LOCATIONS[key] = (event.xdata, event.ydata)
            print(f"PC Choice location for {key}:({event.xdata}, {event.ydata})")
            plt.gcf().canvas.mpl_disconnect(cid)
            plt.close()

        if key not in CHOICE_LOCATIONS:
            plt.imshow(images[key])
            global cid
            cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
            plt.show(block=True)
    return images[key]


# %% load data and preprocess
data_frames = dict()
images = dict()

data_filenames = os.listdir(DATA_PATH / "raw")
for filename in data_filenames:
    if filename.endswith(".csv"):
        cid = None
        df = load_data(filename)
        for i in range(len(df)):
            location = df.loc[df.index[i], 'pc_choice']
            get_image_from_url(df.loc[df.index[i], 'pc_choice_image'], df.loc[df.index[i], 'pc_choice'])
            df.loc[df.index[i], ['pc_choice_x', 'pc_choice_y']] = CHOICE_LOCATIONS[location]
        df["dist"] = np.sqrt(((df["x"] - df["pc_choice_x"]) ** 2) + ((df["y"] - df["pc_choice_y"]) ** 2))
        data_frames[filename] = df
        df.to_csv(PROCESSED_DATA_PATH / filename, index=False)
