from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import statsmodels.api as sm

plt.rcParams["axes.labelsize"] = "large"
data_frames = {"gamblers_different.csv": None,
               "gamblers_same.csv": None}
for k in data_frames.keys():
    data_frames[k] = pd.read_csv(DATA_PATH / "processed" / k)

same_df = data_frames["gamblers_same.csv"]
different_df = data_frames["gamblers_different.csv"]
# %% Spatial gamblers fallacy - do we have an effect for the location?
effect_of_location = distance_permutation_test_across_locations(same_df, n_perm=100000)
print("testing whether people move away from the virtual player's choice when searching for same shape:",
      effect_of_location)  # yes

effect_of_location = distance_permutation_test_across_locations(different_df, n_perm=100000)
print("testing whether people move away from the virtual player's choice when searching for a different shape:",
      effect_of_location)  # no

# %% Spatial gamblers fallcay heatmaps - all collapsed

# calculate rotations
rotations = dict()
rotations["bottom left"] = 0
rotations["top left"] = get_angle(CHOICE_LOCATIONS["bottom left"], CHOICE_LOCATIONS["top left"], EXP1_IMAGE_SIZE)
rotations["right"] = get_angle(CHOICE_LOCATIONS["bottom left"], CHOICE_LOCATIONS["right"], EXP1_IMAGE_SIZE)

# plot
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig: plt.Figure
axes = axes.ravel()
for i, (title, df) in enumerate([("Same", same_df), ("Different", different_df)]):
    res_x, res_y = [], []
    for loc in df["pc_choice"].unique():
        idx = (df["pc_choice"] == loc).to_numpy()
        x, y = df.loc[idx, "x"].to_numpy(), df.loc[idx, "y"].to_numpy()
        x, y = rotate(x, y, rotations[loc], EXP1_IMAGE_SIZE, EXP1_IMAGE_SIZE)
        res_x.append(x)
        res_y.append(y)
    x = np.concatenate(res_x)
    y = np.concatenate(res_y)
    plot_heatmap(x, EXP1_IMAGE_SIZE - y, EXP1_IMAGE_SIZE, IMAGE_PATH / "bottom left.png", axes[i])
    axes[i].set_title(y=-0.1, label=title, fontweight="bold")

for ax in axes:
    ax.axis('off')
savefig(fig, "spatial gamblers fallacy collapsed heatmaps")
# %% heatmaps - separate
nrows = same_df["pc_choice"].unique().size
fig, axes = plt.subplots(nrows, 2, figsize=(7, 1 + nrows * 3))
fig: plt.Figure
for j, (title, df) in enumerate([("Same", same_df), ("Different", different_df)]):
    res_x, res_y = [], []
    for i, loc in enumerate(sorted(df["pc_choice"].unique())):
        idx = (df["pc_choice"] == loc).to_numpy()
        x, y = df.loc[idx, "x"].to_numpy(), df.loc[idx, "y"].to_numpy()
        plot_heatmap(x, EXP1_IMAGE_SIZE - y, EXP1_IMAGE_SIZE, IMAGE_PATH / f"{loc}.png", axes[i, j])
        if i == 0:
            axes[i, j].set_title(label=title, fontweight="bold")
        if j == 0:
            axes[i, j].set_ylabel(ylabel=f"{loc}", fontweight="bold")

savefig(fig, "spatial gamblers fallacy heatmaps")

# %% plot distance kde
import seaborn as sns

same_cpy = same_df.copy()
different_cpy = different_df.copy()
same_cpy.loc[:, "Condition"] = "Same"
different_cpy.loc[:, "Condition"] = "Different"

joined_df = pd.concat([same_cpy, different_cpy], axis=0)
joined_df = joined_df.rename(columns={"dist": "Distance (pixels)"})
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.kdeplot(joined_df, x="Distance (pixels)", hue="Condition", ax=ax, clip=[0, 500],
            palette=["#FF0000", "#00A08A"], common_norm=False,
            hue_order=["Different", "Same"], linewidth=2)
savefig(fig, "spatial gamblers fallacy distance kde")

# %%
same_cpy = same_df.copy()
different_cpy = different_df.copy()
same_cpy.loc[:, "condition"] = 1
different_cpy.loc[:, "condition"] = 0
joined_df = pd.concat([same_cpy, different_cpy], axis=0)
reg_df = joined_df.loc[:, ["dist", "condition"]]
y = reg_df["dist"]
x = reg_df["condition"]
x = sm.add_constant(x)
res = sm.OLS(y, x).fit()
print(res.summary())
