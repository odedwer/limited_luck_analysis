import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

data_frames = {"run1_monetary.csv": None, "run1_shapes.csv": None}
for k in data_frames.keys():
    data_frames[k] = pd.read_csv(DATA_PATH / "processed" / k)

shapes_df = data_frames["run1_shapes.csv"]
monetary_df = data_frames["run1_monetary.csv"]

# %%
# =========================================================\
# Is there a spatial gambler's\hot hand fallacy?
# =========================================================
effect_of_location = distance_permutation_test_across_locations(shapes_df)
print("testing whether people move away from the virtual player's choice when searching for same shape:",
      effect_of_location)
# answer - yes.

# %%
# =========================================================
#  Is luck viewed as limited in space?
# =========================================================
# regression analysis
tmp_df = shapes_df.copy()
tmp_df["pc_result"] = "shape"

run1_df = pd.concat([monetary_df, tmp_df], axis=0)
reg_df = run1_df[["dist", "pc_result"]]
dummy_data = pd.get_dummies(reg_df).drop(columns=["pc_result_shape"])

y = dummy_data["dist"].values
x = dummy_data.iloc[:, 1:].astype(int)
X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
real_params = est2.params

n_perm = 10000
max_dist = np.zeros((n_perm,), dtype=float)
min_dist = np.zeros((n_perm,), dtype=float)
np.random.seed(97)
for i in tqdm(range(n_perm)):
    perm_y = np.random.permutation(y)
    params = sm.OLS(perm_y, X2).fit().params[1:].to_numpy()
    max_dist[i] = np.abs((params - params[:, None])).max()
    min_dist[i] = np.abs((params - params[:, None])).min()

five_minus_0 = np.abs(real_params["pc_result_5.0"] - real_params["pc_result_0.0"])
five_minus_1 = np.abs(real_params["pc_result_5.0"] - real_params["pc_result_1.0"])
one_minus_0 = np.abs(real_params["pc_result_1.0"] - real_params["pc_result_0.0"])

print(
    f"1-0 coef permutation - larger than max pvalue: {(one_minus_0 < max_dist).mean()}, smaller than min pvalue: {(one_minus_0 > min_dist).mean()}")
print(
    f"5-1 coef permutation - larger than max pvalue: {(five_minus_1 < max_dist).mean()}, smaller than min pvalue: {(five_minus_1 > min_dist).mean()}")
print(
    f"5-0 coef permutation - larger than max pvalue: {(five_minus_0 < max_dist).mean()}, smaller than min pvalue: {(five_minus_0 > min_dist).mean()}")


# %% Heatmaps - all collapsed

# calculate rotations
rotations = dict()
rotations["bottom left"] = 0
rotations["top left"] = get_angle(CHOICE_LOCATIONS["bottom left"], CHOICE_LOCATIONS["top left"], EXP1_IMAGE_SIZE)
rotations["right"] = get_angle(CHOICE_LOCATIONS["bottom left"], CHOICE_LOCATIONS["right"], EXP1_IMAGE_SIZE)

titles = dict()
title_kwargs = {
    0: {"label": "Bad luck", "color": "red", "fontweight": "bold"},
    1: {"label": "Mild luck", "color": "cyan", "fontweight": "bold"},
    5: {"label": "Good luck", "color": "green", "fontweight": "bold"},
}
# plot
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig: plt.Figure
axes = axes.ravel()
for i, res in enumerate(sorted(monetary_df["pc_result"].unique().astype(int))):
    res_x = []
    res_y = []
    for loc in monetary_df["pc_choice"].unique():
        idx = (monetary_df["pc_choice"] == loc).to_numpy() & (monetary_df["pc_result"] == res).to_numpy()
        x, y = monetary_df.loc[idx, "x"].to_numpy(), monetary_df.loc[idx, "y"].to_numpy()
        x, y = rotate(x, y, rotations[loc], EXP1_IMAGE_SIZE, EXP1_IMAGE_SIZE)
        res_x.append(x)
        res_y.append(y)
    x = np.concatenate(res_x)
    y = np.concatenate(res_y)
    plot_heatmap(x, EXP1_IMAGE_SIZE - y, EXP1_IMAGE_SIZE, IMAGE_PATH / "bottom left.png", axes[i])
    axes[i].set_title(y=-0.1, **title_kwargs[res])

res_x = []
res_y = []
for loc in shapes_df["pc_choice"].unique():
    idx = (shapes_df["pc_choice"] == loc).to_numpy()
    x, y = shapes_df.loc[idx, "x"].to_numpy(), shapes_df.loc[idx, "y"].to_numpy()
    x, y = rotate(x, y, rotations[loc], EXP1_IMAGE_SIZE, EXP1_IMAGE_SIZE)
    res_x.append(x)
    res_y.append(y)
x = np.concatenate(res_x)
y = np.concatenate(res_y)
plot_heatmap(x, EXP1_IMAGE_SIZE - y, EXP1_IMAGE_SIZE, IMAGE_PATH / "bottom left.png", axes[-1])
axes[-1].set_title(label="Control", y=-0.1, color="gray", fontweight="bold")
for ax in axes:
    ax.axis('off')
utils.savefig(fig, "exp1 collapsed heatmaps")
# %% heatmaps - separate
nrows = monetary_df["pc_choice"].unique().size
ncols = monetary_df["pc_result"].unique().size + 1
plt.rcParams["axes.labelsize"] = "large"
fig, axes = plt.subplots(nrows, ncols, figsize=(1 + ncols * 3, 1 + nrows * 3))
for j, res in enumerate(sorted(monetary_df["pc_result"].unique().astype(int))):
    for i, loc in enumerate(sorted(monetary_df["pc_choice"].unique())):
        idx = (monetary_df["pc_choice"] == loc).to_numpy() & (monetary_df["pc_result"] == res).to_numpy()
        x, y = monetary_df.loc[idx, "x"].to_numpy(), monetary_df.loc[idx, "y"].to_numpy()
        plot_heatmap(x, EXP1_IMAGE_SIZE - y, EXP1_IMAGE_SIZE, IMAGE_PATH / f"{loc}.png", axes[i, j])
        if not j:
            axes[i, j].set_ylabel(f"{loc}", fontweight="bold")
        if not i:
            axes[i, j].set_title(**title_kwargs[res])

for i, loc in enumerate(sorted(shapes_df["pc_choice"].unique())):
    idx = (shapes_df["pc_choice"] == loc).to_numpy()
    x, y = shapes_df.loc[idx, "x"].to_numpy(), shapes_df.loc[idx, "y"].to_numpy()
    plot_heatmap(x, EXP1_IMAGE_SIZE - y, EXP1_IMAGE_SIZE, IMAGE_PATH / f"{loc}.png", axes[i, -1])
    if not i:
        axes[i, -1].set_title(label="Control", color="gray", fontweight="bold")
savefig(fig, "exp1 heatmaps")

# %% Gaussian KDE distance histogram
import seaborn as sns

tmp_df = shapes_df.copy()
tmp_df["pc_result"] = "Control"

joined_df = pd.concat([monetary_df, tmp_df], axis=0)
joined_df["pc_result"].replace([0, 1, 5], ["Bad luck", "Mild luck", "Good luck"], inplace=True)
joined_df = joined_df.rename(columns={"pc_result": "Virtual player result", "dist": "Distance (pixels)"})
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.kdeplot(joined_df, x="Distance (pixels)", hue="Virtual player result", ax=ax, clip=[0, 500],
            palette=["#FF0000", "#5BBCD6", "#00A08A", "#899DA4"], common_norm=False,
            hue_order=["Bad luck", "Mild luck", "Good luck", "Control"], linewidth=2)
savefig(fig, "exp1 distance kde")

# %% Run regression with the questionnaire data
from statsmodels.formula.api import ols

tmp_df = shapes_df.copy()
tmp_df["pc_result"] = "Control"

joined_df = pd.concat([monetary_df, tmp_df], axis=0).dropna()
reg_df = joined_df.loc[:,
         ["dist", "pc_result", 'age', 'gender', 'education', 'bigl', 'gambling_expectancies', 'illusion_of_control',
          'predictive_control', 'inability_to_stop_gambling', 'interpretive_bias', 'rational_ability',
          'rational_engagement',
          'experimental_ability', 'experimental_engagement']]
reg_df["age"] = pd.to_numeric(reg_df["age"])
reg_df.loc[:, "pc_result"] = reg_df.loc[:, "pc_result"].replace([0, 1, 5], ['0', '1', '5'])

reg_df.loc[:, "grcs"] = reg_df.loc[:, ('gambling_expectancies', 'illusion_of_control', 'predictive_control',
                                       'inability_to_stop_gambling', 'interpretive_bias')].mean(axis=1)
reg_df.loc[:, "rei"] = reg_df.loc[:, ('rational_ability', 'rational_engagement',
                                      'experimental_ability', 'experimental_engagement')].mean(axis=1)
reg_df = reg_df.drop(columns=['gambling_expectancies', 'illusion_of_control', 'predictive_control',
                              'inability_to_stop_gambling', 'interpretive_bias', 'rational_ability',
                              'rational_engagement', 'experimental_ability', 'experimental_engagement'])
dummy_data = pd.get_dummies(reg_df).drop(
    columns=["pc_result_Control", "gender_Female", "education_Less than high school"])
# conditions_df = dummy_data[[f"pc_result_{i}" for i in ['0', '1', '5']]]
# predictors_df = dummy_data.drop(columns=[f"pc_result_{i}" for i in ['0', '1', '5']] + ["dist"])
# interaction_values = (conditions_df.values[:, :, None] * predictors_df.values[:, None, :]).reshape(
#     (len(conditions_df), -1))
# interaction_df = pd.DataFrame(interaction_values,
#                               columns=[f"{cond} * {pred}" for cond in conditions_df.columns for pred in
#                                        predictors_df.columns])
# dummy_data = pd.concat([dummy_data.reset_index(drop=True), interaction_df.reset_index(drop=True)], axis=1)
y = dummy_data["dist"].values
x = dummy_data.iloc[:, 1:].astype(float)

X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
res = est.fit()
print(res.summary())

# %%
