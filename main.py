import sys
import pdb

from tqdm import tqdm
from rich import print as rprint
import arff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
import lightgbm as lgb


def rmse(x_vec, y_vec):
    rmse_ = np.sqrt(np.sum((x_vec - y_vec) ** 2))
    rmse_ = np.sqrt(np.mean((x_vec - y_vec) ** 2))
    # rmse_ = np.sum(np.abs((x_vec - y_vec)))

    return rmse_


def make_csv():
    """
    This function
        - loads the data set
        - makes adjustments to the data set
    """

    # Load risk features / frequency data
    data_freq = arff.load("freMTPL2freq.arff")

    df_freq = pd.DataFrame(
        data_freq,
        columns=[
            "IDpol",
            "ClaimNb",
            "Exposure",
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
        ],
    )

    # Load claim amounts
    data_sev = arff.load("freMTPL2sev.arff")

    df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])

    # Sum all the claims of a contract
    df_sev = df_sev.groupby(["IDpol"], as_index=False).sum()

    # Remove the outliers
    up_lim = df_sev["ClaimAmount"].quantile(0.75)
    # up_lim = 5e3

    df = df_sev[df_sev["ClaimAmount"] < up_lim]

    # Combine the claim data with frequency data
    df = df_freq.set_index("IDpol").join(df_sev.set_index("IDpol"))

    df["ClaimAmount"] = df["ClaimAmount"].replace(to_replace=np.nan, value=0.0)

    # # Combine the claim data with frequency data
    # df = df_sev.set_index("IDpol").join(df_freq.set_index("IDpol"))

    # df["ClaimAmount"] = df["ClaimAmount"].replace(to_replace=np.nan, value=0.0)

    # # More visualization
    # df_sev = df_sev[df_sev["ClaimAmount"] < up_lim]

    # Claim per unit of time
    df["ClaimExp"] = df["ClaimAmount"] / df["Exposure"]

    # # Make categorical data categorical with integer values
    # keys_list = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]
    # for key_ in keys_list:
    #     df[key_] = df[key_].astype("category").cat.codes

    # Some transformations to the regressors
    df["VehAgeSq"] = df["VehAge"] ** 2
    df["DrivAgeSq"] = df["DrivAge"] ** 2

    df_tmp = pd.get_dummies(df["Area"], drop_first=True)
    df = pd.concat([df, df_tmp], axis=1)

    df_tmp = pd.get_dummies(df["VehGas"], drop_first=True)
    df = pd.concat([df, df_tmp], axis=1)

    df_tmp = pd.get_dummies(df["VehBrand"], drop_first=True)
    df = pd.concat([df, df_tmp], axis=1)

    df_tmp = pd.get_dummies(df["Region"], drop_first=True)
    df = pd.concat([df, df_tmp], axis=1)

    df["Density"] = np.log(df["Density"])

    # df["ClaimNb"] = df["ClaimNb"].astype("bool")

    # Save as csv
    df.to_csv("claim_freq_data.csv")

    # Return the data frame
    return df


def load_csv():
    """Reads the csv data into a data frame"""

    df = pd.read_csv("claim_freq_data.csv")

    return df


def glm_tweedie(df):
    """
    This function trains the generalized linear model with tweedie distribution
    """

    # Train and Test split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        df[
            [
                "Exposure",
                "ClaimNb",
                "'B'",
                "'C'",
                "'D'",
                "'E'",
                "'F'",
                "VehPower",
                "VehAge",
                "VehAgeSq",
                "DrivAge",
                "DrivAgeSq",
                "BonusMalus",
                "'B2'",
                "'B3'",
                "'B4'",
                "'B5'",
                "'B6'",
                "'B10'",
                "'B11'",
                "'B12'",
                "'B13'",
                "'B14'",
                "Regular",
                "Density",
                # "Region",
            ]
        ],
        df["ClaimExp"],
        # df["ClaimAmount"],
    )

    # GLM model with Tweedie distribution
    tw_mod = linear_model.TweedieRegressor(
        power=1.5,
        alpha=1e-2,
        fit_intercept=True,
        link="auto",
        solver="newton-cholesky",  # "lbfgs",
        max_iter=int(1e3),
        tol=1e-4,
        warm_start=False,
        verbose=1,
    )

    # Fit the model
    # tw_mod.fit(X_train, Y_train)
    # tw_mod.fit(X_train, Y_train)
    tw_mod.fit(X_train, Y_train)

    rprint(f"In-sample fit is {tw_mod.score(X_train, Y_train)}")
    rprint(f"Out-of-sample fit is {tw_mod.score(X_test, Y_test)}")

    rmse_in = rmse(Y_train, tw_mod.predict(X_train))
    rprint(f"In-sample rmse is {rmse_in}")

    rmse_out = rmse(Y_test, tw_mod.predict(X_test))
    rprint(f"Out-of-sample rmse is {rmse_out}")

    # # Linear regression
    # lin_mod = linear_model.LinearRegression(fit_intercept=True, copy_X=True)

    # lin_mod.fit(X_train, Y_train)

    # rprint(f"In-sample fit is {lin_mod.score(X_train, Y_train)}")
    # rprint(f"Out-of-sample fit is {lin_mod.score(X_test, Y_test)}")

    # # Some info
    # rprint(f"Intercept is {tw_mod.intercept_}\n")

    # rprint(f"The coefficients are: \n{tw_mod.coef_}\n")

    # # Fit the GLM with Tweedie distribution
    # int_len = 100
    # score_vec = np.empty(int_len, dtype=float)
    # interval = np.linspace(start=0.01, stop=1.99, num=int_len)

    # for ii, pow in enumerate(tqdm(interval)):
    #     tw_mod = linear_model.TweedieRegressor(
    #         power=pow,
    #         alpha=1e-2,
    #         fit_intercept=True,
    #         link="auto",
    #         solver="newton-cholesky",  # "lbfgs",
    #         max_iter=1000,
    #         tol=1e-4,
    #         warm_start=False,
    #         verbose=0,
    #     )

    #     tw_mod.fit(X_train, Y_train)

    #     score_vec[ii] = tw_mod.score(X_test, Y_test)

    # np.savetxt("score.csv", score_vec)
    # np.savetxt("power.csv", interval)


def train_gbm(df):

    # Train and Test split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        df[
            [
                "'B'",
                "'C'",
                "'D'",
                "'E'",
                "'F'",
                "VehPower",
                "VehAge",
                "VehAgeSq",
                "DrivAge",
                "DrivAgeSq",
                "BonusMalus",
                "'B2'",
                "'B3'",
                "'B4'",
                "'B5'",
                "'B6'",
                "'B10'",
                "'B11'",
                "'B12'",
                "'B13'",
                "'B14'",
                "Regular",
                "Density",
                # "Region",
            ]
        ],
        df["ClaimExp"],
    )

    gbm_mod = lgb.LGBMRegressor(
        boosting_type="gbdt",
        num_leaves=21,
        max_depth=-1,
        learning_rate=0.3,
        n_estimators=10,
        subsample_for_bin=int(2e5),
        objective="regression",
        # class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=20.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
    )

    gbm_mod.fit(X_train, Y_train)

    score_in = gbm_mod.score(X_train, Y_train)
    score_out = gbm_mod.score(X_test, Y_test)

    rprint(f"In-sample fit: {score_in} \nOut-of-sample fit: {score_out}")


if __name__ == "__main__":
    """
    Main instructions
    """

    if len(sys.argv) > 1 and sys.argv[1] == "make_csv":
        make_csv()

    # Load the data set
    df = load_csv()

    # Train the glm model with tweedie
    glm_tweedie(df)

    # Train the lightgbm model
    train_gbm(df)
