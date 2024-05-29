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
    """
    Root Mean Squared Error (RMSE) for forecast/prediction accuracy
    """
    rmse_ = np.sqrt(np.mean((x_vec - y_vec) ** 2))

    return rmse_


def make_csv():
    """
    This function
        - loads the data set
        - makes adjustments to the data set, i.e. feature engineering
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

    # Combine the claim data with frequency data
    df = df_freq.set_index("IDpol").join(df_sev.set_index("IDpol"))

    df["ClaimAmount"] = df["ClaimAmount"].replace(to_replace=np.nan, value=0.0)

    df.to_csv("ins_claims.csv")


def load_csv():
    """Reads the csv data into a data frame"""

    # Load the csv file
    df = pd.read_csv("ins_claims.csv", index_col=0)

    # Remove the outliers
    up_lim = df["ClaimAmount"].quantile(0.995)
    df = df[df["ClaimAmount"] < up_lim]

    # up_lim = df["VehAge"].quantile(0.995)
    # df = df[df["VehAge"] < up_lim]

    # === New data frame ===
    new_df = pd.DataFrame()

    # Claim per unit time
    new_df["ClaimExp"] = df["ClaimAmount"] / df["Exposure"]

    # Area
    df_tmp = pd.get_dummies(df["Area"], drop_first=True)
    new_df = pd.concat([new_df, df_tmp], axis=1)

    # Vehicle Power
    new_df["VehPower"] = df["VehPower"]

    # Vehicle Age
    new_df["VehAge"] = df["VehAge"]
    # new_df["VehAgeSq"] = df["VehAge"] ** 2

    # Driver Age
    new_df["DrivAge"] = df["DrivAge"]
    # new_df["DrivAgeSq"] = df["DrivAge"] ** 2

    # Bonus Malus
    new_df["BonusMalus"] = df["BonusMalus"]

    # # Vehicle brand
    # df_tmp = pd.get_dummies(df["VehBrand"], drop_first=True)
    # new_df = pd.concat([new_df, df_tmp], axis=1)

    # # Vehicle gas-type
    # df_tmp = pd.get_dummies(df["VehGas"], drop_first=True)
    # new_df = pd.concat([new_df, df_tmp], axis=1)

    # Density
    new_df["Density"] = df["Density"]

    # # Region
    # df_tmp = pd.get_dummies(df["Region"], drop_first=True)
    # new_df = pd.concat([new_df, df_tmp], axis=1)

    # Claim number/frequency
    # new_df["ClaimYes"] = (df["ClaimAmount"] > 0.0).astype(int)
    # new_df["ClaimNb"] = df["ClaimNb"]
    # new_df["ClaimNb"] = df["ClaimNb"].astype("bool")

    # # Everything is a interaction variable with "ClaimYes"
    # new_df.mul(new_df["ClaimYes"], axis="index")

    # # Classifier - ClaimYes or ClaimNo
    # logi_mod = linear_model.LogisticRegression(
    #     penalty="l2",
    #     tol=1e-4,
    #     C=1.0,
    #     solver="newton-cholesky",
    #     max_iter=10000,
    #     verbose=1,
    # )

    # logical_vec = (new_df.columns != "ClaimExp") * (
    #     new_df.columns != "ClaimYes"
    # )
    # logi_mod.fit(new_df.loc[:, logical_vec], new_df["ClaimYes"])

    # new_df["ClaimYes"] = logi_mod.predict(new_df.loc[:, logical_vec])

    return new_df


def glm_tweedie(df):
    """
    Trains the generalized linear model with Tweedie distribution
    """

    # Train and Test split
    logi = df.columns != "ClaimExp"

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        df.loc[:, logi], df["ClaimExp"], shuffle=False
    )

    # GLM model with Tweedie distribution
    tw_mod = linear_model.TweedieRegressor(
        power=1.2,
        alpha=1e-2,
        fit_intercept=True,
        link="auto",
        solver="newton-cholesky",
        max_iter=int(1e3),
        tol=1e-4,
        warm_start=False,
        verbose=1,
    )

    # Fit the model
    tw_mod.fit(X_train, Y_train)

    rprint(f"\n\nTweedie Regression:")
    rprint(f"In-sample fit is {tw_mod.score(X_train, Y_train)}")
    rprint(f"Out-of-sample fit is {tw_mod.score(X_test, Y_test)}")

    rmse_in = rmse(Y_train, tw_mod.predict(X_train))
    rprint(f"In-sample rmse is {rmse_in}")

    rmse_out = rmse(Y_test, tw_mod.predict(X_test))
    rprint(f"Out-of-sample rmse is {rmse_out}")

    # Plot the coefficients of the features
    res = {"names": tw_mod.feature_names_in_, "vals": tw_mod.coef_}

    df_res = pd.DataFrame(data=res)

    df_res = df_res.sort_values(by=["vals"])

    df_res.plot(x="names", y="vals", kind="bar")

    plt.show()

    # Linear regression
    lin_mod = linear_model.LinearRegression(fit_intercept=True, copy_X=True)

    lin_mod.fit(X_train, Y_train)

    rprint(f"\n\nLinear Regression:")
    rprint(f"In-sample fit is {lin_mod.score(X_train, Y_train)}")
    rprint(f"Out-of-sample fit is {lin_mod.score(X_test, Y_test)}")

    rmse_in = rmse(Y_train, lin_mod.predict(X_train))
    rprint(f"In-sample rmse is {rmse_in}")

    rmse_out = rmse(Y_test, lin_mod.predict(X_test))
    rprint(f"Out-of-sample rmse is {rmse_out}\n\n")


def train_gbm(df):
    """Trains the Gradient boosting model"""

    # Train and Test split
    logi = df.columns != "ClaimExp"

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        df.loc[:, logi], df["ClaimExp"], shuffle=True
    )

    gbm_mod = lgb.LGBMRegressor(
        boosting_type="gbdt",
        learning_rate=0.1,
        n_estimators=100,
        importance_type="gain",
    )

    gbm_mod.fit(X=X_train, y=Y_train)

    lgb.plot_importance(gbm_mod)
    plt.show()

    score_in = gbm_mod.score(X_train, Y_train)
    score_out = gbm_mod.score(X_test, Y_test)

    rprint(f"\n\nGradient Boosting:")
    rprint(f"In-sample fit: {score_in}")
    rprint(f"Out-of-sample fit: {score_out}")

    rmse_in = rmse(Y_train, gbm_mod.predict(X_train))
    rprint(f"In-sample rmse is {rmse_in}")

    rmse_out = rmse(Y_test, gbm_mod.predict(X_test))
    rprint(f"Out-of-sample rmse is {rmse_out}")


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
