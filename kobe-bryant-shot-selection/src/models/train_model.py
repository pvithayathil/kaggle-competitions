import os
from dotenv import find_dotenv, load_dotenv
import logging
import pandas as pd
from scipy import sparse
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


load_dotenv(find_dotenv())
PROCESSED_PATH = os.getcwd() + os.getenv("PROCESSED_PATH")


def read_target_data(input_path: str, is_train: bool) -> pd.DataFrame:
    """

    Args:
        input_path: location of processed data
        is_train: specifies if reading train or test

    Returns:
        df:  pre-processed dataframe ready to be made into features
    """
    if is_train:
        target = pd.read_csv(input_path + "/train_target.csv")

    else:
        target = pd.read_csv(input_path + "/test_target.csv")

    return target


def read_features_data(input_path: str, is_train: bool) -> sparse.csr_matrix:
    """

    Args:
        input_path: location of processed data
        is_train: specifies if reading train or test

    Returns:
        data_matrix: features matrix ready for training
    """
    if is_train:
        data_matrix = sparse.load_npz(input_path + "/train.npz")

    else:
        data_matrix = sparse.load_npz(input_path + "/test.npz")

    return data_matrix


def train(df_target: pd.DataFrame, train: sparse.csr_matrix, output_path: str):
    """

    Args:
        df_target: dataframe of whether shot is made or not
        train: features matrix used for training

    Returns:
        side effect: trains model and saves it as a model file
    """
    """
    classifier_pipeline = Pipeline(
        steps=[("rfc", RandomForestClassifier(random_state=0))]
    )

    # Declare dynamic parameters here
    pipeline_params = {
        "rfc__max_depth": [3, 5, 7, 9, 11],
        "rfc__max_features": ["sqrt", "log2"],
        "rfc__max_leaf_nodes": [6, 8, 10],
    }
    """

    classifier_pipeline = Pipeline(
        steps=[("xgb", XGBClassifier(objective= 'binary:logistic',
                                     eval_metric='aucpr',
                                     random_state=0))]
    )

    # Score:
    # {'xgb__learning_rate': 0.01, 'xgb__max_depth': 5, 'xgb__n_estimators': 1000}
    # weight avg .70 .69 .68

    # Declare dynamic parameters here
    pipeline_params = {
    'xgb__n_estimators': [10, 50,100,1000],
    'xgb__max_depth': [3, 5, 9],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    }

    search = GridSearchCV(
        classifier_pipeline,
        pipeline_params,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=False,
    )
    search.fit(train, df_target)

    # model performance stats
    print("Best parameter (ROC AUC score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    print(classification_report(df_target, search.predict(train)))

    # save the model to disk
    filename = "/finalized_model.sav"
    pickle.dump(search, open(output_path + filename, "wb"))


def main(input_filepath, output_filepath):
    """"""
    logger = logging.getLogger(__name__)
    logger.info("training model")

    # Read Data
    train_matrx = read_features_data(input_filepath, True)
    df_target = read_target_data(input_filepath, True)

    train(df_target, train_matrx, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(PROCESSED_PATH, PROCESSED_PATH)
