from dotenv import find_dotenv, load_dotenv
import os
import logging
import pandas as pd
import numpy as np
from pandasql import sqldf
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())
RAW_DATA_PATH = os.getcwd() + os.getenv("RAW_DATA_PATH")
PROCESSED_PATH = os.getcwd() + os.getenv("PROCESSED_PATH")
FEATURES_QUERY = """
    WITH game_date_features as
    (
    SELECT 
    date(d1.game_date) as game_date,
    CASE
    WHEN
    COUNT(DISTINCT d2.game_date) >=1 THEN True
    ELSE False
    END game_is_back_to_back,
    COUNT(DISTINCT d3.game_date) num_game_last_7_days
    FROM dates_table as d1
    LEFT JOIN
    dates_table d2
    ON 
    date(d1.game_date) = date(d2.game_date, '+1 day')
    LEFT JOIN
    dates_table d3
    ON 
    date(d1.game_date, '-6 day') <= date(d3.game_date) AND
    date(d1.game_date) > date(d3.game_date)
    GROUP BY d1.game_date
    ),
    dates_table as
    (
    SELECT game_date FROM data GROUP BY game_date
    )
    SELECT
    data.*, 
    (60 * data.minutes_remaining + data.seconds_remaining) seconds_from_period_end,
    CASE 
    WHEN 
    data.period >= 4 AND
    data.seconds_remaining < 24 THEN True
    ELSE
    False
    END shot_is_in_possibly_clutch_moment,
    SUM(is_shot) OVER (PARTITION BY game_id
                         ORDER BY period asc, minutes_remaining desc, seconds_remaining desc) - 1 AS num_shots_taken_in_game,
    SUM(shot_made_flag) OVER (PARTITION BY game_id
                         ORDER BY period asc, minutes_remaining desc, seconds_remaining desc) - shot_made_flag AS num_shots_made_in_game,
    SUM(shot_made_flag * shot_point_value) OVER (PARTITION BY game_id
                         ORDER BY period asc, minutes_remaining desc, seconds_remaining desc) - 
                         shot_made_flag * shot_point_value AS num_points_made_in_game, 
    
    game_date_features.game_is_back_to_back,
    game_date_features.num_game_last_7_days
    FROM data
    LEFT JOIN
    game_date_features
    ON date(data.game_date) = game_date_features.game_date;
    """


def sql_feature_eng(data: pd.DataFrame) -> pd.DataFrame:
    """
    Notes
    """
    df = sqldf(FEATURES_QUERY, locals())
    return df


def main(input_filepath: str, output_filepath: str):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    Gets some features that's only possible from looking at the full data
    Eg. Is the Game a Back to Back Game?
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Read Data
    data = pd.read_csv(input_filepath, parse_dates=["game_date"])

    # Data Cleansing
    data = data[data["shot_made_flag"].notna()]

    # Data Features
    data["shot_point_value"] = np.where(
        data["shot_zone_basic"] == "Above the Break 3",
        3,
        np.where(
            data["shot_zone_basic"] == "Left Corner 3",
            3,
            np.where(data["shot_zone_basic"] == "Right Corner 3", 3, 2),
        ),
    )
    data["is_two_point"] = np.where(data["shot_point_value"] == 2, True, False)
    data["is_three_point"] = np.where(data["shot_point_value"] == 3, True, False)
    data["is_shot"] = 1

    data = sql_feature_eng(data)

    # Split Data
    X_data, Y_data = data.drop("shot_made_flag", axis=1), data["shot_made_flag"]
    x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2)

    # Write Data Tables
    x_train.to_csv(output_filepath + "/train.csv", index=False)
    y_train.to_csv(output_filepath + "/train_target.csv", index=False)
    x_val.to_csv(output_filepath + "/test.csv", index=False)
    y_val.to_csv(output_filepath + "/test_target.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(RAW_DATA_PATH, PROCESSED_PATH)
