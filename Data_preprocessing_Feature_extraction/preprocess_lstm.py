import pandas as pd


def get_binary_column(df, column, value1):
    return df[column].apply(lambda x: 1 if x == value1 else 0)


def preprocess_traffic_events_data(df):
    boolean_traffic_columns = ['Amenity', 'Bump', 'Crossing', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
                               'Stop',
                               'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

    processed_columns = pd.DataFrame()

    for column in boolean_traffic_columns:
        try:
            processed_columns[column] = df[column].astype(int)
        except Exception as ex:
            print(f"Exception while converting traffic features in {column}")
            print(ex)
            print("Rolling back changes")
            return df

    for column in boolean_traffic_columns:
        df[column] = processed_columns[column]
    return df
