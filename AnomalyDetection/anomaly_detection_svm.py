import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import where


def get_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_values = df.values
    standard_scaler = preprocessing.StandardScaler()
    df_values_scaled = standard_scaler.fit_transform(df_values)
    return pd.DataFrame(df_values_scaled)


def detect_anomalies_and_save_to_csv(computing_type: str, check: bool = False, normalization: bool = False) -> None:
  data = pd.read_csv('xd.csv')

  df = data.loc[data['type'] == computing_type]
  df = df.drop('type', axis='columns')

  if normalization:
    df_normalized = get_normalized_dataframe(df)

    model = OneClassSVM(kernel = 'rbf', gamma = 5e-2, nu = 0.05).fit(df_normalized)
    y_pred = model.predict(df_normalized)
  
  else:
    model = OneClassSVM(kernel = 'rbf', gamma = 1e-9, nu = 0.05).fit(df)
    y_pred = model.predict(df)

  outlier_index_list = where(y_pred == -1)
  outlier_df = df.iloc[outlier_index_list]

  normal_guys_index_list = where(y_pred == 1)
  normal_guys_df = df.iloc[normal_guys_index_list]

  if check:
    df_battery_converted = df[' battery'] * -1
    df_battery_converted_outlier = outlier_df[' battery'] * -1

    plt.scatter(df[' time'], df_battery_converted)
    plt.scatter(outlier_df[' time'], df_battery_converted_outlier, c = 'r')
    plt.xlabel('time [ms]', fontsize=12)
    plt.ylabel('battery usage [mAh]', fontsize=12)
    plt.show()

  normal_guys_df.to_csv(f'SVM_{computing_type}_no_anomalies.csv', index=False)


if __name__ == '__main__':
  detect_anomalies_and_save_to_csv('cloud', check=True, normalization=True)
  detect_anomalies_and_save_to_csv('local', check=True, normalization=True)