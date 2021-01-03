import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import where


def get_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_values = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    df_values_scaled = min_max_scaler.fit_transform(df_values)
    return pd.DataFrame(df_values_scaled)


def detect_anomalies_and_save_to_csv(computing_type: str, check: bool = False, normalization: bool = False) -> None:
  data = pd.read_csv('xd.csv')

  df = data.loc[data['type'] == computing_type]
  df = df.drop('type', axis='columns')

  if normalization:
    df_normalized = get_normalized_dataframe(df)

    model = OneClassSVM(kernel = 'rbf', gamma = 1e-3, nu = 0.05).fit(df_normalized)
    y_pred = model.predict(df_normalized)
  
  else:
    model = OneClassSVM(kernel = 'rbf', gamma = 1e-9, nu = 0.05).fit(df)
    y_pred = model.predict(df)

  outlier_index_list = where(y_pred == -1)
  outlier_df = df.iloc[outlier_index_list]

  normal_guys_index_list = where(y_pred == 1)
  normal_guys_df = df.iloc[normal_guys_index_list]

  if check:
    plt.scatter(df[' time'], df[' battery'])
    plt.scatter(outlier_df[' time'], outlier_df[' battery'], c = 'r')
    plt.show()

  normal_guys_df.to_csv(f'SVM_{computing_type}_no_anomalies.csv', index=False)


if __name__ == '__main__':
  detect_anomalies_and_save_to_csv('cloud', check=True, normalization=True)
  detect_anomalies_and_save_to_csv('local', check=True, normalization=True)