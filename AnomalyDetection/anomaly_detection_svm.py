import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from numpy import where


def detect_anomalies_and_save_to_csv(computing_type: str, check: bool = False) -> None:
  data = pd.read_csv('xd.csv')

  df = data.loc[data['type'] == computing_type]
  df = df.drop('type', axis='columns')

  model = OneClassSVM(kernel = 'rbf', gamma = 0.000000001, nu = 0.05).fit(df)
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
  detect_anomalies_and_save_to_csv('cloud', check=True)
  detect_anomalies_and_save_to_csv('local', check=True)