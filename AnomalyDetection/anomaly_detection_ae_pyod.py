import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pyod.models.auto_encoder import AutoEncoder

# Constants
DATASET_LOCATION = 'labeled.csv'

TEST_SIZE = 0.2

EPOCHS = 50
BATCH_SIZE = 1

def detect_anomalies(location):
  # Load dataset
  dataset_df = pd.read_csv(DATASET_LOCATION)

  # Prepare data
  X_location = dataset_df[dataset_df['type'] == location]\
    .drop('type', 1)

  X_normal = X_location[X_location['isAnomaly'] == 0]
  X_anomalies = X_location[X_location['isAnomaly'] == 1]

  X_train_df, X_test_df = train_test_split(X_normal, test_size=TEST_SIZE)

  X_train = X_train_df.drop('isAnomaly', axis=1).values

  X_test_df = pd.concat([X_test_df, X_anomalies])
  X_test = X_test_df.drop('isAnomaly', axis=1).values

  # Normalize data
  X_train = keras.utils.normalize(X_train)
  X_test = keras.utils.normalize(X_test)

  # Create model
  input_dim = X_train.shape[1]

  model = AutoEncoder(
    hidden_neurons=[input_dim, 2, 2, input_dim],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
  )

  # Train model
  model.fit(X_train)

  # Calculate threshold
  X_train_pred = model.decision_scores_
  threshold = X_train_pred.max()

  # Test model
  X_test_pred = model.decision_function(X_test)

  # Detect anomalies
  anomalies = np.where(X_test_pred > threshold)
  detected = X_test_df.iloc[anomalies]

  # Calculate detection accuracy
  total_anomalies = X_anomalies.shape[0]
  total_detected = detected.shape[0]

  false_positives = detected[detected['isAnomaly'] == 0].shape[0]
  true_positives = detected[detected['isAnomaly'] == 1].shape[0]

  # Print summary
  print()
  print('################################################################')
  print()

  print('Detected rows:')
  print(detected)
  print()

  print(f'Total number of anomalies: {total_anomalies}')
  print(f'Total number of detected anomalies: {total_detected}')
  print(f'Total number of false positives: {false_positives}')
  print(f'Total number of true positives: {true_positives}')
  print()

  # Save new dataset to csv
  dataset_removed_anomalies = X_location.drop(detected.index)
  dataset_removed_anomalies = dataset_removed_anomalies.drop('isAnomaly', axis=1)

  dataset_removed_anomalies.to_csv(f'AE_{location}_no_anomalies.csv', index=False)

detect_anomalies('cloud')
detect_anomalies('local')
