import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Constants
DATASET_LOCATION = 'labeled.csv'

TEST_SPLIT = 0.2

EPOCHS = 50
BATCH_SIZE = 1
LR = .001
VALIDATION_SPLIT = 0.2

PATIENCE = 5

THRESHOLD_SCALAR = 0.7

# MAD score
def mad_score(points):
  m = np.median(points)
  ad = np.abs(points - m)
  mad = np.median(ad)

  return 0.6745 * ad / mad

# Main function
def detect_anomalies(location):
  # Load dataset
  dataset_df = pd.read_csv(DATASET_LOCATION)

  # Prepare data
  X_location = dataset_df[dataset_df['type'] == location]\
    .drop('type', 1)

  X_normal = X_location[X_location['isAnomaly'] == 0]
  X_anomalies = X_location[X_location['isAnomaly'] == 1]

  X_train_df, X_test_df = train_test_split(X_normal, test_size=TEST_SPLIT)

  X_train = X_train_df.drop('isAnomaly', axis=1).values

  X_test_df = pd.concat([X_test_df, X_anomalies])
  X_test, y_test = X_test_df.drop('isAnomaly', axis=1).values, X_test_df.isAnomaly.values

  # Normalize data
  X_train = keras.utils.normalize(X_train)
  X_test = keras.utils.normalize(X_test)

  # Create model
  input_dim = X_train.shape[1]

  model = keras.Sequential([
    layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
    layers.Dense(2, activation='elu'),
    layers.Dense(2, activation='elu'),
    layers.Dense(input_dim, activation='elu')
  ])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss='mse')

  # Train model
  model.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[
      keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        restore_best_weights=True
      )
    ]
  )

  # Calculate threshold
  X_train_pred = model.predict(X_train)

  train_mse = np.mean(np.power(X_train_pred - X_train, 2), axis=1)
  train_mad_score = mad_score(train_mse)

  threshold = np.max(train_mad_score) * THRESHOLD_SCALAR

  # Test model
  X_test_pred = model.predict(X_test)

  # Detect anomalies
  test_mse = np.mean(np.power(X_test_pred - X_test, 2), axis=1)
  test_mad_score = mad_score(test_mse)

  outliers = test_mad_score > threshold

  anomalies = np.where(outliers)
  detected = X_test_df.iloc[anomalies]

  # Calculate metrics
  conf_mat = confusion_matrix(y_test, outliers)
  tn, fp, fn, tp = conf_mat.flatten()

  precision = tp / (fp + tp) * 100 if fp + tp != 0 else 0
  recall = tp / (fn + tp) * 100 if fn + tp != 0 else 0

  # Print summary
  print()
  print('#################################  SUMMARY  #################################')
  
  print(f'Original data example: {X_train[0]}')
  print(f'Learned reconstruction example: {X_train_pred[0]}')
  print()

  print('Detected rows')
  print(detected)
  print()

  print('Confusion matrix')
  print(conf_mat)
  print()

  print(f'Model precision: {precision:.2f}%')
  print(f'Model recall: {recall:.2f}%')
  print('#############################################################################')

  # Save new dataset to csv
  dataset_removed_anomalies = X_location.drop(detected.index)
  dataset_removed_anomalies = dataset_removed_anomalies.drop('isAnomaly', axis=1)

  dataset_removed_anomalies.to_csv(f'AE_{location}_no_anomalies.csv', index=False)
  
  print()
  print('Saved cleaned dataset')
  print()

  # Save trained model
  model.save(f'AE_{location}_trained_model')
  
  print()
  print('Saved trained model')
  print()

detect_anomalies('cloud')
detect_anomalies('local')
