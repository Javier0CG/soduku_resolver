import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

print("version de Tensorflow",tf.__version__)
print("version de Keras", keras.__version__)
print("version de Numpy", np.__version__)
print("version de Pandas", pd.__version__)
print("version de Matplotlib", matplotlib.__version__)



# units=1               numero de neuronas
# entradas              input_shape=[1]
# activation="linear"   activacion (linear, relu, sigmoid, softmax, etc)

red_neuronal_steels = tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation="relu", input_shape=[15]),
      tf.keras.layers.Dense(units=128, activation="relu"),
      tf.keras.layers.Dense(units=64, activation="relu"),
      tf.keras.layers.Dense(units=32, activation="relu"),
      tf.keras.layers.Dense(units=1)
                                          ])

print(red_neuronal_steels.summary())


#optimizers: Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax, Ftrl, Lion, LossScaleOptimizer, Nadam, Optimizer, RMSprop, SGD
#loss: BinaryCrossentropy, BinaryFocalCrossentropy, CTC, CategoricalCrossentropy, CategoricalFocalCrossentropy, CategoricalHinge, CosineSimilarity, Dice, Hinge, Huber, KLDivergence, LogCosh, Loss, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MeanSquaredLogarithmicError, Poisson, SparseCategoricalCrossentropy, SquaredHinge, Tversky
#metrics: Accuracy, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy, TopKCategoricalAccuracy, SparseTopKCategoricalAccuracy, Mean, Sum, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, CosineSimilarity, LogCoshError, KLDivergence, Poisson, Precision, Recall, AUC, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy

red_neuronal_steels.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss="mae",
    metrics=["mae"]
                            )

#datos
df = pd.read_csv("/content/steel_compositions_db.csv")
print(df.head())

datos_fe = df["Fe"]
datos_c = df["C"]
datos_cr = df["Cr"]
datos_ni = df["Ni"]
datos_mn = df["Mn"]
datos_s = df["S"]
datos_si = df["Si"]
datos_mo = df["Mo"]
datos_v = df["V"]
datos_w = df["W"]
datos_co = df["Co"]
datos_al = df["Al"]
datos_p = df["P"]
datos_cu = df["Cu"]
datos_ti = df["Ti"]
datos_ts = df["Tensile_Strength_MPa"]

print("datos cargados")

datos_entrada = np.stack((datos_fe, datos_c, datos_cr, datos_ni, datos_mn, datos_s, datos_si, datos_mo, datos_v, datos_w, datos_co, datos_al, datos_p, datos_cu, datos_ti))
print(datos_entrada.shape)
datos_entrada = datos_entrada.T
print(datos_entrada.shape)
print(datos_ts.shape)

print("datos de entrada listos "*20)

# datos_entrada = df.drop(columns=["Tensile_Strength_MPa"]).values
# datos_ts = df["Tensile_Strength_MPa"].values

# print(datos_entrada.shape)
# print(datos_ts.shape)

entrenamiento_modelo = red_neuronal_steels.fit(
    datos_entrada,
    datos_ts,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
                                              )

# perdida (loss)
plt.figure()
plt.plot(entrenamiento_modelo.history['loss'], label='Train Loss')
plt.plot(entrenamiento_modelo.history['val_loss'], label='Val Loss')
plt.xlabel('epocas')
plt.ylabel('mse')
plt.title('evolución del Loss')
plt.legend()
plt.show()


# MAE error absoluto medio
plt.figure()
plt.plot(entrenamiento_modelo.history['mae'], label='Train MAE')
plt.plot(entrenamiento_modelo.history['val_mae'], label='Val MAE')
plt.xlabel('epocas')
plt.ylabel('mae (MPa)')
plt.title('evolucion del error')
plt.legend()
plt.show()


X_test = np.array([[55.7,0.46,17.11,14.64,1.2,0.01,0.31,0.29,0.87,3.01,3.54,.02,0.05,2.5,0.21]], dtype=float)
predictions = red_neuronal_steels.predict(X_test)

print("Predicciones:")
for i, val in enumerate(X_test):
    print(f"x = {val[0]} -> y_pred = {predictions[i][0]}")