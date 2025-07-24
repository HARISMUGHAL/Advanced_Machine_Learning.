import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === Paths ===
csv_path = "images+tabular house dataset/filtered_housing_data1.csv"
train_folder = "images+tabular house dataset/socal_pics/train/house"
val_folder = "images+tabular house dataset/socal_pics/val/house"

# === Load & Preprocess CSV ===
df = pd.read_csv(csv_path)

def get_image_path(image_id):
    path = os.path.join(train_folder, f"{image_id}.jpg")
    if not os.path.exists(path):
        path = os.path.join(val_folder, f"{image_id}.jpg")
    return path if os.path.exists(path) else None

df["image_path"] = df["image_id"].apply(get_image_path)
df = df[df["image_path"].notnull()].copy()

# === Features & Targets ===
tabular_features = ["n_citi", "bed", "bath", "sqft"]
df["log_price"] = np.log1p(df["price"])
target_col = "log_price"

# === Scale Tabular ===
scaler_X = StandardScaler()
X_tabular = scaler_X.fit_transform(df[tabular_features])

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(df[target_col].values.reshape(-1, 1))
joblib.dump(scaler_y, "target_scaler.pkl")

# === Preprocess Images ===
X_images = np.array([
    preprocess_input(img_to_array(load_img(path, target_size=(128, 128))))
    for path in df["image_path"]
])

# === Train-Test Split ===
X_tab_train, X_tab_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
    X_tabular, X_images, y_scaled, test_size=0.2, random_state=42
)

# === Image Branch ===
base_model = EfficientNetB0(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
base_model.trainable = False

image_input = Input(shape=(128, 128, 3), name="image_input")
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
img_features = x

# === Tabular Branch ===
tabular_input = Input(shape=(X_tabular.shape[1],), name="tabular_input")
y_ = Dense(128, activation='relu')(tabular_input)
y_ = BatchNormalization()(y_)
y_ = Dropout(0.3)(y_)
y_ = Dense(64, activation='relu')(y_)
tab_features = y_

# === Merge Branches ===
combined = Concatenate()([img_features, tab_features])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.3)(z)
output = Dense(1)(z)

# === Compile Model ===
model = Model(inputs=[image_input, tabular_input], outputs=output)
model.compile(optimizer=Adam(1e-4), loss=Huber(), metrics=["mae"])

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ModelCheckpoint("best_multimodal_model.keras", monitor='val_loss', save_best_only=True, verbose=1)
]

# === Train Model ===
history = model.fit(
    [X_img_train, X_tab_train], y_train,
    validation_data=([X_img_val, X_tab_val], y_val),
    epochs=30,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# === Evaluation ===
model = load_model("best_multimodal_model.keras")
scaler_y = joblib.load("target_scaler.pkl")

y_pred_scaled = model.predict([X_img_val, X_tab_val])
y_true_scaled = y_val

y_pred_actual = np.expm1(scaler_y.inverse_transform(y_pred_scaled))
y_true_actual = np.expm1(scaler_y.inverse_transform(y_true_scaled))

mae = mean_absolute_error(y_true_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))

print(f"\n✅ MAE: {mae:.2f}")
print(f"✅ RMSE: {rmse:.2f}")

# === Save Final Model if Good Enough ===
if rmse < 350000:
    model.save("multimodal_housing_model.keras")
    print("✅ Final model saved as 'multimodal_housing_model.keras'")
else:
    print("⚠️ Model not saved — performance needs improvement.")

# === Visualization ===
plt.figure(figsize=(8, 6))
plt.scatter(y_true_actual, y_pred_actual, alpha=0.5, color='dodgerblue')
plt.plot([y_true_actual.min(), y_true_actual.max()],
         [y_true_actual.min(), y_true_actual.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
