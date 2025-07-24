
# 🏡 Multimodal House Price Prediction using Images + Tabular Data

This project predicts house prices using a **multimodal deep learning approach** that combines both **tabular features** (like number of beds, baths, size) and **images of houses** using TensorFlow and EfficientNet.

---

## 📁 Dataset

The dataset includes:

- CSV file: `filtered_housing_data1.csv`
- Image folders:
  - `socal_pics/train/house/`
  - `socal_pics/val/house/`

### Features used:
- `n_citi`: Numeric city encoding
- `bed`: Number of bedrooms
- `bath`: Number of bathrooms
- `sqft`: Square footage
- `image_id`: For locating the house image
- `price`: Target (actual house price)

---

## 🧠 Model Architecture

### 🔹 Image Branch:
- Base: `EfficientNetB0` (frozen)
- Global Average Pooling
- Dense layers + Dropout

### 🔸 Tabular Branch:
- Dense → BatchNorm → Dropout layers

### 🔀 Merged:
- Concatenated features
- Dense layers
- Final output: House price (log-scaled)

---

## 🔧 Training

- Loss: **Huber**
- Optimizer: **Adam (1e-4)**
- EarlyStopping and ReduceLROnPlateau used
- Trained for 30 epochs with batch size = 16
- Best model saved as: `best_multimodal_model.keras`

---

## 📈 Evaluation

Evaluation is done on the **validation set** with metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

```
✅ MAE: ~204,000
✅ RMSE: ~314,000
```

If RMSE < 350,000, the final model is saved as `multimodal_housing_model.keras`.

---

## 📊 Visualization

A scatter plot is generated comparing **Actual vs Predicted House Prices**.

![Predicted vs Actual](your_plot.png)

---

## 🧪 Single Prediction

To perform a prediction on a single sample:
1. Load the image using `load_img` and preprocess it.
2. Normalize the tabular features using the saved scaler.
3. Call `model.predict()` with both inputs.
4. Apply `inverse_transform` and `expm1` to get the final price.

---

## 🛠️ Dependencies

- Python ≥ 3.7
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

---

## 📂 Folder Structure

```
.
├── images+tabular house dataset/
│   ├── filtered_housing_data1.csv
│   ├── socal_pics/
│   │   ├── train/house/
│   │   └── val/house/
├── target_scaler.pkl
├── best_multimodal_model.keras
├── multimodal_housing_model.keras
├── your_script.py
└── README.md
```

---

## ✨ Future Work

- Fine-tune EfficientNetB0
- Add image augmentation
- Experiment with regression losses
- Add geolocation or date features

---

## 📜 License

This project is for educational and research purposes only.

---

## 🙋‍♂️ Author

Built with ❤️ by [M.Haris Kamran]
