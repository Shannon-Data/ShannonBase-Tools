import joblib
import pandas as pd
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import save_model
import lightgbm as lgb

# 1. load LightGBM model
model = joblib.load("rapid_offload_classifier_model.pkl")

# 2. read the data to get the num of features.
df = pd.read_csv("mysql_offload_balanced_5000_IS_OLAP.csv")
X = df.drop(columns=["IS_OLAP"])

n_features = X.shape[1]
print(f"number of features: {n_features}")

# 3. using LightGBM Booster
if hasattr(model, 'booster_'):
    booster = model.booster_
else:
    booster = model

# 4. build up  ONNX format input.
initial_types = [("input", FloatTensorType([None, n_features]))]

# 5. using LightGBM official converter
try:
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types
    )
    
    # 6. save the model in ONNX file
    save_model(onnx_model, "rapid_offload_classifier.onnx")
    
    print("✅ conversion succeed：rapid_offload_classifier.onnx")
    
except Exception as e:
    print(f"❌ coversion failed: {e}")
    import traceback
    traceback.print_exc()
