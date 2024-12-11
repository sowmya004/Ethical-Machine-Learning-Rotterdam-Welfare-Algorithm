import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn

data = pd.read_csv('../data/data/investigation_train_large_checked.csv')

y = data['checked']
X = data.drop(['checked'], axis=1)
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

classifier = XGBClassifier(max_depth=25, n_estimators=200)


pipeline = Pipeline([
    #Redacted step that does not affect the number of columns 
    ('scaling', StandardScaler()),                
    ('classification', classifier)               
])


pipeline.fit(X_train, y_train)

update_registered_converter(
    XGBClassifier, 
    "XGBoostXGBClassifier",  
    calculate_linear_classifier_output_shapes,  
    convert_xgboost, 
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)

onnx_model = convert_sklearn(
    pipeline, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
    target_opset={'': 12, 'ai.onnx.ml': 3})

onnx.save(onnx_model, "model1.onnx")

testing_session = rt.InferenceSession("model1.onnx")

#Redacted tests done 