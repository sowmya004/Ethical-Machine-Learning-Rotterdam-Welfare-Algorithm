### TODO alter this script to train and export a model of our choice

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import utils
import pandas as pd
import numpy as np
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from sklearn.model_selection import train_test_split
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../data/investigation_train_large_checked.csv')

data = pd.read_csv(data_path)
data = utils.preprocess_data(data)


y = data['checked']
X = data.drop(['checked'], axis=1)
X = X.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

selector = VarianceThreshold()
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
pipeline = Pipeline(steps=[('feature selection', selector), ('classification', classifier)])
statistics = utils.test_model(pipeline,X,y,X_train, X_test, y_train, y_test )
print("model stats")
print(statistics)

#print(pipeline.named_steps['classification'].feature_importances_)
# Let's convert the model to ONNX
onnx_model = convert_sklearn(
    pipeline, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
    target_opset=12)

# Let's check the accuracy of the converted model
sess = rt.InferenceSession(onnx_model.SerializeToString())
statistics_onnx = utils.test_model(sess,X,y,X_train, X_test, y_train, y_test )
print("ONNX rep stats")
print(statistics_onnx)

# Let's save the model
onnx.save(onnx_model, "gboost.onnx")

# Let's load the model
new_session = rt.InferenceSession("gboost.onnx")

# Let's predict the target
y_pred_onnx2 =  new_session.run(None, {'X': X_test.values.astype(np.float32)})

accuracy_onnx_model = accuracy_score(y_test, y_pred_onnx2[0])
print('Accuracy of the loaded ONNX model: ', accuracy_onnx_model)


