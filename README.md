# 🚀 Software Testing Project

## 🛠 Setting up a Virtual Environment

To run the scripts in this project, follow these steps:

1. Ensure you have **Python 3.11** installed. If not, download and install it from [here](https://www.python.org/downloads/release/python-3110/).
2. Install the `pipenv` package using:

   ```bash
   pip install pipenv
   ```

3. Activate the virtual environment:

   ```bash
   pipenv shell
   ```

   If this doesn't work because your system can't locate the Python installation, use:

   ```bash
   pipenv --python '/path/to/your/python' shell
   ```

4. Install dependencies:

   ```bash
   pipenv install
   ```

Now, you should have an active virtual environment in your terminal, which includes all necessary dependencies to run the scripts.

---

## 📂 Folder Structure

Here's an overview of the project structure:

- **📁 `data/`**  
  Contains various CSV files used in the project.

- **📊 `dataset_analysis/`**  
  Includes Jupyter notebooks (`.ipynb`) for dataset visualizations and analysis.

- **📄 `docs/`**  
  Holds reports and articles related to the problem statement.

- **📌 `external_models/`**  
  - Contains Jupyter notebooks for **neighborhood stability tests** and **hill-climbing tests** and all previous tests to test for other teams models.  
  - The subfolder `received_folder/` contains models received from another team.

- **🧪 `model_1/`**  
  - Contains different testing approaches for Model 1.

- **🔬 `model_2/`**  
  - Contains different testing approaches for Model 2.

- **📝 `onnx-example-main/`**  
  - Includes sample code provided by the instructors.

- **🛠 `testing/`**  
  - Contains various test scripts run on the models.

- **⚙️ `utils/`**  
  - Includes utility functions used across the project.

---

## 🔍 Main Files to Look At

- **Model 1**
  - 🏗 `model_1/modelxgb.ipynb` → Core model file for Model 1.

- **Model 2**
  - 🏗 `model_2/model_2.py` → Core script for Model 2.

- **Hill Climbing**
  - ⛰ `external_models/hill_climbing.ipynb` → Notebook for hill climbing tests.

- **Neighborhood Stability Tests**
  - 🏘 `external_models/model_1.ipynb` → Stability analysis and old tests for Model 1.
  - 🏘 `external_models/model_2.ipynb` → Stability analysis and old tests for Model 2.

- **Bias and Other Model Tests**
  - ⚖️ `testing/bias_metrics.ipynb` → Contains bias analysis.
  - Additional test notebooks in `model_1/` cover various other test cases.

---

This readme provides an overview of how to set up, navigate, and utilize key components of our project.
