import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

# Load dataset
df = pd.read_csv('data/investigation_train_large_checked.csv')

def calculate_disparate_impact(df, subset, outcome_col):
    # Create the privileged group column
    df['privileged_group'] = (~df.index.isin(subset.index)).astype(int)
    
    # Map the outcome column such that favorable outcome is 1
    df['outcome_mapped'] = (~df[outcome_col]).astype(int)  # `False` becomes 1 (favorable)
    
    # Create an AIF360 BinaryLabelDataset
    aif360_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=['outcome_mapped'],
        protected_attribute_names=['privileged_group']
    )
    
    # Calculate metrics
    metrics = BinaryLabelDatasetMetric(aif360_dataset, privileged_groups=[{'privileged_group': 1}], unprivileged_groups=[{'privileged_group': 0}])
    
    # Return the disparate impact
    return metrics.disparate_impact()

# Helper function to evaluate predicates
def evaluate_predicate(df, predicate):
    return df.loc[predicate(df)], df.loc[~predicate(df)]

# Visualization function for predicates
def visualize_predicates(df, predicates, checked_col, labels=None):
    data = []
    predicate_labels = []
    
    for i, predicate in enumerate(predicates):
        # Apply the predicate
        subset,_ = evaluate_predicate(df, predicate)
        
        # Calculate fairness using AIF360
        di = calculate_disparate_impact(df, subset, checked_col)
        print(f"DI for {labels[i]}: {di}")
        
        # Prepare data for visualization
        total_count = subset.shape[0]
        checked_count = subset[subset[checked_col] == True].shape[0]
        not_checked_count = total_count - checked_count
        
        data.append([checked_count, not_checked_count, total_count])
        predicate_labels.append(labels[i] if labels else f'Predicate {i+1}')
    
    # Convert data to NumPy array for easier plotting
    data = np.array(data)
    
    # Create the bar plot
    x = np.arange(len(predicate_labels))  # Label locations
    width = 0.25  # Bar width
    
    plt.figure(figsize=(15, 7))
    plt.bar(x - width, data[:, 0], width, label='Checked', color='orange')
    plt.bar(x, data[:, 1], width, label='Not Checked', color='skyblue')
    plt.bar(x + width, data[:, 2], width, label='Total', color='green')
    
    # Add labels and legend
    plt.xlabel('Predicates')
    plt.ylabel('Counts')
    plt.title('Checked, Not Checked, and Total Counts per Predicate')
    plt.xticks(x, predicate_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Define reusable predicate functions
def is_age_between(min_age, max_age):
    return lambda df: (df['persoon_leeftijd_bij_onderzoek'] >= min_age) & (df['persoon_leeftijd_bij_onderzoek'] < max_age)

def is_woman():
    return lambda df: df['persoon_geslacht_vrouw'] == True

def is_not_woman():
    return lambda df: df['persoon_geslacht_vrouw'] == False

def has_child():
    return lambda df: df['relatie_kind_heeft_kinderen'] == True

# Logical combinations of predicates
def combine_and(*predicates):
    return lambda df: np.logical_and.reduce([predicate(df) for predicate in predicates])

# Example usage
predicates = [
    is_woman()
]

labels = [
    "womannn"
]

# Visualize predicates
visualize_predicates(df, predicates, 'checked', labels)
