import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('data/investigation_train_large_checked.csv')

def calculate_fairness_stats(privileged_total, privileged_checked, unprivileged_total, unprivileged_checked):
    unprivileged_unchecked = unprivileged_total - unprivileged_checked
    privileged_unchecked = privileged_total - privileged_checked

    favorable_outcome_given_unprivileged = unprivileged_unchecked / unprivileged_total
    favorable_outcome_given_privileged = privileged_unchecked / privileged_total

    spd = favorable_outcome_given_unprivileged - favorable_outcome_given_privileged
    di = favorable_outcome_given_unprivileged / favorable_outcome_given_privileged

    return {
        "Disparate Impact": di,
        "Statistical Parity Difference": spd
    }


# Helper function to evaluate predicates
def evaluate_predicate(df, predicate):
    return df.loc[predicate(df)], df.loc[~predicate(df)]

# Visualization function for predicates
def visualize_predicates(df, predicates, checked_col, labels=None):
    data = []
    predicate_labels = []
    
    for i, predicate in enumerate(predicates):
        # Apply the predicate
        subset, subset_complement = evaluate_predicate(df, predicate)
        
        total_count = subset.shape[0]
        checked_count = subset[subset[checked_col] == True].shape[0]
        not_checked_count = total_count - checked_count

        total_count_complement = subset_complement.shape[0]
        checked_count_complement = subset_complement[subset_complement[checked_col] == True].shape[0]

        fairness_stats = calculate_fairness_stats(total_count_complement, checked_count_complement, total_count, checked_count)
        print(f"Stats for: {labels[i]}")
        for stat in fairness_stats:
            print(f"{stat}: {fairness_stats[stat]}")
        
        data.append([checked_count, not_checked_count, total_count])
        # Use provided labels or default to predicate index
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
    plt.xlabel('Column Filtering Combinations')
    plt.ylabel('No. of People')
    plt.title('Checked, Not Checked, and Total Counts per Predicate')
    plt.xticks(x, predicate_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Define reusable predicate functions
def is_age_between(min_age, max_age):
    return lambda df: (df['persoon_leeftijd_bij_onderzoek'] >= min_age) & (df['persoon_leeftijd_bij_onderzoek'] < max_age)

def is_not_woman():
    return lambda df: df['persoon_geslacht_vrouw'] == False

def has_child():
    return lambda df: df['relatie_kind_heeft_kinderen'] == True

def has_no_language_skills():
    return lambda df: df['persoonlijke_eigenschappen_taaleis_voldaan'] == False

def has_roommate():
    return lambda df: df['relatie_overig_kostendeler'] == True

def financial_problems():
    return lambda df: df['belemmering_financiele_problemen'] == True

# Logical combinations of predicates
def combine_and(*predicates):
    return lambda df: np.logical_and.reduce([predicate(df) for predicate in predicates])


# Example usage
predicates = [
    is_age_between(20, 30),
    combine_and(is_not_woman(), is_age_between(20, 35), has_child()),
    combine_and(has_roommate(), financial_problems(), has_no_language_skills())
]

labels = [
    "Age Between 20-30",
    "Not Woman, \n Younger than 35, \n Has Child",
    "Has Roommate, \n Financial Problems, \n Lang. Requirement not Met"
]

# Visualize predicates
visualize_predicates(df, predicates, 'checked', labels)
