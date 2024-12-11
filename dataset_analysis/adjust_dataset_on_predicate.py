import pandas as pd
import numpy as np
import math

# Load dataset
df = pd.read_csv('../data/investigation_train_large_checked.csv')

# Helper function to evaluate predicates
def evaluate_predicate(df, predicate):
    return df.loc[predicate(df)], df.loc[~predicate(df)]

def calculate_di(privileged_total, privileged_checked, unprivileged_total, unprivileged_checked):
    unprivileged_unchecked = unprivileged_total - unprivileged_checked
    privileged_unchecked = privileged_total - privileged_checked

    favorable_outcome_given_unprivileged = unprivileged_unchecked / unprivileged_total
    favorable_outcome_given_privileged = privileged_unchecked / privileged_total

    di = favorable_outcome_given_unprivileged / favorable_outcome_given_privileged
    return di

# Logical combinations of predicates
def combine_and(*predicates):
    return lambda df: np.logical_and.reduce([predicate(df) for predicate in predicates])

def has_roommate():
    return lambda df: df['relatie_overig_kostendeler'] == True

def financial_problems():
    return lambda df: df['belemmering_financiele_problemen'] == True

def has_no_language_skills():
    return lambda df: df['persoonlijke_eigenschappen_taaleis_voldaan'] == False

def is_age_between(min_age, max_age):
    return lambda df: (df['persoon_leeftijd_bij_onderzoek'] >= min_age) & (df['persoon_leeftijd_bij_onderzoek'] < max_age)

def is_parent():
    return lambda df: df['relatie_kind_heeft_kinderen'] == True

def has_child():
    return lambda df: df['relatie_kind_heeft_kinderen'] == True

def is_not_woman():
    return lambda df: df['persoon_geslacht_vrouw'] == False


# Function to adjust the dataset
def adjust_dataset(df, predicate, target_di, checked_col):
    # Calculate original counts for each predicate
    subset, subset_complement = evaluate_predicate(df, predicate)

    total_count = subset.shape[0]
    checked_count = subset[subset[checked_col] == True].shape[0]
    not_checked_count = total_count - checked_count

    total_count_complement = subset_complement.shape[0]
    checked_count_complement = subset_complement[subset_complement[checked_col] == True].shape[0]

    di = calculate_di(total_count_complement, checked_count_complement, total_count, checked_count)
    print(f"di: {di}")
    
    a = target_di / di
    print(f"a: {a}")

    # If we're discriminating in a certain direction, we gotta remove either unchecked or checked
    if a > 1:
        # remove checked
        new_unprivileged_total = (1/a) * total_count
        amount_to_remove = total_count - new_unprivileged_total
        new_checked_unprivileged_amount = math.floor(checked_count - amount_to_remove)

        checked_unprivileged = subset[subset[checked_col] == True]
        new_checked_unprivileged = checked_unprivileged.sample(n=new_checked_unprivileged_amount)
        
        ## copy unchecked
        new_unchecked_unprivileged = subset[subset[checked_col] == False]
    else:
        # remove unchecked
        print(f"unchecked before: {not_checked_count}")
        new_unprivileged_total = a * total_count
        amount_to_remove = total_count - new_unprivileged_total
        new_unchecked_unprivileged_amount = math.floor(not_checked_count - amount_to_remove)
        print(f"unchecked after: {new_unchecked_unprivileged_amount}")

        unchecked_unprivileged = subset[subset[checked_col] == False]
        new_unchecked_unprivileged = unchecked_unprivileged.sample(n=new_unchecked_unprivileged_amount)
        
        ## copy checked
        new_checked_unprivileged = subset[subset[checked_col] == True]


    # Combine adjusted subsets
    adjusted_df = pd.concat([new_checked_unprivileged, new_unchecked_unprivileged, subset_complement], ignore_index=True)
    return adjusted_df

# Define predicates
predicates = combine_and(has_roommate(), financial_problems(), has_no_language_skills())

# Target DI for age 20-30
target_di = 0.8

# Adjust dataset
adjusted_df = adjust_dataset(df, predicates, target_di, 'checked')

# Save the adjusted dataset
adjusted_df.to_csv('../data/investigation_train_large_checked_adjusted.csv', index=False)

print("Adjusted dataset saved to 'data/investigation_train_large_checked_adjusted.csv'.")
