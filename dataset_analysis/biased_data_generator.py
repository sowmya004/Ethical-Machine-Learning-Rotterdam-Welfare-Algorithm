import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


df = pd.read_csv('../data/investigation_train_large_checked.csv')


# all people who have met language requirement 
lang_met = df[df['persoonlijke_eigenschappen_taaleis_voldaan'] == 1]

# people who have not met met language requirement and checked for suspicion 
lang_not_met_checked_true = df[(df['persoonlijke_eigenschappen_taaleis_voldaan'] == 0) & (df['checked'] == 1)]

# people who have not met language requirement and not checked for suspicion 
lang_not_met_checked_false = df[(df['persoonlijke_eigenschappen_taaleis_voldaan'] ==0) & (df['checked'] == 0)]

# Sample 10% of not met language requirement with 'checked = 0'
lang_not_met_checked_false_sampled = lang_not_met_checked_false.sample(frac=0.1, random_state=42)  # random_state ensures reproducibility


biased_df = pd.concat([lang_met, lang_not_met_checked_true, lang_not_met_checked_false_sampled])


biased_df.reset_index(drop=True, inplace=True)


# Save to a new file 
biased_df.to_csv('../data/biased_dataset.csv', index=False)
