import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('data/investigation_train_large_checked.csv')

# Function to visualize multiple columns with checked, not checked, and total counts
def visualize_columns_with_totals(df, columns, checked_col, binary=False):
    data = []
    labels = []
    
    for col in columns:
        if binary:
            # Include both True and False counts
            total_true = df[df[col] == True].shape[0]
            total_false = df[df[col] == False].shape[0]
            checked_true = df[(df[col] == True) & (df[checked_col] == True)].shape[0]
            not_checked_true = total_true - checked_true
            data.append([checked_true, not_checked_true, total_true])
            labels.append(f'{col} (True)')
            
            checked_false = df[(df[col] == False) & (df[checked_col] == True)].shape[0]
            not_checked_false = total_false - checked_false
            data.append([checked_false, not_checked_false, total_false])
            labels.append(f'{col} (False)')
        else:
            # Only include True counts
            total_true = df[df[col] == True].shape[0]
            checked_true = df[(df[col] == True) & (df[checked_col] == True)].shape[0]
            not_checked_true = total_true - checked_true
            data.append([checked_true, not_checked_true, total_true])
            labels.append(col)

    # Convert data to NumPy array for easier plotting
    data = np.array(data)

    # Create the bar plot
    x = np.arange(len(labels))  # Label locations
    width = 0.25  # Bar width

    plt.figure(figsize=(15, 7))
    plt.bar(x - width, data[:, 0], width, label='Checked', color='orange')
    plt.bar(x, data[:, 1], width, label='Not Checked', color='skyblue')
    plt.bar(x + width, data[:, 2], width, label='Total', color='green')

    # Add labels and legend
    plt.xlabel('Attributes')
    plt.ylabel('Counts')
    plt.title('Checked, Not Checked, and Total Counts per Attribute')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Call the function with desired columns and the `checked` column
visualize_columns_with_totals(df, ['adres_recentste_wijk_charlois', 'adres_recentste_wijk_delfshaven', 'adres_recentste_wijk_feijenoord',  'adres_recentste_wijk_ijsselmonde', 'adres_recentste_wijk_kralingen_c', 'adres_recentste_wijk_noord', 'adres_recentste_wijk_prins_alexa', 'adres_recentste_wijk_stadscentru', 'adres_recentste_wijk_noord', 'adres_recentste_wijk_other'], 'checked', binary=False)



# Call the function with desired columns and the `checked` column
#visualize_with_checked(df, ['persoon_geslacht_vrouw'], 'checked', binary=False)  # Only True values
#visualize_with_checked(df, ['persoon_geslacht_vrouw'], 'checked', binary=True)   # Include False and True
#visualize_with_checked(df, ['adres_recentste_wijk_noord', 'adres_recentste_wijk_other'], 'checked', binary=False)  # Only True values
