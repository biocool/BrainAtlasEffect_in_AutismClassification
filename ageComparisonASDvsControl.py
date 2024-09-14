import pandas as pd
from scipy.stats import ttest_ind

if __name__ == '__main__':

    preprocessed_data_metadata = pd.read_csv('preprocessed_data_metadata.csv')

    asd_samples_df = preprocessed_data_metadata.loc[preprocessed_data_metadata['Group'] == 'Autism']
    control_samples_df = preprocessed_data_metadata.loc[preprocessed_data_metadata['Group'] == 'Control']

    # Extract the 'Age' column from both dataframes
    asd_samples_age = asd_samples_df['Age']
    control_samples_age = control_samples_df['Age']

    # Perform an independent two-sample t-test
    t_stat, p_value = ttest_ind(asd_samples_age, control_samples_age, equal_var=False)  # Assuming unequal variances

    # Print the result
    # T-Statistic: 1.6138, P-Value: 0.1070
    print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

    # Interpret the result
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("There is a significant difference between the ages of the two groups (ASD and Control).")
    else:
        print("There is no significant difference between the ages of the two groups (ASD and Control).")
