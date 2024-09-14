import pandas as pd

if __name__ == '__main__':
    preprocessed_data_file_id_df = pd.read_csv('PreprocessedData.csv')
    original_metadata = pd.read_csv('Labels.csv')

    preprocessed_data_metadata = original_metadata.loc[
        original_metadata['FILE_ID'].isin(preprocessed_data_file_id_df['file name'].values)]

    preprocessed_data_metadata.to_csv('preprocessed_data_metadata.csv', index=False)
