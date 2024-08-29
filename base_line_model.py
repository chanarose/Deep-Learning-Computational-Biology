from collections import defaultdict
import numpy as np

def base_line_predict(df_original, intensities_df, kmers_tokenizer):
    df_last = df_original[df_original['label'] == df_original['label'].max()]
    k_mer = 7

    # Create a dictionary to count all 7-mers in sequences with label 3
    window_counts = defaultdict(int)
    for sequence, count in zip(df_last['sequence'], df_last['count']):
        for i in range(len(sequence) - k_mer + 1):  # 7-mer length
            window = sequence[i:i+k_mer].upper()  # Ensure uppercase for consistency
            window_counts[window] += count

    # Iterate over each row in 'intensities_df', calculate max window count
    windows_count = []
    for idx, row in intensities_df.iterrows():
        print(f'\r{idx + 1}/{len(intensities_df)}', end='')
        seq = row['sequence']
        windows = list(kmers_tokenizer.Kmers_funct(seq, k_mer))  # Assume this returns a list of 7-mers
        current_counts = [window_counts[window.upper()] for window in windows]  # Use upper to match the case used during counting
        max_val = np.mean(current_counts) if current_counts else 0  # Handle case where there are no counts
        windows_count.append(max_val)

    print('\nCompleted.')
    windows_count = np.array(windows_count)

    return windows_count