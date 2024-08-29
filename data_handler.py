import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pandas as pd


class customDict():
    def __init__(self):
        self.counter = 2
        self.add('pad', 0)
        self.add('sep', 1)

    def add(self, key, value):
        self.__dict__[key] = value

    def get_add(self, key):
        if key not in self.__dict__:
            self.add(key, self.counter)
            self.counter += 1
        return self.__dict__[key]

    def __len__(self):
        return self.counter
    

class SelexDataset(Dataset):
    def __init__(self, size, df):
        self.size = size
        self.df = df

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        df_sample = self.df.iloc[idx]
        sequence = torch.tensor(df_sample['sequence'], dtype=torch.long)
        label = torch.tensor(df_sample['label'], dtype=torch.float)
        return sequence, label
    

class CompeteDataset(Dataset):
        def __init__(self, size, kmers_tokenizer, intensities_df):
            self.size = size
            self.kmers_tokenizer = kmers_tokenizer
            self.intensities_df = intensities_df

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            df_sample = self.intensities_df.iloc[idx]
            sequence = torch.tensor(self.kmers_tokenizer.encode(df_sample['sequence'], 50), dtype=torch.long)
            return sequence
        

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_compete_loader(intensities_df, kmers_tokenizer):
    g = torch.Generator()
    g.manual_seed(0)

    compete_dataset = CompeteDataset(len(intensities_df), kmers_tokenizer, intensities_df)

    batch_size = 256
    intensities_df_dataloader = DataLoader(compete_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=os.cpu_count(),
                                  pin_memory=False,
                                  worker_init_fn=seed_worker,
                                  generator=g)
    return intensities_df_dataloader


# with open(f"{root}/RNAcompete_intensities/{compete_selex_train[rbp_idx][0]}") as f:
#     file_contents = f.read()
#     intensities = file_contents.split("\n")
#     intensities = [float(inten) for inten in intensities if inten != ""]

def get_compete_sequences_df(path):
    with open(path) as f:
            file_contents = f.read()
            RNAcompete_sequences = file_contents.split("\n")
            RNAcompete_sequences = RNAcompete_sequences[:-1]

    data = {'sequence': RNAcompete_sequences}

    # Create the DataFrame using the dictionary
    intensities_df = pd.DataFrame(data)
    return intensities_df

# compete_selex_train_RBP = compete_selex_train[rbp_idx]


def get_dfs(compete_selex_train_RBP, kmers_tokenizer):
    htr_selex_cycles = []
    for i in range(len(compete_selex_train_RBP)):
        with open(compete_selex_train_RBP[i]) as f:
            file_contents = f.read()
            htr_selex = file_contents.split("\n")
            htr_selex = htr_selex[:-1]
            htr_selex_cycles.append(np.array(htr_selex))


    data_original = [
        [seq, int(count), i] 
        for i, cycle in enumerate(htr_selex_cycles) 
        for seq, count in (item.split(',') for item in cycle)
    ]

    df_original = pd.DataFrame(data_original, columns=['sequence', 'count', 'label'])


    df = df_original.copy()
    df['sequence'] = df['sequence'].apply(lambda seq: kmers_tokenizer.encode(seq, 50))

    # Convert lists in 'sequence' to tuples
    df['sequence'] = df['sequence'].apply(tuple)

    # Now group by 'sequence' and find the maximum 'label' for each group
    df = df.groupby('sequence')['label'].max().reset_index()

    return df, df_original






def get_train_loader(df):
    g = torch.Generator()
    g.manual_seed(0)

    train_dataset = SelexDataset(len(df), df)

    batch_size = 256
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=os.cpu_count(),
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g)
    return train_dataloader