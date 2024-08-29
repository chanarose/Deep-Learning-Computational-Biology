import sys
import data_handler
import model
import base_line_model
import tokenizer
import numpy as np
import torch
from torch import nn



def main():
    args = sys.argv
    compete_path = args[1]
    print(compete_path)
    selex_paths = args[2:]
    print(selex_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    kmers_tokenizer = tokenizer.kmersTokenizer()

    compete_sequences_df = data_handler.get_compete_sequences_df(compete_path)
    df, df_original = data_handler.get_dfs(selex_paths, kmers_tokenizer)

    if df['label'].max() == 0:
        print("Only one label available, using base line model")
        base_line_results = base_line_model.base_line_predict(df_original, compete_sequences_df, kmers_tokenizer)
        np.savetxt(f'output.txt', base_line_results, fmt='%s')
        print("Complete! results at: output.txt")
        return

    compete_sequences_loader = data_handler.get_compete_loader(compete_sequences_df, kmers_tokenizer)
    train_loader = data_handler.get_train_loader(df)

    reg_model = model.Model(len(kmers_tokenizer))
    reg_model = nn.DataParallel(reg_model, device_ids=[0])
    reg_model.to(device)
    print(sum(p.numel() for p in reg_model.parameters())/1e6, 'M parameters')
    loss_fn = model.Loss()
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=1e-4, weight_decay=0.05)
    epochs = 3
    scheduler = model.WarmupCosineSchedule(optimizer, warmup_steps=5, t_total=epochs*len(train_loader))
    print("Train start")
    model.train(reg_model, loss_fn, optimizer, scheduler, train_loader, epochs, device)

    print("Predict start")
    model_results = model.predict_on_loader(reg_model, compete_sequences_loader, device)

    np.savetxt(f'output.txt', model_results, fmt='%s')
    print()
    print("Complete! results at: output.txt")


if __name__ == '__main__':
    main()