import torch
from torch import nn
import time
import numpy as np

class Model(nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, num_filters=128, kernel_size=3):
            super(Model, self).__init__()
            self.embed = nn.Embedding(vocab_size, embedding_dim)
            self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.bn = nn.BatchNorm1d(num_filters)  # Batch normalization layer
            self.relu = nn.ReLU()  # Activation function
            self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling
            self.linear = nn.Linear(num_filters, 1)

        def forward(self, x):
            x = self.embed(x).permute(0, 2, 1)  # [batch, embedding_dim, sequence_length]
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x).view(x.size(0), -1)  # Flatten the output for the linear layer
            x = self.linear(x)
            return x.squeeze()


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, prediction, target):
        target = target.float()
        loss = nn.functional.mse_loss(prediction, target)
        return loss 
    
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupCosineSchedule, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.last_epoch / max(1, self.warmup_steps) * base_lr for base_lr in self.base_lrs]
        return [0.5 * (1.0 + np.cos(np.pi * (self.last_epoch - self.warmup_steps) / (self.t_total - self.warmup_steps))) * base_lr for base_lr in self.base_lrs]
        

def show_progress(epoch, step, total_steps, loss, added_text='', width=30, bar_char='█', empty_char='░'):
    print('\r', end='')
    progress = ""
    for i in range(width):
        progress += bar_char if i < int(step / total_steps * width) else empty_char
    print(f"epoch:{epoch + 1} [{progress}] {step}/{total_steps} loss: {loss:.4f}" + added_text, end='')
        

def train(model, loss, optimizer, scheduler, train_dataloader, epochs, device):
    model.train()
    epoch = 0
    while epoch < epochs:
        running_loss = 0.0
        start = time.time()
        for i, (feature, target) in enumerate(train_dataloader):
            feature = feature.to(device)
            target = target.to(device)
            with torch.autocast('cuda'):
                output = model.forward(feature)
                loss_val = loss(output, target)
            # model.zero_grad()
            optimizer.zero_grad()
            loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            feature.detach()
            target.detach()
            output.detach()
            with torch.no_grad():
                running_loss += loss_val.item()
                show_progress(epoch, i, len(train_dataloader), running_loss/(i + 1), f'')
        with torch.no_grad():
            show_progress(epoch, i, len(train_dataloader), running_loss/(i + 1), f' time: {time.time() - start:.4f}')
            print()
        epoch += 1


def predict_on_loader(model, intensities_df_dataloader, device):
    predicted_intensities = None
    model.eval()
    with torch.no_grad():
        for i, features in enumerate(intensities_df_dataloader):
            print(f'\r{i}/{len(intensities_df_dataloader)}', end='')
            features = features.to(device)
            with torch.autocast('cuda'):
                output = model.forward(features)
            features.detach()
            output.detach()
            for j in range(output.shape[0]):
                if predicted_intensities is None:
                    predicted_intensities = np.array(output[j].cpu().numpy().reshape(1, -1))
                    continue
                predicted_intensities = np.concatenate([predicted_intensities, output[j].cpu().numpy().reshape(1, -1)], axis=0)
    #             predicted_intensities.append(output[j].cpu().numpy())
            del output
            del features


    predicted_intensities = np.array(predicted_intensities)

    return predicted_intensities