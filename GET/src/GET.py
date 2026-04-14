import time

import GEBlocks
import torch
import torch.nn as nn
from tqdm import tqdm


class GETClassifier(nn.Module):
    def __init__(self, N, channels, out_classes):
        super().__init__()
        self.N = N
        self.channels = channels

        self.local_to_regular = GEBlocks.GELocalToRegularLinearBlock(N, channels)
        self.self_attention1 = GEBlocks.GESelfAttentionBlock(N, channels)
        self.self_attention2 = GEBlocks.GESelfAttentionBlock(N, channels)
        self.group_pool = GEBlocks.GEGroupPoolingBlock()
        self.global_average_pool = GEBlocks.GEGlobalAveragePoolingBlock()
        self.fc = nn.Linear(channels, out_classes)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        # x: (B, N_v, 3)

        # Local to regular transformation
        x0 = self.local_to_regular(x)  # (B, N_v, channels, N)
        x0 = torch.relu(x0)  # (B, N_v, channels, N)

        # Self-attention
        x = self.self_attention1(
            x0, neighbors, mask, parallel_transport_matrices, rel_pos_u
        )  # (B, N_v, in_channels, N)

        x = torch.relu(x)  # (B, N_v, in_channels, N)

        x = self.self_attention2(
            x, neighbors, mask, parallel_transport_matrices, rel_pos_u
        )  # (B, N_v, in_channels, N)

        x = x + x0  # Residual connection

        x = self.group_pool(x)  # (B, N_v, in_channels)
        x = self.global_average_pool(x)  # (B, in_channels)

        return self.fc(x)


def train(model, dataloader, optimizer, criterion, device, epochs=1):
    model.train()
    loss_hist = []
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        for mesh in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, neighbors, mask, parallel_transport_matrices, rel_pos_u, labels = mesh
            x = x.to(device)
            neighbors = neighbors.to(device)
            mask = mask.to(device)
            parallel_transport_matrices = parallel_transport_matrices.to(device)
            rel_pos_u = rel_pos_u.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(x, neighbors, mask, parallel_transport_matrices, rel_pos_u)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        epoch_loss = total_loss / len(dataloader)
        loss_hist.append(epoch_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} finished in {epoch_time:.2f}s - avg loss: {epoch_loss:.4f}"
        )

        save_path = f"model_final_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model state_dict to {save_path}")

    return loss_hist
    
