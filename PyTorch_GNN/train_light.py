import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import sys

from model_light import GNNLightModel
from baseline import RolloutBaseline
from data import generate_data, Generator
from config_light import Config, load_pkl, train_parser


def train(cfg, log_path=None):
    """Train lightweight GNN model"""
    torch.backends.cudnn.benchmark = True

    def rein_loss(model, inputs, bs, t, device):
        """Compute REINFORCE loss"""
        L, ll = model(inputs, decode_type='sampling')
        b = bs[t] if bs is not None else baseline.eval(inputs, L)
        return ((L - b.to(device)) * ll).mean(), L.mean()

    # Initialize lightweight model
    model = GNNLightModel(
        embed_dim=cfg.embed_dim,
        n_encode_layers=cfg.n_encode_layers,
        tanh_clipping=cfg.tanh_clipping
    )
    model.train()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 70}")
    print(f"  GNN Light Model - Lightweight VRP Solver")
    print(f"{'=' * 70}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Embedding Dimension: {cfg.embed_dim}")
    print(f"  GNN Layers: {cfg.n_encode_layers}")
    print(f"  Batch Size: {cfg.batch}")
    print(f"  Learning Rate: {cfg.lr}")
    print(f"{'=' * 70}\n")

    # Initialize baseline and optimizer
    baseline = RolloutBaseline(
        model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples,
        cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs, device
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    t1 = time()
    for epoch in range(cfg.epochs):
        ave_loss, ave_L = 0., 0.

        # Generate training data
        dataset = Generator(device, cfg.batch * cfg.batch_steps, cfg.n_customer)

        # Evaluate baseline
        bs = baseline.eval_all(dataset)
        bs = bs.view(-1, cfg.batch) if bs is not None else None

        # Training loop
        dataloader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)
        for t, inputs in enumerate(dataloader):

            # Compute loss
            loss, L_mean = rein_loss(model, inputs, bs, t, device)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            ave_loss += loss.item()
            ave_L += L_mean.item()

            # Logging
            if t % cfg.batch_verbose == 0:
                t2 = time()
                elapsed_min = int((t2 - t1) // 60)
                elapsed_sec = int((t2 - t1) % 60)
                print(
                    f'Epoch {epoch:2d} (batch {t:4d}): '
                    f'Loss: {ave_loss / (t + 1):1.3f} | '
                    f'Cost: {ave_L / (t + 1):1.3f} | '
                    f'{elapsed_min:03d}m{elapsed_sec:02d}s'
                )

                if cfg.islogger:
                    if log_path is None:
                        log_path = f'{cfg.log_dir}{cfg.task}_{cfg.dump_date}.csv'
                        with open(log_path, 'w') as f:
                            f.write('time,epoch,batch,loss,cost\n')

                    with open(log_path, 'a') as f:
                        f.write(
                            f'{elapsed_min:3d}m{elapsed_sec:02d}s,'
                            f'{epoch},{t},{ave_loss / (t + 1):1.3f},{ave_L / (t + 1):1.3f}\n'
                        )
                t1 = time()

        # Save checkpoint
        baseline.epoch_callback(model, epoch)
        weight_path = f'{cfg.weight_dir}{cfg.task}_GNN_Light_epoch{epoch}.pt'
        torch.save(model.state_dict(), weight_path)
        print(f'  → Model saved: {weight_path}\n')

    print(f"\n{'=' * 70}")
    print(f"  Training completed!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    cfg = load_pkl(train_parser().path)
    train(cfg)