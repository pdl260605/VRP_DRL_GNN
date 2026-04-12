import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import argparse
from tqdm import tqdm

from model_light import GNNLightModel
from data import data_from_txt, generate_data


def load_model_light(path, embed_dim=64, n_customer=20, n_encode_layers=2):
    """Load lightweight GNN model"""
    model = GNNLightModel(
        embed_dim=embed_dim,
        n_encode_layers=n_encode_layers,
        tanh_clipping=10.
    )

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    return model


def plot_vrp(depot, customers, solution, title='VRP Solution'):
    """Plot VRP solution"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot depot
    ax.plot(depot[0], depot[1], 'r*', markersize=20, label='Depot', zorder=5)

    # Plot customers
    ax.plot(customers[:, 0], customers[:, 1], 'bo', markersize=8, label='Customers', zorder=4)

    # Add customer indices
    for i, (x, y) in enumerate(customers):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot routes
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution)))
    for route_idx, route in enumerate(solution):
        route_coords = [depot] + [customers[node] for node in route] + [depot]
        route_coords = np.array(route_coords)
        ax.plot(route_coords[:, 0], route_coords[:, 1], 'o-',
                color=colors[route_idx], linewidth=2, markersize=6,
                label=f'Route {route_idx + 1}', alpha=0.7)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig, ax


def get_tours(model, x, n_customer, batch_size=1, device='cpu'):
    """Get tours from model"""
    x_device = (x[0].to(device), x[1].to(device), x[2].to(device))

    with torch.no_grad():
        cost, _, pi = model(x_device, return_pi=True, decode_type='greedy')

    return pi.cpu().numpy(), cost.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', metavar='P', type=str, required=True,
                        help='Path to model weights (.pt file)')
    parser.add_argument('-b', '--batch', metavar='B', type=int, default=2,
                        help='batch size')
    parser.add_argument('-n', '--n_customer', metavar='N', type=int, default=20,
                        help='number of customer nodes')
    parser.add_argument('-s', '--seed', metavar='S', type=int, default=123,
                        help='random seed')
    parser.add_argument('-t', '--txt', metavar='T', type=str,
                        help='test data file (optional)')
    parser.add_argument('-d', '--decode_type', metavar='D', default='greedy',
                        type=str, choices=['greedy', 'sampling'],
                        help='decoding strategy')
    parser.add_argument('-e', '--embed_dim', metavar='EM', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('-ne', '--n_encode_layers', metavar='NE', type=int, default=2,
                        help='number of GNN layers')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"=" * 60}')
    print(f'Loading Lightweight GNN Model')
    print(f'{"=" * 60}')
    print(f'Model path: {args.path}')
    print(f'Embed dim: {args.embed_dim}')
    print(f'GNN layers: {args.n_encode_layers}')
    print(f'Device: {device}')
    print(f'{"=" * 60}\n')

    # Load model
    pretrained = load_model_light(
        args.path,
        embed_dim=args.embed_dim,
        n_customer=args.n_customer,
        n_encode_layers=args.n_encode_layers
    )
    pretrained.to(device)
    pretrained.eval()

    # Load data
    if args.txt is not None:
        print(f'Loading data from: {args.txt}')
        x = data_from_txt(args.txt)
        x = list(map(lambda t: t.to(device), x))
        n_customer = x[1].size(1)
    else:
        torch.manual_seed(args.seed)
        x = generate_data(device, n_samples=args.batch, n_customer=args.n_customer)
        n_customer = args.n_customer

    # Get predictions
    print(f'Getting predictions...')
    pi, costs = get_tours(pretrained, x, n_customer, args.batch, device)

    print(f'{"=" * 60}')
    print(f'Results')
    print(f'{"=" * 60}')

    depot = x[0].cpu().numpy()
    customers = x[1].cpu().numpy()

    # Plot each sample in batch
    for i in range(min(args.batch, len(pi))):
        print(f'\nSample {i + 1}:')
        print(f'  Route cost: {costs[i]:.3f}')

        # Convert tour to routes
        route = pi[i]

        # Split into multiple routes based on depot visits
        routes = []
        current_route = []
        for node in route:
            if node == 0:  # Depot
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node - 1)  # Adjust for 0-indexing

        if current_route:
            routes.append(current_route)

        print(f'  Number of routes: {len(routes)}')
        for j, r in enumerate(routes):
            print(f'    Route {j + 1}: {r}')

        # Plot
        fig, ax = plot_vrp(
            depot[i],
            customers[i],
            routes,
            title=f'VRP Solution (Sample {i + 1}, Cost: {costs[i]:.3f})'
        )
        plt.tight_layout()
        plt.savefig(f'vrp_solution_sample_{i + 1}.png', dpi=150, bbox_inches='tight')
        print(f'  Saved: vrp_solution_sample_{i + 1}.png')
        plt.show()

    print(f'\n{"=" * 60}')
    print(f'Average cost: {costs.mean():.3f}')
    print(f'Std cost: {costs.std():.3f}')
    print(f'Min cost: {costs.min():.3f}')
    print(f'Max cost: {costs.max():.3f}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()