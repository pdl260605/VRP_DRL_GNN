import torch
import torch.nn as nn
# from torchsummary import summary

from layers import MultiHeadAttention
from data import generate_data
import math
from torch_geometric.nn import GCNConv
class Normalization(nn.Module):

	def __init__(self, embed_dim, normalization = 'batch'):
		super().__init__()

		normalizer_class = {
			'batch': nn.BatchNorm1d,
			'instance': nn.InstanceNorm1d}.get(normalization, None)
		self.normalizer = normalizer_class(embed_dim, affine=True)
		# Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
	# 	self.init_parameters()

	# def init_parameters(self):
	# 	for name, param in self.named_parameters():
	# 		stdv = 1. / math.sqrt(param.size(-1))
	# 		param.data.uniform_(-stdv, stdv)

	def forward(self, x):

		if isinstance(self.normalizer, nn.BatchNorm1d):
			# (batch, num_features)
			# https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
			return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
		
		elif isinstance(self.normalizer, nn.InstanceNorm1d):
			return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
		else:
			assert self.normalizer is None, "Unknown normalizer type"
			return x


class ResidualBlock_BN(nn.Module):
	def __init__(self, MHA, BN, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA
		self.BN = BN

	def forward(self, x, mask = None):
		if mask is None:
			return self.BN(x + self.MHA(x))
		return self.BN(x + self.MHA(x, mask))

class SelfAttention(nn.Module):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA

	def forward(self, x, mask = None):
		return self.MHA([x, x, x], mask = mask)

class EncoderLayer(nn.Module):
	# nn.Sequential):
	def __init__(self, n_heads = 8, FF_hidden = 512, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.FF_hidden = FF_hidden
		self.BN1 = Normalization(embed_dim, normalization = 'batch')
		self.BN2 = Normalization(embed_dim, normalization = 'batch')

		self.MHA_sublayer = ResidualBlock_BN(
				SelfAttention(
					MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, need_W = True)
				),
			self.BN1
			)

		self.FF_sublayer = ResidualBlock_BN(
			nn.Sequential(
					nn.Linear(embed_dim, FF_hidden, bias = True),
					nn.ReLU(),
					nn.Linear(FF_hidden, embed_dim, bias = True)
			),
			self.BN2
		)
		
	def forward(self, x, mask=None):
		"""	arg x: (batch, n_nodes, embed_dim)
			return: (batch, n_nodes, embed_dim)
		"""
		return self.FF_sublayer(self.MHA_sublayer(x, mask = mask))

#using graph Neural Network
class GNNLayer(nn.Module):
	def __init__(self, embed_dim=128, **kwargs):
		super().__init__(**kwargs)
		# Use PyTorch Geometric or implement custom GNN
		# Option 1: GCNConv

		self.gnn = GCNConv(embed_dim, embed_dim)


		self.BN = Normalization(embed_dim, normalization='batch')
		self.FF = nn.Sequential(
			nn.Linear(embed_dim, 512, bias=True),
			nn.ReLU(),
			nn.Linear(512, embed_dim, bias=True)
		)

	def forward(self, x, edge_index):
		# x: node features (batch, n_nodes, embed_dim)
		# edge_index: graph connectivity (PyTorch Geometric format)
		x_gnn = self.gnn(x, edge_index)
		x = self.BN(x + x_gnn)
		x = self.BN(x + self.FF(x))
		return x
		
class GraphAttentionEncoder(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, n_layers = 3, FF_hidden = 512):
		super().__init__()
		self.init_W_depot = torch.nn.Linear(2, embed_dim, bias = True)
		self.init_W = torch.nn.Linear(3, embed_dim, bias = True)
		self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])
	
	def forward(self, x, mask = None):
		""" x[0] -- depot_xy: (batch, 2) --> embed_depot_xy: (batch, embed_dim)
			x[1] -- customer_xy: (batch, n_nodes-1, 2)
			x[2] -- demand: (batch, n_nodes-1)
			--> concated_customer_feature: (batch, n_nodes-1, 3) --> embed_customer_feature: (batch, n_nodes-1, embed_dim)
			embed_x(batch, n_nodes, embed_dim)

			return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
				=((batch, n_nodes, embed_dim), (batch, embed_dim))
		"""
		x = torch.cat([self.init_W_depot(x[0])[:, None, :],
				self.init_W(torch.cat([x[1], x[2][:, :, None]], dim = -1))], dim = 1)
	
		for layer in self.encoder_layers:
			x = layer(x, mask)

		return (x, torch.mean(x, dim = 1))

#encoder using GNN
class GraphNeuralEncoder(nn.Module):
	def __init__(self, embed_dim=128, n_layers=3, FF_hidden=512):
		super().__init__()
		self.init_W_depot = torch.nn.Linear(2, embed_dim, bias=True)
		self.init_W = torch.nn.Linear(3, embed_dim, bias=True)

		# GNN layers instead of attention layers
		self.gnn_layers = nn.ModuleList([
			GNNLayer(embed_dim, FF_hidden) for _ in range(n_layers)
		])

	def _create_edge_index(self, n_nodes, device, batch_size):
		"""Create fully-connected graph edge index for all nodes"""
		# For VRP, typically use fully-connected graph
		edge_index = torch.combinations(
			torch.arange(n_nodes, device=device), r=2
		).t().contiguous()

		# Extend for batch dimension
		edge_index_list = []
		for b in range(batch_size):
			edge_index_list.append(edge_index + b * n_nodes)
		return torch.cat(edge_index_list, dim=1)

	def forward(self, x):
		"""
        x[0]: depot_xy (batch, 2)
        x[1]: customer_xy (batch, n_nodes-1, 2)
        x[2]: demand (batch, n_nodes-1)
        """
		batch_size = x[0].size(0)
		device = x[0].device

		# Embed input features
		depot_embed = self.init_W_depot(x[0])[:, None, :]
		customer_embed = self.init_W(torch.cat([x[1], x[2][:, :, None]], dim=-1))
		node_features = torch.cat([depot_embed, customer_embed], dim=1)

		n_nodes = node_features.size(1)

		# Reshape for GNN processing (flatten batch)
		node_features = node_features.view(batch_size * n_nodes, -1)

		# Create edge indices
		edge_index = self._create_edge_index(n_nodes, device, batch_size)

		# Apply GNN layers
		for layer in self.gnn_layers:
			node_features = layer(node_features, edge_index)

		# Reshape back to (batch, n_nodes, embed_dim)
		node_features = node_features.view(batch_size, n_nodes, -1)

		# Return node embeddings and graph embedding
		return (node_features, torch.mean(node_features, dim=1))
if __name__ == '__main__':
	batch = 5
	n_nodes = 21
	encoder = GraphAttentionEncoder(n_layers = 1)
	data = generate_data(n_samples = batch, n_customer = n_nodes-1)
	# mask = torch.zeros((batch, n_nodes, 1), dtype = bool)
	output = encoder(data, mask = None)
	print('output[0].shape:', output[0].size())
	print('output[1].shape', output[1].size())
	
	# summary(encoder, [(2), (20,2), (20)])
	cnt = 0
	for i, k in encoder.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)
	print(cnt)

	# output[0].mean().backward()
	# print(encoder.init_W_depot.weight.grad)

