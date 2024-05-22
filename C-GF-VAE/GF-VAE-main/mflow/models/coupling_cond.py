import torch
import torch.nn as nn
import torch.nn.functional as F
from mflow.models.basic import GraphLinear, GraphConv, ActNorm, ActNorm2D
import math

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=False):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mask_swap = mask_swap
        last_h = math.ceil(in_channel / 2)
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
            vh = tuple(hidden_channels) + (math.ceil(in_channel / 2),)
        for h in vh:
            self.layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
            self.norms.append(nn.BatchNorm2d(h))
            last_h = h

        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.fc_branch = nn.Sequential(
            nn.Linear(200, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, in_channel * 2 * 2),  
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Unflatten(1, (in_channel , 2, 2))  
        )

    def forward(self, input, C):
        in_a, in_b = input.chunk(2, 1)
        if self.mask_swap:
            in_a, in_b = in_b, in_a

        if self.affine:
            s, t = self._s_t_function(in_a, C)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
        else:
            _, t = self._s_t_function(in_a, C)
            out_b = in_b + t
            logdet = None

        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    def reverse(self, output, C):
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        if self.affine:
            s, t = self._s_t_function(out_a, C)
            in_b = out_b / s - t
        else:
            _, t = self._s_t_function(out_a, C)
            in_b = out_b - t

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result

    def _s_t_function(self, x, C):
        # print("X FROM _s_t_function SHAPE: ", str(x.shape))

        h = x
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            h = torch.relu(h)
        h = self.layers[-1](h)

        conv_out = self.conv_branch(h)
        fc_out = self.fc_branch(C)
        combined = torch.cat((conv_out, fc_out), dim=1)
        fc_out = self.combined_fc(combined)
        #print(f"fc_out shape after combined_fc: {fc_out.shape}")  # Debugging print
        h = self.output_layer(fc_out)
        #print(f"h shape after output_layer: {h.shape}")  # Debugging print

        if self.affine:
            log_s, t = h.chunk(2, dim=1)
            # print(f"log_s shape: {log_s.shape}, t shape: {t.shape}")  # Debugging print
            log_s = log_s.view(x.size(0), -1, 2, 2)
            t = t.view(x.size(0), -1, 2, 2)
            s = torch.sigmoid(log_s).expand_as(x)
            t = t.expand_as(x)
        else:
            t = h.view(x.size(0), -1, 2, 2).expand_as(x)

        return s, t


class GraphAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine

        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']

        self.net = nn.ModuleList()
        self.norm = nn.ModuleList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1d(n_node))
            last_dim = out_dim

        self.net_lin = nn.ModuleList()
        self.norm_lin = nn.ModuleList()
        for i, out_dim in enumerate(self.hidden_dim_linear):
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(nn.BatchNorm1d(n_node))
            last_dim = out_dim

        final_dim = in_dim * 2 if affine else in_dim  
        self.net_lin.append(GraphLinear(last_dim, final_dim))

        self.scale = nn.Parameter(torch.zeros(1))
        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0
        self.register_buffer('mask', mask)
        # print("N_NODE:", n_node)
        self.conv_branch = nn.Sequential(
            nn.Conv1d(in_channels=n_node, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.fc_branch = nn.Sequential(
            nn.Linear(200, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(608, 256), 
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, n_node * 30),  
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Unflatten(1, (n_node, 30)) 
        )

    def forward(self, adj, input, C):
        masked_x = self.mask * input
        s, t = self._s_t_function(adj, masked_x, C)

        #print(f"masked_x shape: {masked_x.shape}")  # Debugging print
        #print(f"s shape: {s.shape}, t shape: {t.shape}")  # Debugging print

        if self.affine:
            out = masked_x + (1 - self.mask) * (input + t) * s
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
        else:
            out = masked_x + t * (1 - self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output, C):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y, C)

        #print(f"masked_y shape: {masked_y.shape}")  # Debugging print
        #print(f"s shape: {s.shape}, t shape: {t.shape}")  # Debugging print

        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x, C):
        #print("X FROM _s_t_function SHAPE: ", str(x.shape))
        s = None
        h = x

        for i in range(len(self.net)):
            h = self.net[i](adj, h)
            h = self.norm[i](h)
            h = torch.relu(h)

        for i in range(len(self.net_lin) - 1):
            h = self.net_lin[i](h)
            h = self.norm_lin[i](h)
            h = torch.relu(h)

        h = self.net_lin[-1](h)
        #print(f"h shape before conv_branch: {h.shape}")  # Debugging print

        conv_out = self.conv_branch(h)
        #print(f"conv_out shape: {conv_out.shape}")  # Debugging print

        C_flatten = C.view(C.size(0), -1)
        #print(f"C_flatten shape: {C_flatten.shape}")  # Debugging print

        fc_out = self.fc_branch(C_flatten)
        #print(f"fc_out shape: {fc_out.shape}")  # Debugging print

        combined = torch.cat((conv_out, fc_out), dim=1)
        #print(f"combined shape: {combined.shape}")  # Debugging print

        fc_out = self.combined_fc(combined)
        #print(f"fc_out after combined_fc shape: {fc_out.shape}")  # Debugging print

        h = self.output_layer(fc_out)
        #print(f"h shape after output_layer: {h.shape}")  # Debugging print

        if self.affine:
            log_s, t = h.chunk(2, dim=-1)
            #print('log_s SHAPE:', log_s.shape, 't SHAPE:', t.shape)  # Debugging print
            s = torch.sigmoid(log_s)
        else:
            t = h

        return s, t
