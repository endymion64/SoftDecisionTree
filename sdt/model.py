import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


class SoftDecisionTree(nn.Module):
    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.depth = self.args.depth
        # Inner nodes
        self.num_inners = 2 ** self.depth - 1
        self.weights = nn.Parameter(torch.empty(self.num_inners, self.args.input_dim), requires_grad=True)
        self.biases = nn.Parameter(torch.empty(self.num_inners), requires_grad=True)
        self.betas = nn.Parameter(torch.empty(self.num_inners), requires_grad=True)
        self.lambda_per_inner = torch.tensor(
            [self.args.lmbda * (2 ** -d) for d in [int(math.log2(x + 1)) for x in range(self.num_inners)]],
            device=args.device)
        self._alpha_target = torch.full((self.num_inners,), fill_value=0.5, device=args.device)
        # Leafs
        self.num_leafs = 2 ** self.depth
        self.leafs = nn.Parameter(torch.empty(self.num_leafs, self.args.output_dim), requires_grad=True)
        # Initializing parameters
        self._reset_parameters()
        # Ancestry
        self.ancestors_inners = [self._ancestors(i) for i in range(self.num_inners)]
        self.ancestors_leafs = [self._ancestors(i + self.num_inners) for i in range(self.num_leafs)]
        # Optimizer
        self.to(device=args.device)
        self.optimizer = optim.Adam(self.parameters())
        # Logging
        self.best_test_loss = math.inf
        self.writer = SummaryWriter(Path(self.args.save)) if self.args.tensorboard else None
        self.step = 0

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5), nonlinearity='sigmoid')
        bound = 1 / math.sqrt(self.args.input_dim)
        nn.init.uniform_(self.biases, -bound, bound)
        nn.init.normal_(self.betas)
        nn.init.normal_(self.leafs)

    def _ancestors(self, node):
        branched_left = []
        branched_right = []
        while node > 0:
            parent_node = math.floor((node - 1) / 2)
            if node % 2 == 0:
                branched_right.append(parent_node)
            else:
                branched_left.append(parent_node)
            node = parent_node
        return dict(left=torch.tensor(list(reversed(branched_left)), dtype=torch.long, device=self.args.device),
                    right=torch.tensor(list(reversed(branched_right)), dtype=torch.long, device=self.args.device))

    def train_(self, train_loader, epoch):
        self.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.args.device), target.to(self.args.device)
            # Optimize
            self.optimizer.zero_grad()
            loss_total, loss_leafs, loss_inners = self._calc_loss(data, target)
            loss_total.backward()
            # Gradient norm clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 40)
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ('
                    f'{100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_total.item():.6f}')
                if self.writer is not None:
                    for loss_name, loss in zip(['loss_total', 'loss_leafs', 'loss_inners'],
                                               [loss_total, loss_leafs, loss_inners]):
                        self.writer.add_scalar(f'Training/{loss_name}', loss.item(), self.step)
                    if batch_idx % (10 * self.args.log_interval) == 0:
                        for name, param in self.named_parameters():
                            self.writer.add_histogram(name, param.clone().cpu(), self.step)
                            self.writer.add_histogram(name + '/grad_norm', param.grad.data.norm(), self.step)
            self.step += 1

    def _calc_path_probs(self, probs_right, ancestors):
        path_probs_per_ancestor = []
        probs_left = torch.ones_like(probs_right) - probs_right

        for d in ancestors:
            start_probs = torch.ones(probs_left.shape[0], 1, device=self.args.device)
            left_descent_probs = probs_left.index_select(1, d['left'])
            right_descent_probs = probs_right.index_select(1, d['right'])
            path_prob_per_sample = torch.cat((start_probs, left_descent_probs, right_descent_probs), dim=1).prod(dim=1)
            path_probs_per_ancestor.append(path_prob_per_sample)

        return torch.stack(path_probs_per_ancestor).t_()

    def _calc_loss(self, x, y):
        # Leaf loss
        leafs_pred = F.log_softmax(self.leafs, dim=1)
        probs_right = torch.sigmoid(self.betas * torch.addmm(self.biases, x, self.weights.t()))
        leaf_path_probs = self._calc_path_probs(probs_right, self.ancestors_leafs)
        # loss_leafs = torch.sum(leaf_path_probs * y.matmul(leafs_pred.t()), dim=1).neg().log().mean() # Loss
        # according to paper, yet this diverges after a couple of epochs
        loss_leafs = torch.sum(leaf_path_probs * y.matmul(leafs_pred.t()), dim=1).neg().mean()

        # Regularization inners: tree balancing by binary cross-entropy with discrete uniform(2) distribution
        inner_path_probs = self._calc_path_probs(probs_right, self.ancestors_inners)

        # clamps to avoid errors in binary_cross_entropy
        alpha_inners = torch.clamp_max_(
            torch.sum(inner_path_probs * probs_right, dim=0) / torch.sum(inner_path_probs, dim=0).clamp_min_(1e-5), 1)

        loss_inners = F.binary_cross_entropy(alpha_inners, self._alpha_target, weight=self.lambda_per_inner,
                                             reduction='sum')

        total_loss = loss_leafs + loss_inners

        return total_loss, loss_leafs, loss_inners

    @torch.no_grad()
    def test_(self, test_loader, epoch):
        self.eval()

        set_test_loss_total = torch.tensor(0, dtype=torch.float32)
        set_test_loss_inners = torch.tensor(0, dtype=torch.float32)
        set_test_loss_leafs = torch.tensor(0, dtype=torch.float32)

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.args.device), target.to(self.args.device)
            test_loss_total, test_loss_leafs, test_loss_inners = self._calc_loss(data, target)

            set_test_loss_total += ((test_loss_total - set_test_loss_total) / (batch_idx + 1))
            set_test_loss_inners += ((test_loss_inners - set_test_loss_inners) / (batch_idx + 1))
            set_test_loss_leafs += ((test_loss_leafs - set_test_loss_leafs) / (batch_idx + 1))

        print(f'\nTest Loss: {set_test_loss_total.item():.6f}\n')

        if self.writer is not None:
            for loss_name, loss in zip(['loss_total', 'loss_leafs', 'loss_inners'],
                                       [set_test_loss_total, set_test_loss_leafs, set_test_loss_inners]):
                self.writer.add_scalar(f'Test/{loss_name}', loss.item(), epoch)

        if set_test_loss_total < self.best_test_loss:
            self.save(self.args.save, 'best.pt')
            print('Saved a new best model!')
            self.best_test_loss = set_test_loss_total

    def save(self, folder_path, save_name):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        with open(Path(folder_path, save_name), 'wb') as output_file:
            torch.save(self.state_dict(), output_file)
