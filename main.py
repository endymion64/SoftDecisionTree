import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from args import parser
from sdt.model import SoftDecisionTree

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        args.device = torch.device('cuda')
        if args.cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print('Using GPU')
    else:
        args.device = torch.device('cpu')
        print('Using CPU')

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(torch.flatten)])

    target_transform = transforms.Lambda(
        lambda t: torch.as_tensor(torch.nn.functional.one_hot(torch.tensor(t), num_classes=10), dtype=torch.float))

    train_loader = DataLoader(datasets.MNIST(args.root, train=True, download=True, transform=data_transform,
                                             target_transform=target_transform),
                              batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(datasets.MNIST(args.root, train=False, transform=data_transform,
                                            target_transform=target_transform),
                             batch_size=args.batch_size, shuffle=True, **kwargs)

    soft_dec_tree = SoftDecisionTree(args)

    for epoch in range(1, args.epochs + 1):
        soft_dec_tree.train_(train_loader, epoch)
        soft_dec_tree.test_(test_loader, epoch)

    soft_dec_tree.save(args.save, 'final.pt')
    print('Saved the final resulted model.')
