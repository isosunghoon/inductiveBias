from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloader(args):
    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

        trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)

        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                              num_workers=args.num_workers, pin_memory=True)
        
        return train_loader, test_loader