from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloader(args):
    if args.dataset == 'cifar100':
        train_transforms = []
        test_transforms = []
        if args.img_size != 32:
            train_transforms.append(transforms.Resize((args.img_size, args.img_size)))
            test_transforms.append(transforms.Resize((args.img_size, args.img_size)))

        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
        test_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

        transform_train = transforms.Compose(train_transforms)
        transform_test = transforms.Compose(test_transforms)

        trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)

        common_loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            common_loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            **common_loader_kwargs,
        )
        test_loader = DataLoader(
            testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **common_loader_kwargs,
        )
        
        return train_loader, test_loader
