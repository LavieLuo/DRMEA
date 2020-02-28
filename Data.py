from torchvision import datasets, transforms
import torch.utils.data as data



def Get_Domain_Meta(Dataset_Name):
    if Dataset_Name == 'ImageCLEF':
        source_domain_set = ('ImageNet','Pascal','ImageNet','Caltech','Caltech','Pascal')
        target_domain_set = ('Pascal','ImageNet','Caltech','ImageNet','Pascal','Caltech')
        save_name = ('I2P','P2I','I2C','C2I','C2P','P2C')
        num_classes = 12
    return source_domain_set, target_domain_set, save_name, num_classes



def data_loader(Dataset_Name, Domain, batch_size):
    means = {
        'imagenet': [0.485, 0.456, 0.406]
    }
    stds = {
        'imagenet': [0.229, 0.224, 0.225]
    }
    if Dataset_Name == 'ImageCLEF':
        datas = {
            'ImageNet': '../Dataset/image_CLEF_torch/i/',
            'Pascal': '../Dataset/image_CLEF_torch/p/',
            'Caltech': '../Dataset/image_CLEF_torch/c/'
        }
        transform = [
            transforms.Scale((256, 256)),
            transforms.Scale(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means['imagenet'], stds['imagenet']),
        ]
        data_loader = data.DataLoader(
            dataset=datasets.ImageFolder(
                datas[Domain],
                transform=transforms.Compose(transform)
            ),
            batch_size=batch_size,num_workers=4,
            shuffle=True,
        )
    return data_loader
            
            