import torchvision

models_dict = {
    'inception_v3':  torchvision.models.inception_v3(pretrained =True),
    'resnet18':      torchvision.models.resnet18(pretrained =True),
    'resnet50':      torchvision.models.resnet50(pretrained =True),
    'resnet101':     torchvision.models.resnet101(pretrained =True),
    'googlenet':     torchvision.models.googlenet(pretrained =True),
    'deeplabv3_resnet50': torchvision.models.segmentation.deeplabv3_resnet50(pretrained= True)
}