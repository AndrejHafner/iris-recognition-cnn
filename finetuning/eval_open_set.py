from collections import defaultdict

from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, models, transforms
import numpy as np

from models import DenseNet161Iris, ResNet101Iris, ResNet152Iris, InceptionV3Iris, DenseNet201Iris


def get_model(model_name, checkpoint_path, num_classes=1500):

    model = None
    input_size = 0

    if model_name == "resnet101":
        model = ResNet101Iris(num_classes=num_classes)
        input_size = 224
        model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == "resnet152":
        model = ResNet152Iris(num_classes=num_classes)
        input_size = 224
        model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == "densenet161":
        model = DenseNet161Iris(num_classes=num_classes)
        input_size = 224
        model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == "densenet201":
        model = DenseNet201Iris(num_classes=num_classes)
        input_size = 224
        model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == "inception":
        model = InceptionV3Iris(num_classes=num_classes)
        input_size = 299
        model.load_state_dict(torch.load(checkpoint_path))

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

def get_dataloader(data_path, input_size, batch_size=32):

    transform = transforms.Compose([
        # transforms.Pad((0, (192) // 2)),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__':


    print("Loading model...")
    # checkpoint_path = "./densenet_models/densenet201_e_80_lr_0_0001_best.pth"
    # model_name = "densenet201"
    # test_data_path = "./CASIA_thousand_norm_256_64_e_nn_stacked/test"

    checkpoint_path = "./resnet_models/resnet101_e_80_lr_2e-05_best.pth"
    model_name = "resnet101"
    enrollment_data_path = "./"
    test_data_path = "./CASIA_thousand_norm_256_64_e_nn_stacked/test"
    batch_size = 1

    model, input_size = get_model(model_name, checkpoint_path)

    device = torch.device('cuda')
    model.to(device)
    model.eval()


    enrollment_dataloader = get_dataloader(enrollment_data_path, input_size, batch_size=batch_size)
    test_dataloader = get_dataloader(test_data_path, input_size, batch_size=batch_size)

    print("Enrolling identities...")

    enrolled = defaultdict(list)
    with torch.no_grad():
        for input, labels in enrollment_dataloader:
            inputs = input.to(device)
            prediction = model.feature_extract_avg_pool(inputs).cpu().detach().numpy()
            for idx, label in labels:
                enrolled[label].append(prediction[idx,:])
            e = 0


    print("Recognizing...")