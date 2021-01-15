import json
import pathlib

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
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__':


    # checkpoint_path = "./models/densenet201_e_80_lr_0_0001_best.pth"
    # model_name = "densenet201"

    checkpoint_path = "./models/resnet101_e_80_lr_2e-05_best.pth"
    model_name = "resnet101"

    test_data_path = "./data/CASIA_thousand_norm_256_64_e_nn_stacked/test"

    print("Loading model...")
    model, input_size = get_model(model_name, checkpoint_path)

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    dataloader = get_dataloader(test_data_path, input_size)

    print("Running evaluation....")
    rank_n = 50
    total = 0
    rank_n_correct = np.zeros(rank_n)
    with torch.no_grad():
        for inputs, labels in dataloader:
            input = inputs.to(device)
            prediction = model(input).cpu()

            labels_np = labels.detach().numpy()
            labels_prob = softmax(prediction, dim=1).detach().numpy()
            rank_n_pred = (-labels_prob).argsort(axis=-1)[:, :rank_n]
            for i in range(labels_np.size):
                rank_n_pred_ith_label = list(rank_n_pred[i,:])
                total +=1
                for rank in range(rank_n):
                    rank_n_correct[rank] += 1 if labels_np[i] in rank_n_pred_ith_label[:rank+1] else 0

    rank_n_correct /= total

    rank_1_accuracy = rank_n_correct[0]
    rank_5_accuracy = rank_n_correct[4]

    print(f"Rank-1 accuracy: {rank_1_accuracy}, rank-5 accuracy: {rank_5_accuracy}")

    results = {
        "rank_1_acc": rank_1_accuracy,
        "rank_5_acc": rank_5_accuracy,
        "rank_n_accuracies": list(rank_n_correct)
    }

    pathlib.Path("./results").mkdir(parents=True, exist_ok=True)

    with open(f'./results/{model_name}_results_closed_set.json', 'w') as f:
        json.dump(results, f)