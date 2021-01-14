from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

if __name__ == '__main__':


    print("Loading model...")
    checkpoint_path = "./densenet_models/densenet201_e_80_lr_0_0001_best.pth"
    model_name = "densenet201"

    # checkpoint_path = "./resnet_models/resnet101_e_80_lr_2e-05_best.pth"
    # model_name = "resnet101"

    enrollment_data_path = "./CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment"
    test_data_path = "./CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test"
    batch_size = 256

    model, input_size = get_model(model_name, checkpoint_path)

    device = torch.device('cuda')
    model.to(device)
    model.eval()


    enrollment_dataloader = get_dataloader(test_data_path, input_size, batch_size=batch_size)
    test_dataloader = get_dataloader(enrollment_data_path, input_size, batch_size=batch_size)

    print("Enrolling identities...")

    enrolled = {}
    with torch.no_grad():
        for input, labels in enrollment_dataloader:
            inputs = input.to(device)
            labels = labels.cpu().detach().numpy()

            # Extract the features using the CNN
            predictions = model.feature_extract_avg_pool(inputs).cpu().detach().numpy()

            # Create a matrix for each users, where a row represents a feature vector extracted from the enrollment image and
            # normalize the matrix bx rows (to reduce the amount of computation in the recognition phase)
            # Results is a dictionary, where a key is a specific identity with the entry containing a matrix where a row is x / ||x||,
            # where x is a feature vector for a given image
            unique_labels = np.unique(labels)
            for i in unique_labels:
                user_features = predictions[labels == i,:]
                if i in enrolled:
                    enrolled[i] = np.vstack((enrolled[i], normalize(user_features, axis=1, norm='l2')))
                else:
                    enrolled[i] = normalize(user_features, axis=1, norm='l2')
    print("Recognizing...")

    total = 0
    correct = 0
    with torch.no_grad():
        for input, labels in test_dataloader:
            inputs = input.to(device)
            labels = labels.cpu().detach().numpy()
            predictions = model.feature_extract_avg_pool(inputs).cpu().detach().numpy()
            for idx, label in enumerate(labels):
                pred = predictions[idx,:].reshape(-1,1)
                pred_norm = normalize(pred, axis=0, norm="l2")
                similarities_id = {}
                for key in enrolled.keys():
                    cosine_similarities = np.matmul(enrolled[key], pred_norm)
                    similarities_id[key] = np.max(cosine_similarities)

                recognized_key = max(similarities_id, key=similarities_id.get)
                total += 1
                if label == recognized_key:
                    correct += 1
                print(f"Ground truth label: {label}, prediction: {recognized_key}")

    print(f"Accuracy: {correct / total}")