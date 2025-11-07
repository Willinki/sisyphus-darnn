import torch
from local_exp.PyTorch_CIFAR10.data import CIFAR10Data
from local_exp.PyTorch_CIFAR10.module import CIFAR10Module
from torchvision.models.feature_extraction import create_feature_extractor


def write_data(data: list[torch.Tensor], name, i):
    data = torch.cat(data)
    torch.save(data, f"cifar10/{name}_{i}.pt")
    return []


def main():
    N = 1000
    data = CIFAR10Data(data_dir=".", batch_size=1)
    classifer = CIFAR10Module(classifier="vgg11_bn")
    model = classifer.model
    return_nodes = {"avgpool": "mlp_input", "classifier.1": "mlp_layer1"}
    feature_extractor = create_feature_extractor(model, return_nodes)

    # save train set
    print("saving training")
    data_mlp_in = []
    data_mlp_l1 = []
    ys = []
    i = 0
    for x, y in data.train_dataloader():
        i += 1
        output = feature_extractor(x)
        data_mlp_in.append(output["mlp_input"])
        data_mlp_l1.append(output["mlp_layer1"])
        ys.append(y)
        if i % N == 0:
            data_mlp_in = write_data(data_mlp_in, "train_mlpinput", i // N)
            data_mlp_l1 = write_data(data_mlp_l1, "train_mlpl1", i // N)
            ys = write_data(ys, "train_y", i // N)

    # save eval set
    print("saving validation")
    data_mlp_in = []
    data_mlp_l1 = []
    ys = []
    i = 0
    for x, y in data.val_dataloader():
        i += 1
        output = feature_extractor(x)
        data_mlp_in.append(output["mlp_input"])
        data_mlp_l1.append(output["mlp_layer1"])
        ys.append(y)
        if i % N == 0:
            data_mlp_in = write_data(data_mlp_in, "eval_mlpinput", i // N)
            data_mlp_l1 = write_data(data_mlp_l1, "eval_mlpl1", i // N)
            ys = write_data(ys, "eval_y", i // N)


if __name__ == "__main__":
    main()
