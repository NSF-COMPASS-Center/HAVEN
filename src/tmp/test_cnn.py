import pandas as pd
import torchvision.datasets
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import tqdm
from prediction.models.cv import cnn2d
from utils import utils, nn_utils


def run_model(model, train_dataset_loader, test_dataset_loader, loss, n_epochs, model_name, mode):
    tbw = SummaryWriter()
    class_weights = utils.get_class_weights(train_dataset_loader).to(nn_utils.get_device())
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=1e-4,
        epochs=n_epochs,
        steps_per_epoch=len(train_dataset_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0)
    model.train_iter = 0
    model.test_iter = 0
    if mode == "train":
        # train the model only if set to train mode
        for e in range(n_epochs):
            model = run_epoch(model, train_dataset_loader, test_dataset_loader, criterion, optimizer,
                              lr_scheduler, tbw, model_name, e)

    return evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch=None, log_loss=False), model


def run_epoch(model, train_dataset_loader, test_dataset_loader, criterion, optimizer, lr_scheduler, tbw, model_name,
              epoch):
    # Training
    model.train()
    for _, record in enumerate(pbar := tqdm.tqdm(train_dataset_loader)):
        input, label = record

        optimizer.zero_grad()

        output = model(input)
        output = output.to(nn_utils.get_device())

        loss = criterion(output, label.long())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        model.train_iter += 1
        curr_lr = lr_scheduler.get_last_lr()[0]
        train_loss = loss.item()
        tbw.add_scalar(f"{model_name}/learning-rate", float(curr_lr), model.train_iter)
        tbw.add_scalar(f"{model_name}/training-loss", float(train_loss), model.train_iter)
        pbar.set_description(
            f"{model_name}/training-loss = {float(train_loss)}, model.n_iter={model.train_iter}, epoch={epoch + 1}")

    # Testing
    evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch, log_loss=True)
    return model


def evaluate_model(model, test_dataset_loader, criterion, tbw, model_name, epoch, log_loss=False):
    with torch.no_grad():
        model.eval()

        results = []
        for _, record in enumerate(pbar := tqdm.tqdm(test_dataset_loader)):
            input, label = record

            output = model(input)  # b x n_classes
            output = output.to(nn_utils.get_device())

            loss = criterion(output, label.long())
            val_loss = loss.item()
            model.test_iter += 1
            if log_loss:
                tbw.add_scalar(f"{model_name}/validation-loss", float(val_loss), model.test_iter)
                pbar.set_description(
                    f"{model_name}/validation-loss = {float(val_loss)}, model.n_iter={model.test_iter}, epoch={epoch + 1}")
            # to get probabilities of the output
            output = F.softmax(output, dim=-1)
            result_df = pd.DataFrame(output.cpu().numpy())
            result_df["y_true"] = label.cpu().numpy()
            results.append(result_df)
    return pd.concat(results, ignore_index=True)


def main():
    transform = transforms.Compose([[transforms.ToTensor()]])
    batch_size = 64
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_dataset_loader = torch.utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = torch.utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    model = cnn2d.get_cnn_model({
        "n_classes": 10,
        "N": 2,
        "n_filters": 3,
        "kernel_size": 3,
        "stride": 3,
        "img_size": 8
    })

    results_df, _ = run_model(model=model,
              train_dataset_loader=train_dataset_loader,
              test_dataset_loader=test_dataset_loader,
              loss="FocalLoss",
              n_epochs=10,
              model_name="CNN-CIFAR10",
              mode="train")

    results_df.to_csv("cnn_2l_test_cifar10.csv")


if __name__ == "__main__":
    main()