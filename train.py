from customModel import CNNClassifier, save_model
from utils import accuracy, load_data
import torch
from torch.utils.tensorboard import SummaryWriter


def train(args):
    from os import path
    model = CNNClassifier()
    tb = SummaryWriter()

    # --- Initializations ---
    model = CNNClassifier()

    # Potential GPU optimization.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lrate)
    criterion = torch.nn.MSELoss()
    train_loader = load_data("/home/asf170004/data/customData/train")
    validation_loader = load_data("/home/asf170004/data/customData/valid")

    # --- SGD Iterations ---
    for epoch in range(args.epochs):
        print("Starting Epoch: ", epoch)
        # Per epoch train loop.
        model.train()
        for _, (rgb_input, sem_input, yhat) in enumerate(train_loader):
            optimizer.zero_grad()
            sem_input.to(device)
            ypred = model(sem_input.cuda())
            loss = criterion(ypred, yhat.cuda())
            loss.backward()
            optimizer.step()

            # Record training loss and accuracy
            tb.add_scalar("Train Loss", loss, epoch)
            tb.add_scalar("Train Accuracy", accuracy(ypred, yhat), epoch)

            # tb.add_scalar("Steer Accuracy", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Throttle Accuracy", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Brake Accuracy", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Steer RMSE", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Throttle RMSE", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Brake RMSE", accuracy(ypred, yhat), epoch)

        # After each train epoch, do validation before starting next train epoch.
        model.eval()
        for _, (rgb_input, sem_input, yhat) in enumerate(validation_loader):
            with torch.no_grad():
                sem_input.to(device)
                ypred = model(sem_input.cuda())
                loss = criterion(ypred, yhat.cuda())

            # Record validation loss and accuracy
            tb.add_scalar("Validation Loss", loss, epoch)
            tb.add_scalar("Validation Accuracy", accuracy(ypred, yhat), epoch)

            # tb.add_scalar("Steer Accuracy", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Throttle Accuracy", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Brake Accuracy", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Steer RMSE", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Throttle RMSE", accuracy(ypred, yhat), epoch)
            # tb.add_scalar("Brake RMSE", accuracy(ypred, yhat), epoch)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lrate', type=float, default=0.0001)
    parser.add_argument('-e', '--epochs', type=int, default=1000)

    args = parser.parse_args()
    train(args)
