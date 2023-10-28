"""Training script"""

import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from dataset import WakeWordData, collate_fn
from model import WakeupModel_LSTM
from sklearn.metrics import classification_report
from tabulate import tabulate


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load model, optimizer, and scheduler states from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, model_params, notes=None):

    torch.save({
        "notes": notes,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)
    return acc


def test(test_loader, model, device, epoch):
    print("\n starting test for epoch %s"%epoch)
    accs = []
    preds = []
    labels = []

    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        total_acc = 0
        total_samples = 0

        for idx, (mfcc, label) in enumerate(test_loader):
            mfcc, label = mfcc.to(device), label.to(device)
            output = model(mfcc)
            pred = torch.sigmoid(output)
            acc = binary_accuracy(pred, label)
            preds += torch.flatten(torch.round(pred)).cpu()
            labels += torch.flatten(label).cpu()
            accs.append(acc)
            total_acc += acc * mfcc.size(0)
            total_samples += mfcc.size(0)
            print("Iter: {}/{}, accuracy: {}".format(idx, len(test_loader), acc), end="\r")
        
        overall_acc = total_acc / total_samples
        print("\nOverall Accuracy: {:.2f}%".format(overall_acc * 100))


    average_acc = sum(accs)/len(accs) 
    print('Average test Accuracy:', average_acc, "\n")
    report = classification_report(labels, preds, zero_division=0)
    print(report)
    return average_acc, report


def train(train_loader, model, optimizer, loss_fn, device, epoch):
    model.train()  # set the model to training mode

    print("\n starting train for epoch %s" % epoch)
    losses = []
    preds = []
    labels = []

    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mfcc)
        loss = loss_fn(torch.flatten(output), label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # get predictions and labels for report
        pred = torch.sigmoid(output)
        preds += torch.flatten(torch.round(pred)).cpu().tolist()
        labels += torch.flatten(label).cpu().tolist()

        print("epoch: {}, Iter: {}/{}, loss: {:.4f}".format(epoch, idx, len(train_loader), loss.item()), end="\r")

    avg_train_loss = sum(losses) / len(losses)
    acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
    print('\navg train loss:', avg_train_loss, "avg train acc", acc)
    report = classification_report(labels, preds, zero_division=1)  # or zero_division=1 if you prefer

    print(report)
    return acc, report


def main(args):
    sample_rate = 8000
    epochs = 100
    batch_size = 32
    eval_batch_size = 32
    lr = 1e-4
    model_name = "wakeup"
    save_checkpoint_path = None  # Replace with your path if needed
    train_data_json = None  # Replace with your path
    test_data_json = None  # Replace with your path
    hidden_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)

    train_dataset = WakeWordData(data_json=train_data_json, sample_rate=sample_rate, valid=False)
    test_dataset = WakeWordData(data_json=test_data_json, sample_rate=sample_rate, valid=True)
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn)
    
    test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=eval_batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    model_params = {
        "num_classes": 1, "feature_size": 40, "hidden_size": hidden_size,
        "num_layers": 1, "dropout" :0.1, "bidirectional": False
    }
    model = WakeupModel_LSTM(**model_params, device=device)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() # Binary Cross-Entropy loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_train_acc, best_train_report = 0, None
    best_test_acc, best_test_report = 0, None
    best_epoch = 0
    
    for epoch in range(epochs):
        print("\nstarting training with learning rate", optimizer.param_groups[0]['lr'])
        train_acc, train_report = train(train_loader, model, optimizer, loss_fn, device, epoch)
        test_acc, test_report = test(test_loader, model, device, epoch)

        # record best train and test
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # saves checkpoint if metrics are better than last
        if save_checkpoint_path and test_acc >= best_test_acc:
            # Create a string representing the current date and hour
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H")
            checkpoint_path = os.path.join(save_checkpoint_path, f"{model_name}_{current_datetime}.pt")
            print("found best checkpoint. saving model as", checkpoint_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, model_params,
                notes="train_acc: {}, test_acc: {}, epoch: {}".format(best_train_acc, best_test_acc, epoch),
            )
            best_train_report = train_report
            best_test_report = test_report
            best_epoch = epoch

        table = [["Train ACC", train_acc], ["Test ACC", test_acc],
                ["Best Train ACC", best_train_acc], ["Best Test ACC", best_test_acc],
                ["Best Epoch", best_epoch]]
        
        print(tabulate(table))
        scheduler.step(train_acc)

    print("Done Training...")
    print("Best Model Saved to", checkpoint_path)
    print("Best Epoch", best_epoch)
    print("\nTrain Report \n")
    print(best_train_report)
    print("\nTest Report\n")
    print(best_test_report)


if __name__ == "__main__":
    main()