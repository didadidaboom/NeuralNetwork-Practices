import os
from argparse import ArgumentParser
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import BanknoteClassificationModel
from config import HP
from dataset_banknote import train_loader, dev_loader

logger = SummaryWriter("./log")

# seed init: ensure reproducible result
torch.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)
torch.cuda.manual_seed(HP.seed)

def evaluate(model_, dev_loader_, crit_):
    model_.eval() # set evaluation flag
    sum_loss = 0.
    with torch.no_grad():
        for batch in dev_loader_:
            x, y = batch
            pred = model_(x)
            loss = crit_(pred, y)
            sum_loss += loss.item()
    model_.train()  # back to training
    return sum_loss/len(dev_loader_)

def save_checkpoints(model_, epoch_, opt, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        "model_state_dict": model_.state_dict(),
        "optimizer_state_dict": opt.state_dict()
    }
    torch.save(save_dict, checkpoint_path)

def train():
    parser = ArgumentParser(description=" Model Training")
    parser.add_argument(
        "--c",
        default=None,
        type=str,
        help="train from scratch or resume training"
    )
    arg = parser.parse_args()

    # new model instance
    model = BanknoteClassificationModel()
    model = model.to(HP.device)

    # loss function (loss.py)
    criterion = nn.CrossEntropyLoss()

    # optimizer
    opt = optim.Adam(model.parameters(), lr=HP.init_lr)

    start_epoch, step = 0, 0

    if arg.c:
        checkpoint = torch.load(arg.c)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print("Resume from %s."%arg.c)
    else:
        print("Training from scratch ")

    model.train()

    step = 0
    # main loop
    for epoch in range(start_epoch, HP.epochs):
        print(" start epoch: %d, steps: %d "% (epoch, len(train_loader)))
        for batch in train_loader:
            x, y = batch # load data
            opt.zero_grad() # gradient clean
            pred = model(x)
            loss = criterion(pred, y) # class calc

            loss.backward() # backward process
            opt.step()

            logger.add_scalar("Loss/Train", loss, step)

            if not step % HP.verbos_step:
                eval_loss = evaluate(model, dev_loader, criterion)
                logger.add_scalar("Loss/Dev", eval_loss, step)

            if not step % HP.save_step:
                model_path = "model_%d_%d.pth"%(epoch, step)
                save_checkpoints(model, epoch, opt, os.path.join("model_save", model_path))

            step += 1
            logger.flush()
            print("Epoch: [%d/%d], step: %d Train loss: %.5f, Dev Loss: %.5f"
                  %(epoch, HP.epochs, step, loss.item(), eval_loss))

    logger.close()

if __name__ == '__main__':
    train()

