import os
import numpy as np
import random
from argparse import ArgumentParser

import torch
from torch import optim
from torch import nn
from tensorboardX import SummaryWriter

from config import HP
from model import FinalTrialNet
from dataset_hg import train_dataloader, eval_dataloader

"""

1. data
2. model
3. loss
4. optimizer
5. evaluate
6. training
7. save
"""
logger = SummaryWriter("./log")
# seed init:
random.seed(HP.seed)
np.random.seed(HP.seed)
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)

def evaluate(model_, dev_loader_, crit_):
    model_.eval() # set evaluation flag
    sum_loss = 0.
    with torch.no_grad():
        for batch in dev_loader_:
            image, y = batch
            pred = model_(image)
            loss = crit_(pred, y.to(HP.device))
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
    parser = ArgumentParser(description=" model training ")
    parser.add_argument(
        "--c",
        default=None,
        type=str,
        help="train from scratch or resume training"
    )
    arg = parser.parse_args()

    model = FinalTrialNet()
    model.to(HP.device)

    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizer
    opt = optim.Adam(model.parameters(), HP.init_lr)
    start_epoch, step =0 ,0

    if arg.c:
        checkpoint = torch.load(arg.c)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print("Resume from %s." % arg.c)
    else:
        print("Training from scratch ")

    model.train()
    for epoch in range(start_epoch, HP.epochs):
        print(" start epoch: %d, steps: %d " % (epoch, len(train_dataloader)))
        for batch in train_dataloader:
            image, y = batch
            opt.zero_grad()  # gradient clean
            pred = model(image)
            loss = criterion(pred, y.to(HP.device))  # class calc
            loss.backward()  # backward process
            opt.step()
            logger.add_scalar("Loss/Train", loss, step)
            if not step % HP.verbos_step:
                eval_loss = evaluate(model, eval_dataloader, criterion)
                logger.add_scalar("Loss/Dev", eval_loss, step)
            if not step % HP.save_step:
                model_path = "model_%d_%d.pth" % (epoch, step)
                save_checkpoints(model, epoch, opt, os.path.join("model_save", model_path))
            step += 1
            logger.flush()
            print("Epoch: [%d/%d], step: %d Train loss: %.5f, Dev Loss: %.5f"
                  % (epoch, HP.epochs, step, loss.item(), eval_loss))
    logger.close()

if __name__ == '__main__':
    train()


