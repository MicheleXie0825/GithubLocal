# Training an infinitely large dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import time
import global_setting
import os, copy, sys
from tqdm import tqdm
from config import get_config
from model_factory import get_model
from datasets_factory import get_dataloaders
from tools import get_optimizer, get_scheduler
from tools import get_dir_name, get_log_name

args, _ = get_config()
log_name = get_log_name(args)
log_dir = get_dir_name(args.out_dir,
                       log_name)
if os.path.exists(log_dir):
    print(f"Log dir {log_dir} already exists. It will be overwritten.")
else:
    os.makedirs(log_dir)

model = get_model(args.model_architecture).to(args.device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = get_optimizer(model,
                          args.optimizer,
                          args.learning_rate,
                          args.momentum,
                          args.weight_decay,
                          amsgrad=args.amsgrad)
scheduler = get_scheduler(optimizer, args.decay_step)

# model_save_path = os.path.join(log_dir, "model.pt")
# result_save_path = os.path.join(log_dir, "result.txt")

best_val_acc = 0.0
since = epoch_time_stamp = time.time()
# sys.maxsize: 取得整數的最大值
# for epoch in range(0, sys.maxsize):
for epoch in range(0, args.epoch):
    dataloaders = get_dataloaders(
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
    
    print('Epoch {}/{}'.format(epoch, args.epoch))
    # print('-' * 10)

    for phase in ['train', 'val']:
        if phase == "train":
            model.train()
            train_dataset = dataloaders[phase].dataset
            train_dataset.shared_batch_base_seed = epoch
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        curr = 0.

        pbar = tqdm(dataloaders[phase], ncols=120)

        for batch, data in enumerate(pbar):
            curr += data[0].size(0)
            if batch >= 1: pbar.set_postfix(loss=running_loss/curr,
                                            acc=float(running_corrects)/curr,
                                            epoch=epoch,
                                            phase=phase)
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()

            # forward
            # track grad if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


        # Logging
        epoch_loss = running_loss / curr
        epoch_acc = running_corrects.double()/ curr
        print('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))
        if phase == 'train':
            scheduler.step()
        elif phase == 'val':
            epoch_use_time = time.time() - epoch_time_stamp
            epoch_time_stamp = time.time()
            print('Epoch {:d} complete in {:.0f}m {:.0f}s'.format(epoch, epoch_use_time // 60, epoch_use_time % 60))
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model).cpu()
                # model_save_path = os.path.join(log_dir, "model_{:4f}_at_epoch_{:4d}.pt".format(best_val_acc, epoch))
                # result_save_path = os.path.join(log_dir, "result_{:4f}_at_epoch_{:4d}.txt".format(best_val_acc, epoch))
                model_save_path = os.path.join(log_dir, "model.pt")
                result_save_path = os.path.join(log_dir, "result.txt")
                print(f"Best model saved to {model_save_path}")
                torch.save(best_model_wts, model_save_path)
                logging_str = 'Best val Acc thus far: {:4f} at epoch {:4d}'.format(best_val_acc, epoch)
                print(logging_str)
                with open(result_save_path, "a+") as file:
                    file.write(logging_str + "\n")
    print()
# finally update the folder name with best_val_acc
