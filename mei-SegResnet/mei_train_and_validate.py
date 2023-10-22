import os
import torch
import time
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from mei_utils import *
from mei_dataset import *
from mei_model import *

def train_and_validate(args, model, train_loader, val_loader,train_ds):
    device = torch.device(args.device)
    max_epochs=args.max_epochs
    roi_size=args.roi_size
    val_interval=args.val_interval
    VAL_AMP=args.VAL_AMP
    root_dir=args.root_dir

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_semitendinosus = []
    metric_values_gracilis = []
    total_start = time.time()

    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
 
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    print(f"val_inputs Shape: {val_inputs.shape}")
                    
                    val_outputs=model(inputs)
                    val_outputs = inference(val_inputs, roi_size, model, VAL_AMP)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate() # 2个class的metric
                
                metric_semitendinosus = metric_batch[0].item()
                metric_values_semitendinosus.append(metric_semitendinosus)
                
                metric_gracilis = metric_batch[1].item()
                metric_values_gracilis.append(metric_gracilis)
                
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    with open(os.path.join(root_dir, "train_logs.txt") , "a") as f:
                        f.write(f"saved new best metric model \n")

                    print("saved new best metric model")
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" semitendinosus: {metric_semitendinosus:.4f} gracilis: {metric_gracilis:.4f} "
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        with open(os.path.join(root_dir, "train_logs.txt") , "a") as f:
            f.write(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f},Time Consuming= {(time.time() - epoch_start):.4f}, Metric={metric:.4f},Metric_Semitendinosus={metric_semitendinosus:.4f} Metric_Gracilis={metric_gracilis:.4f},Best Mean Dice {best_metric:.4f} at Epoch:{best_metric_epoch}  \n")
            
    total_time = time.time() - total_start
    with open(os.path.join(root_dir, "train_logs.txt") , "a") as f:
            f.write(f"Total time {total_time} \n")

        