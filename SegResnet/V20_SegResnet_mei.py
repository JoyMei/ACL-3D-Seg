import argparse
import os
import glob
import torch
import time
from monai.networks.nets import SegResNet
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.losses import DiceCELoss

from V20_SegResnet_utils import *
from V20_SegResnet_dataset import *

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

def main():
    parser = argparse.ArgumentParser(description="Your script description.")
    parser = argparse.ArgumentParser(description="Create a file with a specified name.")
    parser.add_argument("--device", help="Name of the file to create", default="cuda:0")
    parser.add_argument("--data_dir", default='./data-acl', help="Data directory")
    parser.add_argument("--root_dir", default="./V20", help="Root directory")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_val_files", type=int, default=4, help="Number of validation files")
    parser.add_argument("--roi_size", nargs='+',type=int, default=[144, 144, 64], help="ROI size")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval")
    parser.add_argument("--VAL_AMP", action="store_true", help="Enable VAL_AMP")
    parser.add_argument("--cache_rate", type=float, default=0.0, help="Data cache rate")
    
    parser.add_argument("--blocks_down", type=int, nargs="+", default=[1, 2, 2, 4], help="blocks_down")
    parser.add_argument("--blocks_up", type=int, nargs="+", default=[1, 1, 1], help="blocks_up")
    parser.add_argument("--init_filters", type=int, default=64, help="init_filters")
    parser.add_argument("--in_channels", type=int, default=1, help="in_channels")
    parser.add_argument("--out_channels", type=int, default=2, help="out_channels")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout_prob")

    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir=args.data_dir
    max_epochs=args.max_epochs
    roi_size=args.roi_size
    batch_size=args.batch_size
    val_interval=args.val_interval
    VAL_AMP=args.VAL_AMP
    root_dir=args.root_dir
    num_workers=args.num_workers
    cache_rate=args.cache_rate
    num_val_files=args.num_val_files

    create_file(root_dir)
    
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_semitendinosus = []
    metric_values_gracilis = []
    total_start = time.time()

    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.cuda.amp.GradScaler()

    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTR", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTR", "*.nii.gz")))
    model = SegResNet(
        blocks_down=args.blocks_down,
        blocks_up=args.blocks_up,
        init_filters=args.init_filters,
        in_channels=1, 
        out_channels=2, 
        dropout_prob=args.dropout_prob,
    ).to(device)
    
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    train_loader,val_loader,train_ds,val_ds=dataset(num_val_files,roi_size,batch_size,cache_rate,num_workers,train_images,train_labels)
    
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
                metric_batch = dice_metric_batch.aggregate() 
                
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

if __name__ == "__main__":
    main()