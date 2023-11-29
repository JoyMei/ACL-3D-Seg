from mei_dataset import dataset
import glob
import os

def load_data(args):
    train_images = sorted(glob.glob(os.path.join(args.data_dir, "imagesTR", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_dir, "labelsTR", "*.nii.gz")))
    
    train_loader, val_loader, train_ds, val_ds = dataset(
        args.num_val_files, args.roi_size, args.batch_size,
        args.cache_rate, args.num_workers, train_images, train_labels
    )
    return train_loader, val_loader, train_ds, val_ds

