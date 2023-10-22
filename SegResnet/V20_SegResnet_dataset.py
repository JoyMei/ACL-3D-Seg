import torch
from monai.data import CacheDataset, DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
    EnsureChannelFirstd,
)

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
def dataset(num_val_files,roi_size,batch_size,cache_rate,num_workers,train_images,train_labels):
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[roi_size[0],roi_size[1],roi_size[2]], random_size=False),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )    
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandSpatialCropd(keys=["image", "label"], roi_size=[roi_size[0],roi_size[1],roi_size[2]], random_size=False),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        ]
    )

    data_dicts = [{"image": image_name, "label": label_name}
                for image_name, label_name in zip(train_images, train_labels)]

    train_files, val_files = data_dicts[:-num_val_files], data_dicts[-num_val_files:]

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transform,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    val_ds = CacheDataset(data=val_files, transform=val_transform,
                        cache_rate=cache_rate, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_loader,val_loader,train_ds,val_ds

if __name__ == "__main__":
    print('dataset')