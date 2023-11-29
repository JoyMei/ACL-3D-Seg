import os
import torch
from monai.inferers import sliding_window_inference

def create_file(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    else:
        print("File '{}' already exists.".format(foldername))

def inference(input, roi_size, model, VAL_AMP):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

if __name__ == "__main__":
    print('utils')
