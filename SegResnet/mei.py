import torch
from mei_utils import *
from mei_args import get_args
from mei_model import *
from mei_loaddata import load_data
from mei_train_and_validate import train_and_validate

def main():
    args = get_args()
    device = torch.device(args.device)
    create_file(args.root_dir)
    model=initialize_model(args).to(device)
    train_loader, val_loader, train_ds, val_ds = load_data(args)
    train_and_validate(args, model, train_loader, val_loader,train_ds)

if __name__ == "__main__":
    main()