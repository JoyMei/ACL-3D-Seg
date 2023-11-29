import argparse

def get_args():
    parser = argparse.ArgumentParser(description="ACL 3D Seg Args")
    parser.add_argument("--device", help="Name of the file to create", default="cuda:0")
    parser.add_argument("--data_dir", default='../../data-acl', help="Data directory")
    parser.add_argument("--root_dir", default="./V22", help="Root directory")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--num_val_files", type=int, default=4, help="Number of validation files")
    parser.add_argument("--roi_size", nargs='+',type=int, default=[16, 16, 8], help="ROI size") 
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
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
    
    return args