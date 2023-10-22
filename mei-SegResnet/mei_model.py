
from monai.networks.nets import SegResNet

def initialize_model(args):
    model = SegResNet(
        blocks_down=args.blocks_down,
        blocks_up=args.blocks_up,
        init_filters=args.init_filters,
        in_channels=1, 
        out_channels=2, 
        dropout_prob=args.dropout_prob,
    )
    return model

if __name__ == "__main__":
    print('model')