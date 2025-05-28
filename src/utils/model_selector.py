import sys

def select_model(arg_obj):
    model_type = arg_obj.model_type.lower()
    channels = arg_obj.channels
    input_channels = len(channels)
    dropout = float(arg_obj.dropout)

    if model_type == 'rpnet':
        from models.RPNet import RPNet
        model = RPNet(input_channels=input_channels, drop_p=dropout)
    elif model_type == 'physnet':
        from models.PhysNet import PhysNet
        model = PhysNet(input_channels=input_channels, drop_p=dropout, 
                        early_mask=arg_obj.early_mask, mid_mask=arg_obj.mid_mask)
    else:
        print('Could not find model specified.')
        sys.exit(-1)

    return model
