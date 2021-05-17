def make_custom_func(layer_number = 0, channel_number= 0, center = False):
    if center is False: 
        def custom_func(layer_outputs):
            loss = layer_outputs[layer_number][channel_number].mean()
            return -loss
    else:
        def custom_func(layer_outputs):
            output = layer_outputs[layer_number][channel_number]
            loss = output[output.shape[-2]//2, output.shape[-1]//2]
            return -loss
    return custom_func
