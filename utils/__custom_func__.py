
def custom_func(layer_outputs):
    loss = layer_outputs[0][20].mean()
    return -loss
