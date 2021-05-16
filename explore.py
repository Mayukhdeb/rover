import streamlit as st

import torch 
import torchvision
from torch_dreams.dreamer import dreamer

st.set_page_config(
    page_title="Rover",
    page_icon=":mag",  # EP: how did they find a symbol?

)

st.title('Rover')
st.markdown(
    """Built with [torch-dreams](https://github.com/Mayukhdeb/torch-dreams)
    """
)

layers_to_use =  []

models_dict = {
    'resnet18':      torchvision.models.resnet18(pretrained =True),
    'resnet50':      torchvision.models.resnet50(pretrained =True),
    'resnet101':     torchvision.models.resnet101(pretrained =True),
    'googlenet':     torchvision.models.googlenet(pretrained =True),
    'inception_v3':  torchvision.models.inception_v3(pretrained =True),
    'deeplabv3_resnet50': torchvision.models.segmentation.deeplabv3_resnet50(pretrained= True)

}

col1, col2 = st.beta_columns(2)

model_name = col1.selectbox(
        "Pick a model",
        options = list(models_dict.keys())
    )


model = models_dict[model_name]

dreamy_boi = dreamer(model, device = 'cuda')

layers_dict = dict(model.named_modules())

all_buttons = []
count = 0


names = col2.multiselect(
        "Pick layer(s)",
        options = list(layers_dict.keys())
    )

with st.beta_expander(label = 'Show selected layers'):
    st.write('You selected:', names)

layers_to_use = []

for name in names:
    layers_to_use.append(layers_dict[name])


my_custom_func = None

default_str_custom_func = """
def custom_func(layer_outputs):
    loss = layer_outputs[0][20].mean()
    return -loss
"""

if st.checkbox('Channel objective'):

    from utils.default_custom_funcs import make_custom_func

    c1 , c2 = st.beta_columns(2)

    layer_index = c1.number_input(label =   "Enter layer index", value = 0)
    channel_index = c2.number_input(label = "Enter channel index", value = 9)

    my_custom_func = make_custom_func(layer_number= layer_index, channel_number= channel_index)

    
if st.checkbox('Write your own custom objective'):
    text = st.text_area(label = 'Write your objective function here', value = default_str_custom_func, height= 100)

    st.code(text)

    custom_func_file = open("utils/__custom_func__.py", "w")

    n = custom_func_file.write(text)
    custom_func_file.close()

    from utils.__custom_func__ import custom_func
    my_custom_func = custom_func


if len(layers_to_use) != 0:

    image_param = dreamy_boi.render(
        layers = layers_to_use, 
        custom_func= my_custom_func
    )
    

    st.image(image_param.to_hwc_tensor().numpy())
else:
    st.write("you did not pick any layer :(")
    pass

