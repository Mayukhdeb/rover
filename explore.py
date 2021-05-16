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
    'inception_v3':  torchvision.models.inception_v3(pretrained =True),
    'resnet18':      torchvision.models.resnet18(pretrained =True),
    'resnet50':      torchvision.models.resnet50(pretrained =True),
    'resnet101':     torchvision.models.resnet101(pretrained =True),
    'googlenet':     torchvision.models.googlenet(pretrained =True),
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

with st.beta_expander(label = 'Modify args'):
    columns = st.beta_columns(4)

    columns[0].number_input('width', value = 512)
    columns[1].number_input('height', value = 256)
    columns[2].number_input('iters', value = 120)
    columns[3].number_input('lr', value = 9e-3)

    columns2 = st.beta_columns(4)
    columns2[0].number_input('rotate (deg)', value = 15)


    columns2[1].number_input('scale max', value = 1.2)
    columns2[2].number_input('scale min', value = 0.5)
    columns2[3].number_input('gradient clip', value= 1.)

    columns3 = st.beta_columns(3)

    columns3[0].number_input('translate_y', value = 0.2)
    columns3[1].number_input('translate (x)', value= 0.2)



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
    
    _, c, _ = st.beta_columns(3)
    c.image(image_param.to_hwc_tensor().numpy())
else:
    # st.error("you did not pick any layer :(")
    pass

