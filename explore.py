import streamlit as st

import torch 
import torchvision
from torch_dreams.dreamer import dreamer
from utils.default_models import models_dict

st.set_page_config(
    page_title="Rover",
    page_icon=":mag",  # EP: how did they find a symbol?

)

st.title('Rover')
st.markdown(
    """Built with [torch-dreams](https://github.com/Mayukhdeb/torch-dreams)
    """
)


col1, col2 = st.beta_columns(2)

model_name = col1.selectbox(
        "Pick a model",
        options = list(models_dict.keys())
    )


model = models_dict[model_name]

dreamy_boi = dreamer(model, device = 'cuda')

layers_dict = dict(model.named_modules())

names = col2.multiselect(
        "Pick layer(s)",
        options = list(layers_dict.keys())
    )


layers_to_use = []

for name in names:
    layers_to_use.append(layers_dict[name])


my_custom_func = None

default_str_custom_func = """def custom_func(layer_outputs):
    loss = layer_outputs[0][20].mean()
    return -loss"""

with st.beta_expander(label = 'Modify args'):
    columns = st.beta_columns(4)

    width = columns[0].number_input('width', value = 256)
    height = columns[1].number_input('height', value = 256)
    iters = columns[2].number_input('iters', value = 120)
    lr = columns[3].number_input('lr', value = 9e-3)

    columns2 = st.beta_columns(4)
    rotate_degrees = columns2[0].number_input('rotate (deg)', value = 15)


    scale_max = columns2[1].number_input('scale max', value = 1.2)
    scale_min = columns2[2].number_input('scale min', value = 0.5)
    grad_clip = columns2[3].number_input('gradient clip', value= 1.)

    columns3 = st.beta_columns(3)

    translate_x = columns3[0].number_input('translate (x)', value = 0.2)
    translate_y = columns3[1].number_input('translate (y)', value= 0.2)
    weight_decay = columns3[2].number_input('weight decay', value= 1e-3)


if st.checkbox('Channel objective'):

    from utils.default_custom_funcs import make_custom_func

    c1 , c2 = st.beta_columns(2)

    layer_index = c1.number_input(label =   "Enter layer index", value = 0)
    channel_index = c2.number_input(label = "Enter channel index", value = 9)

    my_custom_func = make_custom_func(layer_number= layer_index, channel_number= channel_index, center= False)

    
if st.checkbox('Write your own custom objective'):
    text = st.text_area(label = 'Write your objective function here', value = default_str_custom_func, height= 100)

    st.code(text)

    custom_func_file = open("utils/__custom_func__.py", "w")

    n = custom_func_file.write(text)
    custom_func_file.close()

    from utils.__custom_func__ import custom_func
    my_custom_func = custom_func


if len(layers_to_use) != 0:

    if  st.button(label = 'Run'):
        with st.spinner("running..."):

            image_param = dreamy_boi.render(
                layers = layers_to_use, 
                custom_func= my_custom_func,
                width= width,
                height=height,
                iters= iters,
                lr = lr, 
                rotate_degrees= rotate_degrees,
                scale_max= scale_max,
                scale_min= scale_min,
                translate_x= translate_x,
                translate_y= translate_y,
                weight_decay= weight_decay,
                grad_clip= grad_clip
            )
        
        
        st.image(image_param.to_hwc_tensor().numpy())
else:
    # st.error("you did not pick any layer :(")
    pass


with st.beta_expander(label = 'Read more'):
    st.markdown(
        """
        ## Args
        * `width` (`int`, optional): Width of image to be optimized 
        * `height` (`int`, optional): Height of image to be optimized 
        * `iters` (`int`, optional): Number of iterations, higher -> stronger visualization
        * `lr` (`float`, optional): Learning rate
        * `rotate (deg)` (`int`, optional): Max rotation in default transforms
        * `scale max` (`float`, optional): Max image size factor. 
        * `scale min` (`float`, optional): Minimum image size factor. 
        * `translate (x)` (`float`, optional): Maximum translation factor in x direction
        * `translate (y)` (`float`, optional): Maximum translation factor in y direction
        * `weight decay` (`float`, optional): Weight decay for default optimizer. Helps prevent high frequency noise. 
        * `gradient clip` (`float`, optional): Maximum value of the norm of gradient. 
        """
    )

