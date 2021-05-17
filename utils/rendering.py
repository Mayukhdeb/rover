import streamlit as st

def run_render(dreamer,
            layers , 
            custom_func,
            width,
            height,
            iters,
            lr , 
            rotate_degrees,
            scale_max,
            scale_min,
            translate_x,
            translate_y,
            weight_decay,
            grad_clip):
    with st.spinner("running..."):

        image_param = dreamer.render(
            layers = layers, 
            custom_func= custom_func,
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
