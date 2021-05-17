
import streamlit as st 

@st.cache
def show_arg_defs():
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