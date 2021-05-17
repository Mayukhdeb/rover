from rover import core
from rover.default_models import models_dict

## add in your custom model here if needed
# import torchvision.models as models 
# model = models.resnet34(pretrained= True)  ## or any other model

# models_dict = {
#     'my model': model,
# }

core.run(
    models_dict = models_dict
)

# core.run(
#     models_dict = models_dict
# )