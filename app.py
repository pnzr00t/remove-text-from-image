from typing import Optional

import torch.cuda
from fastapi import FastAPI
from fastapi import HTTPException


# https://stackoverflow.com/questions/62359413/how-to-return-an-image-in-fastapi
from fastapi.responses import FileResponse # need additionaly install pip install aiofiles

from main import *

import gc

app = FastAPI()


###############################################
########Init models for APP####################
###############################################
# Init CraftNets
#request_api_count = 0
craft_args, craft_net, refiner_craft_net = init_craft_networks(refiner=False, debug=False)
edge_connect_model = init_edge_connect_model(mode=3)

# TODO: Remove late, if we don't need reinit models
# def reinit_models_if_needed():
#     global request_api_count
#     global craft_args
#     global craft_net
#     global refiner_craft_net
#     global edge_connect_model
#
#     request_api_count = request_api_count + 1
#     if request_api_count > 5:
#         request_api_count = 0
#
#         del craft_args
#         del craft_net
#         del refiner_craft_net
#         del edge_connect_model
#
#         # Free memory
#         gc.collect()
#         if torch.cuda.is_available() == True:
#             torch.cuda.empty_cache()
#
#         craft_args, craft_net, refiner_craft_net = init_craft_networks(refiner=False, debug=False)
#         edge_connect_model = init_edge_connect_model(mode=3)


###############################################
########/Init models for APP###################
###############################################

@app.get("/")
def read_root():
     return {"Hello": "World", "mega_class" : "mega_object.string1"}

# without async memory leaking
@app.get("/image_remover/")
async def read_image_remover(url: Optional[str] = None):
    if url is None:
        raise HTTPException(status_code=404, detail="URL not exist")

    input_image_url = url

    image_path = input_image_url
    image_file_name = os.path.basename(image_path)

    if not os.path.exists('./results_images'):
        os.makedirs('./results_images')

    if input_image_url is not None and input_image_url != '':

        source_image, output_image = pipeline(
            input_image_url,
            craft_args,  # Args create with craft nets, and use for text polygons detection
            craft_net,
            refiner_craft_net,  # refiner for more text detection accuracy, == none in this project
            edge_connect_model,  # Inpaint EdgeConnect model, "restore" image
            debug=False
        )

        if source_image is None or output_image is None:
            raise HTTPException(status_code=404, detail="URL not exist")

        # Save output image
        output_image_path = os.path.join("./results_images", image_file_name)
        output_image.save(output_image_path)


        print("Safe out image - ", output_image_path)
        #return FileResponse(output_image_path, media_type="image/jpg")
        return FileResponse(output_image_path)
    else:
        print('Provide an image url and try again.')
        raise HTTPException(status_code=404, detail="URL not exist")

