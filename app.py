from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException

# https://stackoverflow.com/questions/62359413/how-to-return-an-image-in-fastapi
from fastapi.responses import FileResponse # need additionaly install pip install aiofiles

from main import *

app = FastAPI()


###############################################
########Init models for APP####################
###############################################
# Init CraftNets
craft_args, craft_net, refiner_craft_net = init_craft_networks(refiner=False, debug=False)
edge_connect_model = init_edge_connect_model(mode=3)
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
        # source_image, output_image = pipeline(input_image_url, model_isr, model_translator, tokenizer_translator, font, debug=False)

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
        #print('Provide an image url and try again.')
        raise HTTPException(status_code=404, detail="URL not exist")

