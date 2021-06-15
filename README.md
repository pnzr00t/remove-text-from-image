# remove-text-from-image with Python from scratch

## How does it work?
Steps:
1. Take original image;
2. Detect words rectangles in image (with CRAFT-pytorch image);
3. Create words mask;
4. Delete text with Automated objects removal inpainter 

<!-- ![Origin image](https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/origin.png?raw=true) -->
<!-- ![Mask image](https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/mask.png?raw=true) -->
<!-- ![Out image](https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/celeba.png?raw=true) -->

<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/origin.png?raw=true" width="30%"></img> 
<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/mask.png?raw=true" width="30%"></img> 
<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/celeba.png?raw=true" width="30%"></img>


## Possible improvement:
1. Create more accurasy mask. For example create mask from words symbols, not from word rectangles (Create and train autoencoder by symbols);
2. Play with function create_craft_args - change args.text_threshold, args.low_text, args.link_threshold - it's can give better result;
3. Autotune edge-connect models;
4. Try another pretrained edge-connect models (psv/celeba/places2);
5. ...

## Installation

### Installation on *nix system:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/remove-text-from-image` (current repository URL);
3. Run command `cd ./remove-text-from-image/` (cloned folder);
4. Run command `bash ./install_project.sh` (Downloading libs, and models);
5. Run command `pip install -r ./requirements.txt`;
6. Run main.py script, you can chage original image URL in `function test_remover_func():`. Output image will save in local folder `./results_images`.


### Run in goole colab:
1. Go https://colab.research.google.com/ ;
2. File->Open notebook->Git Hub;
3. Copy and paste URL for current repository;
4. Chose colab file;
5. Run all cells;
6. Copy and original image URL in special cell with "input_image_url" parameter.

## More examples:
<p align="center">
	<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/tom_cat/origin.png?raw=true" width="30%"></img> 
	<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/tom_cat/mask.png?raw=true" width="30%"></img> 
	<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/tom_cat/out.png?raw=true" width="30%"></img>
</p>
<p align="center">
	<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/truck/origin.png?raw=true" width="30%"></img> 
	<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/truck/mask.png?raw=true" width="30%"></img> 
	<img src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/truck/out.png?raw=true" width="30%"></img>
</p>

## Different edge-connect models
Celeba/Places/PSV(Paris Street View)
<p align="center">
	<img alt="Celeba" src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/celeba.png?raw=true" width="30%"></img> 
	<img alt="Places" src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/places.png?raw=true" width="30%"></img> 
	<img alt="PSV Model" src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/psv.png?raw=true" width="30%"></img>
</p>

### Additional info
Create as part of dlschool.org project.
``` Deep Learning School -- organization supported by PSAMI MIPT and Lab of Innovation (MIPT). ```

#### MIPT links:
  * [MIPT official site](https://mipt.ru);
  * [MIPT Stepik DLS course](https://stepik.org/course/65388);
  * [Deep learning school](https://dlschool.org)

#### Based on 
 * [CRAFT text detecting](https://github.com/clovaai/CRAFT-pytorch.git);
 * [Automated objects removal inpainter](https://github.com/sujaykhandekar/Automated-objects-removal-inpainter);
 * [Generative Image Inpainting with Adversarial Edge Learning](https://github.com/knazeri/edge-connect) as part of Automated objects removal inpainter
 
#### Special thanks
 * [Artem Chumachenko - project MIPT curator](https://t.me/artek_chumak)