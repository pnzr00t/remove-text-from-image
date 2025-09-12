# Remove Text From Image With Python From Scratch

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
5. Try change `args.model` value in `load_object_remover_config`;
6. ...

# Installation

## Install old version python (need for toch==1.4.0) on *nix
0. install `sudo apt-get install libffi-dev` or you see error `ModuleNotFoundError: No module named '_ctypes'` when start install requerements
1. `sudo apt-get install libssl-dev openssl`
2. `wget https://www.python.org/ftp/python/3.7.6/Python-3.7.6.tgz`
3. `tar xzvf Python-3.7.6.tgz`
4. `cd Python-3.7.6`
5. `./configure`
6. `make`
7. `sudo make install`

To ensure write print
`python3.7 -V`
Will print you
`Python 3.7.6`

## Install and activate venv with old python version
1. Install venv `python3.7 -m venv ~/venv-3.7-remove-text-from-image`
2. Activate venv `source ~/venv-3.7-remove-text-from-image/bin/activate`
3. Install `pip install --upgrade pip setuptools wheel` - or you get error when try install opencv (requirements.txt)

After that you may install requerements from instruction below

For deactivate venv print
`deactivate`

### Installation on *nix system:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/remove-text-from-image` (current repository URL);
3. Run command `cd ./remove-text-from-image/` (cloned folder);
4. Run command `bash ./install_project.sh` (Downloading libs, and models);
5. Run command `pip install -r ./requirements.txt`;
6. Run main.py script, you can chage original image URL in `function test_remover_func():`. Output image will save in local folder `./results_images`.

### Installation and run FastAPI service:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/remove-text-from-image` (current repository URL);
3. Run command `cd ./remove-text-from-image/` (cloned folder);
4. Run command `bash ./install_project.sh` (Downloading libs, and models);
5. Run command `pip install -r ./requirements.txt`;
6. Run command `pip install -r ./requirements-fast-api.txt` (modules for FastAPI service);
7. Run command for start up FastAPI service `uvicorn app:app`;
8. Remove text from image by HTTP request `http://127.0.0.1:8000/image_remover/?url=https://img-9gag-fun.9cache.com/photo/axMNd31_460s.jpg` (IP and port will print in console when you start up service *step 7*. url= -- URL to original image).

### Installation and run FastAPI service with gunicorn:
1. Open console;
2. Run command `git clone https://github.com/pnzr00t/remove-text-from-image` (current repository URL);
3. Run command `cd ./remove-text-from-image/` (cloned folder);
4. Run command `bash ./install_project.sh` (Downloading libs, and models);
5. Run command `pip install -r ./requirements.txt`;
6. Run command `pip install -r ./requirements-fast-api.txt` (modules for FastAPI service);
7. Run command for start up FastAPI service `gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app --timeout 600 --max-requests 5`;
8. Remove text from image by HTTP request `http://127.0.0.1:8000/image_remover/?url=https://img-9gag-fun.9cache.com/photo/axMNd31_460s.jpg` (IP and port will print in console when you start up service *step 7*. url= -- URL to original image).

Note: FastAPI with unicorn "eat" a lot of memory and have memory leak, thats why you can use gunicorn service, witch will restart and clean memory every `--max-requests COUNT_REQUEST`

### Run in google colab:
1. Go https://colab.research.google.com/ ;
2. File->Open notebook->Git Hub;
3. Copy and paste URL for current repository;
4. Chose colab file;
5. Run all cells;
6. Copy and paste original image URL in special cell with "input_image_url" parameter.

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

### Different edge-connect models
Celeba/Places/PSV(Paris Street View)
<p align="center">
	<img alt="Celeba" src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/celeba.png?raw=true" width="30%"></img> 
	<img alt="Places" src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/places.png?raw=true" width="30%"></img> 
	<img alt="PSV Model" src="https://github.com/pnzr00t/remove-text-from-image/blob/main/ImagesExamples/joker/psv.png?raw=true" width="30%"></img>
</p>

## FAQ
### A lot of memory usage
This script use a lot of memory, so i recommended restart you service or use gunicorn version ([Install gunicorn service and run](#installation-and-run-fastapi-service-with-gunicorn))

### Uvicorn service memory leaks
Use gunicorn version and set properly  `--max-requests COUNT` COUNT parameter (according of you RAM capacity)

### Can script working faster
Yes, but you need torch lib with GPU, script automatically detecting you GPU device and run on them. (Additional info: At this time it must be NVIDIA GPU with cuda drivers and >= 4GB RAM)


## Additional info and links
Create as part of dlschool.org project.
``` Deep Learning School -- organization supported by PSAMI MIPT and Lab of Innovation (MIPT). ```

### MIPT links:
  * [MIPT official site](https://mipt.ru);
  * [MIPT Stepik DLS course](https://stepik.org/course/65388);
  * [Deep learning school](https://dlschool.org)

### Based on 
 * [CRAFT text detecting](https://github.com/clovaai/CRAFT-pytorch.git);
 * [Automated objects removal inpainter](https://github.com/sujaykhandekar/Automated-objects-removal-inpainter);
 * [Generative Image Inpainting with Adversarial Edge Learning](https://github.com/knazeri/edge-connect) as part of Automated objects removal inpainter
 
### Special thanks
 * [Artem Chumachenko - project MIPT curator](https://t.me/artek_chumak);
 * [Viktor Savin - This project teammate](https://github.com/vsavin)

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International.](https://creativecommons.org/licenses/by-nc/4.0/)

Except where otherwise noted, this content is published under a [CC BY-NC](https://github.com/knazeri/edge-connect) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.
