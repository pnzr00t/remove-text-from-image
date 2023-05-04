git clone https://github.com/clovaai/CRAFT-pytorch.git
git clone https://github.com/sujaykhandekar/Automated-objects-removal-inpainter.git


# create dir for models
mkdir weights
# craft main
wget -O weights/craft_mlt_25k.pth https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ&export=download
# craft refiner
# !!!!!!!!!!Есть вероятность что она не нужна!!!!!!!!!!!!!!!
wget -O weights/craft_refiner_CTW1500.pth https://drive.google.com/uc?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO&export=download
# Automated-objects-removal-inpainter
bash ./Automated-objects-removal-inpainter/scripts/download_model.sh
#Copy edge model
cp ./checkpoints/celeba/* ./checkpoints/
#%cp ./checkpoints/psv/* ./checkpoints/
#%cp ./checkpoints/places2/* ./checkpoints/

# Result images folders, probably it can, be not necessary
mkdir ./results_images/

echo ">>>>>I install all.<<<<<"
echo "Don't forget run command for installing libs \"pip install -r requirements.txt\""
echo "And if you want use it as sevice, run comand after previous command \"pip install -r requirements-fast-api.txt\""
