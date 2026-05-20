# README #

`ExoShadower` is an python pipeline that:
1. Can invert a light curve to recover the "shadow image" that produced it;
2. Can predict the class of returned shadow image as "anomalous" or "planet".

# Installing Dependencies #
Do the installations in this order else some errors might occur.

conda create -n genenv python=3.10 -y
conda activate genenv

pip install --upgrade pip
pip install tensorflow
conda install -c conda-forge ipython scipy notebook jupyterlab matplotlib scikit-learn scipy numpy emcee cython -y

conda install scikit-image
conda install -c conda-forge tqdm
conda install -c conda-forge astropy
conda install -c conda-forge optuna
conda install -c conda-forge lightkurve
conda install -c conda-forge batman-package
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge seaborn corner
pip install EightBitTransit

# Examples #

### How to run the pipeline # 

__Step 1: Update the file paths first in paths.py as per your system__

__Step 2: Generate Some Files required for further processing__
python clean_codes_v2/run_kepler_pipeline.py 

__Step 3: Generate training dataset__
python clean_codes_v2/run_train_pipeline.py --config-file demo_config.json  --train 0 --N 1 --Num 12  --fresh_run 0

__Step 4: Train the model using training dataset for a specified radius ratio bin__
python clean_codes_v2/run_train_pipeline.py --config-file demo_config.json  --train 1 

__Step 5: Test the trained model using a simulated test dataset__
python clean_codes_v2/run_inference_pipeline.py --config-file demo_config.json --Num 250 --test 1 --fresh_run 0

__Step 6: Use the trained model on real Kepler Light Curves__
python clean_codes_v2/run_inference_pipeline.py --config-file demo_config.json --test 0 --fresh_run 0

__Important Points:__

- For training dataset, avoid using __N = 99__ as it is reserved for storing test dataset.
- fresh_run = 0 is set so as to avoid repeating processes in case required files already exist.
- fresh_run = 1 generates everything from scratch e.g generating radius ratio grid, Bezier Shapes, Light Curves etc. Set it true only in case of testing.
- For the purpose of testing the pipeline choose small __Num__ s
- Run __Step 5 to Step 6__ once the __Step 4__ is completed

# ExoShadower #

The pipeline generates 2D Bezier shapes (38x38) and corresponding limb darkened light curves using the [EightBitTransit](https://github.com/esandford/EightBitTransit) package. A convolutional neural network model is trained on the simulated light curves to predict the 2D projected transiting shape (a.k.a Shadow) of the transiting system. The trained model then can be used on real light curves to predict the shadow of the transitor and find anomalous (non-circular) systems. 

__Details of the Dataset__

# 1. Shapes:
   - Random Bezier Shapes (binary images) are generated with size 38x38
   - Binary Images: 1 = opaque pixel blocking star light
                    0 = transparent pixel 
# 2. Light Curves:
    - We select targets in the Kepler KOI cumulative table with SNR >=50 (N=711). For these we get their $Rp/Rs$ and the Limb darkening coefficients.
    - We find the tight ant-correlation b/w the LDC coefficients [a,b]. We approximate this relation as stright line.
    - We generate many (~7500) LDC grid points in addition to including the  value for 711 sources.
    - We then split the Kepler dataset into radius ratio bins and store
   Preprocessing:
   1. EightBit generated light curve is interpolated to 120 points and saved
   2. Noise is added using error from Kepler phase folded data using __SNR = [100,500]__
   3. 
