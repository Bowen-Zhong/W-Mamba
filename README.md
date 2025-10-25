# W-Mamba
# medical-image-fusion
This is the official implementation of the W-MambaFuse model proposed in the paper ("W-MambaFuse: A Wavelet Decomposition and Adaptive State-Space Modeling Approach for Anatomical and Functional Image Fusion") with Pytorch.


# Usage  

## Training  

### 1. Requirements
conda create -n WD-HSSF python=3.8

conda activate WD-HSSF

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 

pip install packaging pip install timm==0.4.12

pip install pytest chardet yacs termcolor

pip install submitit tensorboardX

pip install triton==2.0.0

pip install causal_conv1d==1.0.0 # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install mamba_ssm==1.0.1 # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs


### 2. Data Preparation  
Download the Harvard-Medical-Image-Fusion (HMIF) dataset from [this Baidu Netdisk link](https://pan.baidu.com/s/1qPi8Zu3nMtv0Pzry9dL8xQ?pwd=1z3x)  
Extraction code: `1z3x`  

After downloading, place the folder as `./HMIFDatasets/`.  


### 3. WD-HSSF Training  
Modify the type and settings of the fusion task in `args_setting.py`.  

Run  
```bash
python train.py
```  
and the trained model is available in `./modelsave/`.  

## Testing  

### 1.Testing  
Please put the model in the folder `./model_pre

Run  
```bash
python test.py
```  
