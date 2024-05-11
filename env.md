## Preparing environment
1. Create a virtual environment
```shell 
conda create --name egoins python==3.9.12
conda activate egoins
```

2. Install [torch 2.0.1]((https://pytorch.org/)) and torchvision.
```shell 
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

3. Install [mmcv-full==1.3.14](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for logging
```shell 
pip install -U openmim
mim install mmcv-full==1.3.14
```

4. Install [clip](https://github.com/openai/CLIP.git)
```shell 
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

5. Install other dependencies
```shell
pip install -r requirements.txt
```