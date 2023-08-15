#conda create -n llamax python=3.10
source deactivate
conda activate llamax
cd src
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install transformers==4.29.2
cd ..
pip install -r requirements.txt