# Installation
To setup `mot-mmtrack`, run the following command:
```bash
conda create -n mot-mmtrack python=3.10
pip install torch torchvision
# or install via: conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install openmim
mim install -r requirements/mminstall.txt
pip install -v -e .
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

# Usage
Run the demo:
```bash
python demo/demo_mot_vis.py \
    ${PATH_TO_YOUR_CONFIG} \
    --input demo/demo.mp4 \
    --output outputs
```