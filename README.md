# Installation
To setup `mot-mmtrack`, run the following command:
```bash
conda create -n mot-mmtrack python=3.10 -y
conda activate mot-mmtrack

pip install torch torchvision
pip install -U openmim
mim install mmcv-full mmdet

pip install -v -e .
```

# Usage
Run the demo:
```bash
python demo/demo_mot_vis.py \
    configs/mot/deepsort/my_config.py \
    --input demo/demo.mp4 \
    --output outputs
```