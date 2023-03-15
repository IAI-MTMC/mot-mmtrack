# Installation
To setup `mot-mmtrack`, run the following command:
```bash
conda create -n mot-mmtrack python=3.10
pip install torch torchvision
# or install with conda `conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia`
pip install openmim
mim install -r requirements/packages.txt
pip install -v -e .
```

# Prepare dataset
To extract images from videos, run:
```bash
python tools/dataset_converters/aicity/vid2imgs.py
    ${DATA_DIR}
    --fps 5
    --num-workers ${NUM_WORKERS} # default max
```

Run this script to get the annnotations in `MOTChallenge` format:
```bash
python tools/dataset_converters/aicity/aicity2coco.py \
    ${DATA_DIR} \
    --min-box-height 130 \
    --min-box-width 30
```

# Usage
Run the demo:
```bash
python demo/batch_demo_mot.py \
    ${PATH_TO_YOUR_CONFIG} \
    --input demo/test1.mp4 \
    --output outputs/result.mp4 \
    --batch-size 16
```
