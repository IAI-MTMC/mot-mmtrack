# Installation
To setup `mot-mmtrack`, run the following command:
```bash
conda create -n mot-mmtrack python=3.10
pip install openmim
mim install -r requirements/readthedocs.txt
pip install -v -e .
```

# Usage
Run the demo:
```bash
python demo/batch_demo_mot.py \
    ${PATH_TO_YOUR_CONFIG} \
    --input demo/demo.mp4 \
    --output outputs/result.mp4
```
