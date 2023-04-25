python demo/demo_mot_vis.py \
configs/mot/deepsort/deepsort_reid.py \
--input demo/demo.mp4 --output mot.mp4

python demo/batch_demo_mot.py \
configs/mot/deepsort/deepsort_pose.py \
--input demo/test3.mp4 --output mot.mp4 --batch-size=4

# python tools/test.py configs/mot/deepsort/deepsort_pose.py
