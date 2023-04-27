python demo/batch_demo_mot.py \
configs/mot/deepsort/deepsort_pose.py \
--input demo/vis/test3.mp4 --output outputs/mot.mp4 \
--batch-size=4 --vis-pose=True

# python tools/test.py configs/mot/deepsort/deepsort_pose.py

python demo/batch_demo_mot.py \
configs/mot/deepsort/test_config.py \
--input demo/demo.mp4 --output outputs/mot.mp4 \
--batch-size=4