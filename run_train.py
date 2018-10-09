import os
import subprocess
for net in ["alex","vgg","squeeze"]:
	subprocess.call(r'C:\Users\granotniv\AppData\Local\conda\conda\envs\my_root\python.exe C:\Users\granotniv\Downloads\PerceptualSimilarity-master\PerceptualSimilarity-master\train.py --use_gpu --net {0} --name ${0}_0'.format(net))
	subprocess.call(r'C:\Users\granotniv\AppData\Local\conda\conda\envs\my_root\python.exe C:\Users\granotniv\Downloads\PerceptualSimilarity-master\PerceptualSimilarity-master\test_dataset_model.py --use_gpu --net ${0} --model_path ./checkpoints/${0}_0/latest_net_.pth'.format(net))
