import subprocess
for alt in ["Alt7","Alt8"]:
	# subprocess.call(r'C:\\Users\\hstern\\AppData\\Local\\Continuum\\anaconda3\\python.exe C:\\Users\\hstern\\Downloads\\PerceptualSimilarity-master\\test_dataset_model.py --dataset_mode 2afc --model net --net alex --use_gpu --batch_size 1 --alt {0}'.format(alt))
	subprocess.call(r'C:\Users\granotniv\AppData\Local\conda\conda\envs\my_root\python.exe C:\Users\granotniv\Downloads\PerceptualSimilarity-master\PerceptualSimilarity-master\test_dataset_model.py --dataset_mode 2afc --model net --net alex --use_gpu --alt {0}'.format(alt))