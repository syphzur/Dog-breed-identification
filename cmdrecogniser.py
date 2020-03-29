#%%
from fastai.vision import *

from pathlib import Path
import sys
import os.path
import imghdr

if len(sys.argv) > 1:
	scriptPath = Path(os.path.dirname(os.path.realpath(__file__)))

	modelPath = Path(str(scriptPath.parent) + '\\stanford-dogs-dataset-traintest\\cropped') 
    
	data = ImageDataBunch.from_folder(modelPath, train='train', valid_pct=0.2, size=224, num_workers=0)
	data.normalize(imagenet_stats)
	learner = cnn_learner(data, models.resnet34)

	learner.load("resnet34-fit5-stage2")

	imgPath = Path(sys.argv[1])

	if str(imghdr.what(imgPath)) != "None":
		img = open_image(imgPath, size=224)
		pred_class, pred_idx, outputs = learner.predict(img)
		print(pred_class) 
	else:
		print("Not image") 
else:
	print("No filename") 

# %%
