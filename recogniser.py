#%%
from fastai.vision import *

import os
from pathlib import Path

scriptPath = Path(os.path.dirname(os.path.realpath(__file__)))

modelPath = Path(str(scriptPath.parent) + '\\stanford-dogs-dataset-traintest\\cropped')
    
data = ImageDataBunch.from_folder(modelPath, train='train', valid_pct=0.2, size=224, num_workers=0)
data.normalize(imagenet_stats)
learner = cnn_learner(data, models.resnet34,  metrics=[accuracy, error_rate])

learner.load("resnet34-fit5-stage2")

imgPath = Path(str(modelPath) + "\\test")
correct = 0
wrong = 0
for x in os.listdir(imgPath):

	folfer = Path(str(imgPath) + '\\' + x)
	
	for y in os.listdir(folfer):
		img = open_image(str(folfer) + '\\' + y)
		pred_class, pred_idx, outputs = learner.predict(img)
		print(pred_class) 
		if x[0:9]==y[0:9]:
			correct += 1
		else:
			wrong +=1
		
print("Correct", correct, "Wrong", wrong) 
		
# %%
