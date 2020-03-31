#%%
from fastai.vision import *

from pathlib import Path
import sys
import os.path
import imghdr

def top5outputs(outputs):    
    sortedOutputs = outputs.argsort(descending=True)
    sortedOutputs = sortedOutputs[:5]    
    return sortedOutputs
	
def top5outputs_labels(outputs, classes):
    top_5 = top5outputs(outputs)
    labels = []
    confidence = []
    for i in top_5:
        x = classes[i][10:]
        p = str(float("{0:.4f}".format(outputs[i]*100))) + "%"
        labels.append(x)
        confidence.append(p)
    return labels, confidence

if len(sys.argv) > 1: # if more than 0 args passed
	scriptPath = Path(os.path.dirname(os.path.realpath(__file__)))
	datasetPath = Path(str(scriptPath.parent) + '\\stanford-dogs-dataset-traintest\\cropped') # path to dataset

	data = ImageDataBunch.from_folder(datasetPath, train='train', valid_pct=0.2, size=224, num_workers=0)
	data.normalize(imagenet_stats)
	

	learner = cnn_learner(data, models.resnet34)
	learner.load("resnet34-fit5-stage2") # loading neural network model

	imgPath = Path(sys.argv[1])
	if str(imghdr.what(imgPath)) != "None": # if is an image
		img = open_image(imgPath)
		img.resize(224).refresh() # resize image to 224x224

		pred_class, pred_idx, outputs = learner.predict(img)
		top5_predictions, top5_confidence = top5outputs_labels(outputs, data.classes)
		for i in range(5):
			print(top5_confidence[i], top5_predictions[i])
	else:
		print("Not image.") 
else:
	print("Please pass filename as first argument.") 

# %%
