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

""" imgPath = Path(str(modelPath) + "\\test\\n02085620-Chihuahua")
for img in os.listdir(imgPath):
    img = open_image(imgPath)
    pred_class, pred_idx, outputs = learner.predict(img)
    print(pred_class) """
img = open_image(r"D:\POLITECHNIKA\sem6\biai\Dogs\stanford-dogs-dataset-traintest\cropped\test\n02115641-dingo\n02115641_10506.jpg")
pred_class, pred_idx, outputs = learner.predict(img)
print(pred_class) 
# %%
