#%%
from fastai.vision import *
from pathlib import Path

scriptPath = Path(os.path.dirname(os.path.realpath(__file__)))
modelPath = Path(str(scriptPath.parent) + '\\stanford-dogs-dataset-traintest\\cropped')
np.random.seed(2)
data = ImageDataBunch.from_folder(modelPath, train='train', size=224, num_workers=0, bs=16)
data.normalize(imagenet_stats)
data.show_batch(rows=5, figsize=(25,25))
learner = cnn_learner(data, models.resnet34,  metrics=[accuracy, error_rate])
learner.fit(5)
learner.save("resnet34-fit5")

learner.lr_find()
learner.recorder.plot()

#learner = cnn_learner(data, models.resnet50,  metrics=[accuracy, error_rate])
#learner.fit(10)
#learner.save("test50")

# %%
