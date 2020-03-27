#%%
from fastai.vision import *
from pathlib import Path

scriptPath = Path(os.path.dirname(os.path.realpath(__file__)))
modelPath = Path(str(scriptPath.parent) + '\\stanford-dogs-dataset-traintest\\cropped')
np.random.seed(2)
data = ImageDataBunch.from_folder(modelPath, train='train', valid_pct=0.2, size=224, num_workers=0)
data.normalize(imagenet_stats)
data.show_batch(rows=5, figsize=(25,25))

learner = cnn_learner(data, models.resnet34,  metrics=[accuracy, error_rate])
learner.save("resnet34-fit5")

learner.lr_find()
learner.recorder.plot()

learner.unfreeze()
learner.fit_one_cycle(3, max_lr=slice(1e-6,1e-2))
learner.save("resnet34-fit5-stage2")

interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize = (40,40), dpi = 150)



# %%
