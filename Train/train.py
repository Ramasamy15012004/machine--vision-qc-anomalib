# train.py
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from torchvision.transforms.v2 import Resize, Compose
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Resize, Compose

pre_processor = PreProcessor(
    transform=Compose([
        Resize((512, 512))
    ])
)
datamodule = Folder(
    name                = "metal_part",
    root                = "Dataset",
    normal_dir          = "train/OK",
    abnormal_dir        = "test/NG",
    normal_test_dir     = "val/OK",
    train_batch_size    = 8,
    eval_batch_size     = 8,
    num_workers         = 0,
    val_split_ratio     = 0.5,
)

model = Patchcore(
    backbone               = "wide_resnet50_2",
    layers                 = ["layer2", "layer3"],
    coreset_sampling_ratio = 0.1,
    num_neighbors          = 9,
    pre_processor          = pre_processor,
)

engine = Engine()

engine.train(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule,  model=model)

print("Test results:", results)