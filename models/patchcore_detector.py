
from anomalib.data import Folder
from anomalib.models import Patchcore as PatchCore
from anomalib.engine import Engine

from huggingface_hub import login

import os 
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("Anomalib libraries imported")

    login(token='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')
    print("Logged into Hugging Face!")

    datamodule = Folder(
        name = 'raw',
        root = '/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/raw',
        normal_dir = 'normal galaxies',
        abnormal_dir = 'anomalous galaxies',

    )
    datamodule.setup()

    model = PatchCore(
        backbone = "resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.0001 # Reduced ratio
    )

    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    # Train
    engine = Engine(
        enable_progress_bar=True # Disable progress bar to avoid RecursionError
    )
    engine.fit(model=model, datamodule=datamodule)

    # Testing and Viewing Results
    test_results = engine.test(model=model, datamodule=datamodule) # Capture test results
    predictions = engine.predict(model=model, datamodule=datamodule)

    print(predictions)