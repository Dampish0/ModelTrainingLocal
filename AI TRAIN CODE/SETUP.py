import subprocess
import sys


def inst(package, ):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


inst("gitpython")
inst("transformers")
inst("datasets")
inst("bitsandbytes")
inst("accelerate")
inst("sentencepiece")
inst("gradio")
inst("numba")

from git import Repo


Repo.clone_from("https://huggingface.co/datasets/Dampish/QuickTrain", to_path="QuickTrain")
