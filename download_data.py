from datasets import load_dataset
from datasets.utils.logging import set_verbosity_debug

# Specify the directory where you want the dataset to be downloaded
cache_dir = "data"
set_verbosity_debug()

# Load the dataset and specify the cache directory
ds = load_dataset("OPO-alignment/ListUltraFeedback", cache_dir=cache_dir)