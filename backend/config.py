from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_DTYPE = "bfloat16"
N_GPUS = 6

# Generation defaults — 256px/4step for fast exploration, 512px/20step for detail
DEFAULT_GRID_SIZE = 15
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_SEED = 42

# High-quality re-render
HQ_HEIGHT = 512
HQ_WIDTH = 512
HQ_NUM_INFERENCE_STEPS = 20

# Ridge detection (DINOv2 neighbor distance)
RIDGE_THRESHOLD_TAU = 1.5

# Grid coordinate ranges
ALPHA_RANGE = (0.0, 1.0)
BETA_RANGE = (0.0, 1.0)  # for 3-prompt: both [0,1]

# Thumbnail settings
THUMBNAIL_SIZE = 64
THUMBNAIL_QUALITY = 85

# Paths
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache_data"
THUMBNAILS_DIR = CACHE_DIR / "thumbnails"
RESULTS_DIR = CACHE_DIR / "results"

# Polling
RESULT_POLL_INTERVAL = 0.5
