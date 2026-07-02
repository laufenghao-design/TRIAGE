## Installation

```bash
conda create -n TRIAGE python==3.8.13
conda activate TRIAGE
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Data Preparation

Download Cholec80 from https://camma.unistra.fr/datasets/ and place videos in `./datasets/Cholec80/videos/`, tool annotations in `./datasets/Cholec80/tool_annotations/`, phase annotations in `./datasets/Cholec80/phase_annotations/`.

Download AutoLaparo from https://autolaparo.github.io/ and place videos in `./datasets/AutoLaparo_Task1/videos/`, labels in `./datasets/AutoLaparo_Task1/labels/`.

```bash
# Cholec80
python datasets/data_preprosses/extract_frames_ch80.py
python datasets/data_preprosses/generate_labels_ch80.py
python datasets/data_preprosses/frame_cutmargin.py
python datasets/data_preprosses/resize_frames_ch80.py

# AutoLaparo
python datasets/data_preprosses/extract_frames_autolaparo.py
python datasets/data_preprosses/generate_labels_autolaparo.py
python datasets/data_preprosses/resize_frames.py
```

## Pretrained Weights

```bash
mkdir -p models
wget -O models/TimeSformer_divST_8x32_224_K400.pyth \
  "https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=1"
```

Download Surgformer finetuned weights (refer to the original Surgformer repository for download links) and place at:

```
weight/
└── Surgformer/
    ├── AutoLaparo/
    │   └── Surgformer_HTA_16_4/
    │       └── mp_rank_00_model_states.pt
    └── Cholec80/
        └── Surgformer_HTA_KCA_16_4/
            └── mp_rank_00_model_states.pt
```

## Testing Surgformer with Token Merging

```bash
sh scripts/test_autolaparo.sh
sh scripts/test_cholec80.sh
```

Merge result files:

```bash
python datasets/convert_results/convert_cholec80.py \
    --nproc_per_node <N_GPUS> \
    --main_path <result_dir>

python datasets/convert_results/convert_autolaparo.py \
    --nproc_per_node <N_GPUS> \
    --main_path <result_dir>
```

Copy `phase_annotations/` and `prediction/` from `<result_dir>` into `evaluation_matlab/`, then run `Main.m` (Cholec80) or `Main_AutoLaparo.m` (AutoLaparo).
