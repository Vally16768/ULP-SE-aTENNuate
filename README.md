ULP-SE-ATTENNUATE
High-Fidelity Speech Denoising • aTENNuate Architecture • PyTorch • ONNX • Quantization • PESQ/STOI/SI-SDR

----------------------------------------------------------------------
OVERVIEW
----------------------------------------------------------------------
ULP-SE-ATTENNUATE este o implementare completă a arhitecturii moderne aTENNuate
pentru speech denoising. Proiectul este construit pentru:

- training eficient în PyTorch
- inferență offline pe fișiere WAV
- quantizare 32 / 16 / 8 / 4 / 2 biți
- export ONNX
- evaluare cu metrici intrusive (PESQ, STOI, SI-SDR, ΔSNR)
- suport pentru DNSMOS + NISQA (non-intrusive)

Structură proiect:

ULP-SE-ATTENNUATE/
    attenuate/model.py
    dataset/
        download_voicebank_demand.sh
        prepare_splits_voicebank-demand.sh
        voicebank-demand/train.csv
        voicebank-demand/test.csv
    metrics/*.py
    train.py
    inference.py
    quantize.py
    export_onnx.py
    evaluate_metrics.py

----------------------------------------------------------------------
1. INSTALARE
----------------------------------------------------------------------

Creează mediul virtual:

python -m venv .venv
source .venv/bin/activate              (Linux/Mac)
.\.venv\Scripts\activate               (Windows)

Instalează dependințe:

pip install -r requirements.txt

----------------------------------------------------------------------
2. DESCĂRCARE + PREGĂTIRE VoiceBank-DEMAND
----------------------------------------------------------------------

bash dataset/download_voicebank_demand.sh
bash dataset/prepare_splits_voicebank-demand.sh

Aceste scripturi creează:

dataset/voicebank-demand/train.csv
dataset/voicebank-demand/test.csv

----------------------------------------------------------------------
3. TRAINING MODEL (train.py)
----------------------------------------------------------------------

Exemplu:

python train.py \
  --train-csv dataset/voicebank-demand/train.csv \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-3 \
  --segment-len 32000 \
  --checkpoint-out checkpoints/atennuate_fp32.pt

Parametri:
--train-csv       CSV cu perechi noisy/clean
--epochs          număr epoci
--batch-size      batch size
--lr              learning rate
--segment-len     lungimea segmentelor audio
--checkpoint-out  fișier în care se salvează modelul

----------------------------------------------------------------------
4. INFERENȚĂ PE UN FIȘIER AUDIO (inference.py)
----------------------------------------------------------------------

python inference.py \
  --checkpoint checkpoints/atennuate_fp32.pt \
  --input noisy_samples/example.wav \
  --output denoised_samples/example_denoised.wav

Parametri:
--checkpoint   model .pt (FP32 sau cuantizat)
--input        fișier WAV zgomotos
--output       fișier WAV denoisat

----------------------------------------------------------------------
5. CUANTIZARE 32/16/8/4/2 BIȚI (quantize.py)
----------------------------------------------------------------------

python quantize.py \
  --base-checkpoint checkpoints/atennuate_fp32.pt \
  --out-dir checkpoints_quantized \
  --bits 32 16 8 4 2

Director rezultat:

checkpoints_quantized/
    atennuate_32bit.pt
    atennuate_16bit.pt
    atennuate_8bit.pt
    atennuate_4bit.pt
    atennuate_2bit.pt

----------------------------------------------------------------------
6. EXPORT ONNX (export_onnx.py)
----------------------------------------------------------------------

python export_onnx.py \
  --checkpoint checkpoints_quantized/atennuate_8bit.pt \
  --out onnx_exports/atennuate_8bit.onnx \
  --sample-len 16000 \
  --opset 17

Ieșire:
Model ONNX cu input/output dinamic: [1, 1, T]

----------------------------------------------------------------------
7. EVALUARE METRICI INTRUSIVE (evaluate_metrics.py)
----------------------------------------------------------------------

Rulează modelul pe setul test și apoi măsoară PESQ, STOI, ΔSNR, SI-SDR.

python evaluate_metrics.py \
  --checkpoint checkpoints_quantized/atennuate_8bit.pt \
  --manifest dataset/voicebank-demand/test.csv \
  --enhanced-dir eval_outputs/8bit \
  --oracle-json eval_outputs/8bit/oracle_metrics.json

Director rezultat:

eval_outputs/8bit/
    *.wav (fișiere enhanced)
    manifest_oracle.csv
    oracle_metrics.json

----------------------------------------------------------------------
8. FLUX COMPLET RECOMANDAT
----------------------------------------------------------------------

bash dataset/download_voicebank_demand.sh
bash dataset/prepare_splits_voicebank-demand.sh

python train.py --train-csv dataset/voicebank-demand/train.csv

python quantize.py --base-checkpoint checkpoints/atennuate_fp32.pt

python export_onnx.py \
  --checkpoint checkpoints_quantized/atennuate_8bit.pt \
  --out onnx_exports/atennuate_8bit.onnx

python evaluate_metrics.py \
  --checkpoint checkpoints_quantized/atennuate_8bit.pt \
  --manifest dataset/voicebank-demand/test.csv \
  --enhanced-dir eval_outputs/8bit \
  --oracle-json eval_outputs/8bit/oracle_metrics.json

python inference.py \
  --checkpoint checkpoints/atennuate_fp32.pt \
  --input path/to/noisy.wav \
  --output path/to/noisy_denoised.wav

----------------------------------------------------------------------
9. ROADMAP
----------------------------------------------------------------------

- integrare MRSTFT Loss / SI-SNR Loss
- inferență real-time stateful (SSM streaming)
- optimizări ONNX pentru mobile (CoreML / NNAPI / TensorRT)
- suport pentru Edge TPU
- versiuni mini (Mobile/Tiny)

----------------------------------------------------------------------
10. LICENȚĂ
----------------------------------------------------------------------
MIT License — utilizare liberă academică, comercială și embedded.

