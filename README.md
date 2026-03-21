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
- evaluare cu metrici intrusive (PESQ, STOI, SI-SDR, ΔSNR, CSIG, CBAK, COVL)
- suport pentru DNSMOS + NISQA (non-intrusive)

Structură proiect:

ULP-SE-ATTENNUATE/
    attenuate/model.py
    dataset/
        download_voicebank_demand.sh
        dns5_pipeline.py
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

Pentru a ține dataset-ul pe un disc extern, folosește un root explicit:

bash dataset/download_voicebank_demand.sh vb --root /mnt/ldm/ULP-SE-aTENNuate/dataset
bash dataset/prepare_splits_voicebank-demand.sh \
  --data-dir /mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k

Aceste scripturi creează manifestele:

dataset/voicebank-demand/16k/train.csv
dataset/voicebank-demand/16k/test.csv

Dacă folosești `--data-dir`, manifestele sunt scrise direct în acel director.

----------------------------------------------------------------------
2B. PIPELINE DNS5 TRACK 1 HEADSET
----------------------------------------------------------------------

Pentru un corpus mare, relevant pentru modelul curent, folosește pipeline-ul
offline DNS5 Track 1 Headset. Acesta descarcă, extrage, sintetizează perechi
`noisy,clean` la `16 kHz`, verifică manifestele și poate șterge raw-ul doar
după succes.

Dataset final implicit:

/mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k/
    clean_train/
    noisy_train/
    clean_val/
    noisy_val/
    train_shards/*.csv
    train.csv
    val.csv

Exemplu flux complet:

python dataset/dns5_pipeline.py download \
  --source smoke \
  --staging-root /mnt/ldm/DNS-Challenge \
  --output-root /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k

python dataset/dns5_pipeline.py extract \
  --source smoke \
  --staging-root /mnt/ldm/DNS-Challenge \
  --output-root /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k

python dataset/dns5_pipeline.py synthesize \
  --source smoke \
  --staging-root /mnt/ldm/DNS-Challenge \
  --output-root /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k \
  --max-augmentations-per-clean 1

python dataset/dns5_pipeline.py verify \
  --source smoke \
  --staging-root /mnt/ldm/DNS-Challenge \
  --output-root /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k

python dataset/dns5_pipeline.py cleanup \
  --source smoke shared \
  --staging-root /mnt/ldm/DNS-Challenge \
  --output-root /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k \
  --keep-devtest

Alias-uri utile:

- `smoke` = `VocalSet_48kHz_mono + emotional_speech + vctk_wav48_silence_trimmed + noise_ir + filelists_headset`
- `all_clean` = toate sursele clean DNS5 Track 1
- `all_relevant` = `all_clean + noise_ir + filelists_headset + devtest`

Subcomenzi:

- `download`
- `extract`
- `synthesize`
- `verify`
- `cleanup`

Parametri importanți:

- `--source`
- `--output-root`
- `--staging-root`
- `--free-space-floor-gb`
- `--max-augmentations-per-clean`
- `--keep-devtest`

----------------------------------------------------------------------
3. TRAINING MODEL (train.py)
----------------------------------------------------------------------

Exemplu:

python train.py \
  --train-csv dataset/voicebank-demand/16k/train.csv \
  --val-csv dataset/voicebank-demand/16k/test.csv \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-3 \
  --segment-len 32000 \
  --checkpoint-out checkpoints/atennuate_fp32.pt

Parametri:
--train-csv       CSV cu perechi noisy/clean
--val-csv         CSV opțional noisy/clean pentru validare și early stopping
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

Rulează modelul pe setul test și apoi măsoară următoarele metrici:

- count
- pesq_mean (PESQ mediu)
- stoi_mean (STOI mediu)
- sisdr_mean (SI-SDR mediu)
- delta_snr_mean (ΔSNR mediu)
- csig_mean (CSIG mediu)
- cbak_mean (CBAK mediu)
- covl_mean (COVL mediu)
- dnsmos_sig_mean (DNSMOS SIG mediu) — obligatoriu
- dnsmos_bak_mean (DNSMOS BAK mediu) — obligatoriu
- dnsmos_ovr_mean (DNSMOS OVR mediu) — obligatoriu

Aceste valori sunt exportate în JSON în `oracle_metrics.json`, iar fiecare
rulare de evaluare trebuie să le colecteze pe toate.

python evaluate_metrics.py \
  --checkpoint checkpoints_quantized/atennuate_32bit.pt \
  --manifest dataset/voicebank-demand/16k/test.csv \
  --enhanced-dir eval_outputs/32bit \
  --oracle-json eval_outputs/32bit/oracle_metrics.json

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

python train.py --train-csv dataset/voicebank-demand/16k/train.csv

Sau, pentru DNS5:

python train.py \
  --train-csv /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k/train.csv \
  --val-csv /mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k/val.csv

python quantize.py --base-checkpoint checkpoints/atennuate_fp32.pt

python export_onnx.py \
  --checkpoint checkpoints_quantized/atennuate_8bit.pt \
  --out onnx_exports/atennuate_8bit_T32000.onnx \
  --seq-len 32000

python evaluate_metrics.py \
  --checkpoint checkpoints_quantized/atennuate_8bit.pt \
  --manifest dataset/voicebank-demand/16k/test.csv \
  --enhanced-dir eval_outputs/8bit \
  --oracle-json eval_outputs/8bit/oracle_metrics.json

python inference.py \
  --checkpoint checkpoints/atennuate_fp32.pt \
  --input noisy_samples/audioset_realrec_babycry_2x43exdQ5bo.wav \
  --output clean_samples/audioset_realrec_babycry_2x43exdQ5bo.wav

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
