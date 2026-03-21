# Academic README: Speech Enhancement Study, Baselines, Implemented Methods, and MCU Deployability

## 0. Scope and Comparability Note

Acest document rezuma experimental proiectul din repo, cu accent pe:

1. dataset-ul folosit si protocolul experimental;
2. baseline-urile implementate si cel mai bun rezultat dintre ele;
3. toate familiile de metode implementate, raportate cu metricile disponibile;
4. cea mai buna metoda deployable pe MCU sub constrangerile de produs discutate.

Doua observatii metodologice sunt obligatorii:

- Rezultatele `16 kHz` folosesc `PESQ-WB`, iar rezultatele `8 kHz` folosesc `PESQ-NB`. Valorile nu trebuie comparate direct ca si cum ar fi acelasi benchmark.
- Campaniile au folosit atat `val_select` (validare interna pentru selectie), cat si `test.csv` (hold-out final). Un `PESQ` pe `test.csv` este mai puternic ca dovada decat unul pe `val_select`.

Toate tabelele de mai jos includ coloana `Protocol` tocmai pentru a pastra comparatia corecta.

Smoke/debug runs au fost excluse din acest README.

---

## 1. Dataset Folosit si Cum a Fost Folosit

### 1.1 Corpusul de baza

Setul principal folosit in toate campaniile canonice este `VoiceBank+DEMAND`, pregatit prin:

- `dataset/download_voicebank_demand.sh`
- `dataset/prepare_splits_voicebank-demand.sh`
- `sebench/splits.py`

Manifeste canonice:

- `dataset/voicebank-demand/16k/train.csv`
- `dataset/voicebank-demand/16k/test.csv`
- `dataset/voicebank-demand/16k/campaign/train_fit.csv`
- `dataset/voicebank-demand/16k/campaign/val_rank.csv`
- `dataset/voicebank-demand/16k/campaign/val_select.csv`

Pentru ramura embedded a fost generata si o varianta `8 kHz`, cu splituri paralele:

- `dataset/voicebank-demand/8k/campaign/train_fit.csv`
- `dataset/voicebank-demand/8k/campaign/val_rank.csv`
- `dataset/voicebank-demand/8k/campaign/val_select.csv`

### 1.2 Cardinalitati

Cardinalitatile efective din manifestele curente sunt:

| Split | Sampling rate | Perechi audio |
|---|---:|---:|
| `train.csv` | 16 kHz | 11,572 |
| `test.csv` | 16 kHz | 824 |
| `train_fit.csv` | 16 kHz | 9,754 |
| `val_rank.csv` | 16 kHz | 128 |
| `val_select.csv` | 16 kHz | 1,690 |
| `train_fit.csv` | 8 kHz | 9,754 |
| `val_rank.csv` | 8 kHz | 128 |
| `val_select.csv` | 8 kHz | 1,690 |

### 1.3 Cum au fost folosite split-urile

Spliturile sunt generate prin `build_voicebank_campaign_splits()` in `sebench/splits.py`, folosind speaker-held-out validation.

- `train_fit`:
  antrenare propriu-zisa.
- `val_rank`:
  selectie rapida in bucla de training, pentru ranking intermediar si early stopping.
- `val_select`:
  selectie finala intre configuratii/campanii; acesta este splitul folosit cel mai des pentru tabelele comparative interne.
- `test.csv`:
  evaluare finala hold-out.

### 1.4 Rolul ramurilor 16 kHz si 8 kHz

- `16 kHz`:
  ramura de calitate maxima si benchmark canonical wideband.
- `8 kHz`:
  ramura de deploy embedded, folosita pentru studenti distilati, audit de MCU si selectie low-power.

### 1.5 Dataset suplimentar

Repo-ul contine si pipeline pentru `DNS5 Track 1 Headset` (`dataset/dns5_pipeline.py`), dar rezultatele canonice din acest README provin din VoiceBank+DEMAND. DNS5 a fost pregatit pentru extinderi ulterioare, nu pentru tabelele finale de aici.

---

## 2. Baselines: Implementare, Rol si Cel Mai Bun Rezultat

### 2.1 Baseline clasic: `spectral_gating`

**Cum arata metoda**

Nu este o retea neuronala. Este un post-filtru spectral determinist implementat in `sebench/postfilters.py`.

- STFT fix:
  `n_fft=512`, `hop_length=128`, `win_length=512`, fereastra Hann.
- Moduri:
  `sg_residual_soft` si `sg_input_floor`.
- Masca soft:
  bazata pe estimarea noise floor si sigmoid peste magnitudine.

**Cum a fost implementata**

- configuratie determinista prin `SpectralGateConfig`;
- folosita atat ca baseline clasic, cat si ca guidance auxiliar pentru `tiny_stm32_hybrid_sg`;
- auditata si prin simulatorul MCU (`sebench/stm32sim.py`).

**La ce este buna**

- foarte ieftina computațional;
- complet determinista si usor de implementat pe MCU;
- excelenta ca fallback sau ca feature auxiliar pentru un model neuronal mic.

**Downsides**

- plafon de calitate mai jos decat modelele neuronale bune;
- poate suprasupresa in conditii grele;
- nu beneficiaza de invatare data-driven.

**Cel mai bun rezultat clasic**

| Baseline | Protocol | PESQ | STOI | SI-SDR | Delta SNR | DNSMOS | CSIG/CBAK/COVL | Latenta host | MCU deployable |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| `spectral_gating` | `8 kHz`, `val_select` | **2.3762** | 0.8264 | 6.3067 | 0.0603 | n/a | n/a | n/a | Da |
| `spectral_gating` | `16 kHz`, `val_select` | 2.2139 | n/a | n/a | n/a | n/a | n/a | n/a | Da |

Concluzie: dintre baseline-urile clasice implementate si retinute canonic, `spectral_gating` este baseline-ul clasic cel mai bun.

### 2.2 Baseline-uri neuronale de referinta

Repo-ul implementeaza si evalueaza explicit urmatoarele familii de referinta:

- `aTENNuate`
- `MP-SENet`
- `CMGAN-small`
- `FullSubNet+`
- `MetricGAN+ raw` (wrapper peste modelul SpeechBrain pretrained)

Acestea sunt baseline-uri neuronale de referinta, nu toate sunt candidati de deploy.

**Cel mai bun baseline neuronal**

| Baseline neural | Protocol | PESQ | STOI | SI-SDR | Latenta host |
|---|---|---:|---:|---:|---:|
| `MetricGAN+ raw` exact reference | `16 kHz`, `test` | **3.1245** | **0.9311** | **8.4626** | 0.0360 s / 10 s |

Acesta este cel mai bun baseline neuronal de referinta din proiect, dar nu este deployable pe MCU.

---

## 3. Metodele Implementate, Ordonate dupa PESQ

### 3.1 Interpretarea tabelelor

Pentru rigoare, metodele sunt ordonate dupa `PESQ` in interiorul fiecarui protocol comparabil:

- `16 kHz / val_select`
- `16 kHz / test`
- `8 kHz / val_select`
- `8 kHz / test`

Astfel evitam comparatii incorecte intre `PESQ-WB` si `PESQ-NB`.

### 3.2 Metode pe `16 kHz / val_select`, ordonate crescator dupa PESQ

| Metoda | Arhitectura pe scurt | Protocol | PESQ | STOI | SI-SDR | Delta SNR | DNSMOS OVR | CSIG | CBAK | COVL | Latenta host |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MP-SENet` | 2 ramuri conv2d: magnitudine + faza, head separat pentru `mag_mask` si `delta_phase` | `16 kHz`, `val_select` | 1.6649 | 0.8088 | 10.8351 | 4.9779 | 3.0143 | n/a | n/a | n/a | 0.0103 |
| `aTENNuate` | model waveform-domain encoder-decoder, varianta mica `[16,32,64,96,128]`, `repeat=8` | `16 kHz`, `val_select` | 1.6701 | 0.8392 | 11.7949 | 5.9226 | 3.1867 | 2.6351 | 2.5026 | 2.1111 | 0.0312 |
| `CMGAN-small` | encoder conv2d + Transformer encoder pe frecventa/timp + complex mask head | `16 kHz`, `val_select` | 1.8623 | 0.8345 | 12.7590 | 6.8478 | 3.2276 | n/a | n/a | n/a | 0.2984 |
| `FullSubNet+` | sub-branch conv2d + full-band bidirectional GRU + complex mask head | `16 kHz`, `val_select` | 2.0709 | 0.8554 | 13.7726 | 7.7752 | 3.1173 | n/a | n/a | n/a | 0.0131 |
| `MetricGAN+ refiner` | teacher `MetricGAN+` inghetat + refiner waveform pe 1D conv residual blocks | `16 kHz`, `val_select` | 2.1265 | 0.8651 | 13.2022 | 7.1781 | n/a | 3.4114 | 2.8664 | 2.7514 | 0.2776 |

### 3.3 Metode pe `16 kHz / test`, ordonate crescator dupa PESQ

| Metoda | Arhitectura pe scurt | Protocol | PESQ | STOI | SI-SDR | DNSMOS | Latenta host | Deployable on MCU |
|---|---|---|---:|---:|---:|---|---:|---|
| `MetricGAN+ 16k int8 portable proxy` | proxy auditat pentru cuantizare dinamica, bazat pe `nn.LSTM` + `nn.Linear` | `16 kHz`, `test` | 2.5091 | 0.8781 | 5.2606 | n/a | n/a | Nu |
| `MetricGAN+ raw pretrained` | wrapper exact peste bundle-ul SpeechBrain `speechbrain/metricgan-plus-voicebank` | `16 kHz`, `test` | 3.1225 | 0.9311 | 8.4588 | n/a | 0.0558 | Nu |
| `MetricGAN+ 16k exact ref` | aceeasi referinta de calitate, auditata in fluxul `teacher_audit` | `16 kHz`, `test` | **3.1245** | **0.9311** | **8.4626** | n/a | 0.0360 | Nu |

### 3.4 Metode pe `8 kHz / val_select`, ordonate crescator dupa PESQ

| Metoda | Arhitectura pe scurt | Protocol | PESQ | STOI | SI-SDR | Delta SNR | Latenta host | MCU deployable |
|---|---|---|---:|---:|---:|---:|---:|---|
| `spectral_gating` | baseline clasic STFT deterministic | `8 kHz`, `val_select` | 2.3762 | 0.8264 | 6.3067 | 0.0603 | n/a | Da |
| `tiny_stm32_fc` | features ERB + energie, context 5 cadre, MLP `165 -> 128 -> 64 -> 32 gains` | `8 kHz`, `val_select` | 2.5990 | 0.8174 | 9.2284 | 3.6375 | 0.00195 | Da |
| `tiny_stm32_hybrid_sg` | `tiny_stm32_fc` + guidance `spectral_gating`, MLP `325 -> 160 -> 80 -> 32 gains` | `8 kHz`, `val_select` | 2.6559 | 0.8262 | 9.3495 | 3.7507 | 0.0148 | Da |
| `MetricGAN+ native8k causal_xs` | GRU cauzal, 1 strat, hidden 64, linear 96, front-end `256/80/160` | `8 kHz`, `val_select` | 2.8381 | 0.8516 | 12.9600 | 7.0055 | 0.00280 | Da |
| `MetricGAN+ native8k causal_s` (float) | GRU cauzal, 1 strat, hidden 96, linear 128, front-end `256/80/160` | `8 kHz`, `val_select` | 2.8605 | 0.8542 | 13.0793 | 7.1220 | 0.00303 | Da |
| `MetricGAN+ native8k causal_s` (QAT / final) | varianta `causal_s` dupa cuantizare-aware fine-tuning cu `D2` | `8 kHz`, `val_select` | **2.8608** | **0.8538** | **13.3801** | **7.4184** | 0.00368 | Da |
| `MetricGAN+ native8k` (int8) | model mare bidirectional, cuantizat dinamic, audit-only | `8 kHz`, `val_select` | 3.0716 | 0.8728 | 14.2782 | 8.2391 | 0.2007 | Nu |
| `MetricGAN+ native8k` (fp32) | model mare bidirectional, 2x BiLSTM + linear mask head | `8 kHz`, `val_select` | **3.0736** | 0.8726 | 14.2854 | 8.2469 | 0.0569 | Nu |

### 3.5 Metode pe `8 kHz / test`, ordonate crescator dupa PESQ

| Metoda | Arhitectura pe scurt | Protocol | PESQ | STOI | SI-SDR | Quantization drop | Deployable on MCU |
|---|---|---|---:|---:|---:|---:|---|
| `MetricGAN+ native8k` (int8) | teacher audit-only, cuantizare dinamica | `8 kHz`, `test` | 3.4328 | 0.9403 | 18.0114 | 0.0020 | Nu |
| `MetricGAN+ native8k` (fp32) | teacher audit-only, model maxim de calitate | `8 kHz`, `test` | **3.4372** | **0.9403** | **18.0159** | n/a | Nu |

### 3.6 Ablatii / metode auxiliare implementate

Pe langa familiile de mai sus, proiectul a implementat si evaluat:

- `MetricGAN+ + spectral gating`:
  variante `pfsg_residual_soft-light`, `pfsg_residual_soft-medium`, `pfsg_input_floor-light`, `pfsg_input_floor-medium`.
- `postfilter gating` si pe alte familii:
  `aTENNuate`, `FullSubNet+`, `CMGAN-small`, `MP-SENet`.
- `MetricGAN+ refiner`:
  stage-1 inghetat + refiner residual waveform.

In practica, aceste ablatii nu au depasit combinatia `MetricGAN+ raw` pentru calitate maxima si nici `teacher-lite causal_s` pentru deploy. Din acest motiv ele au ramas directii secundare, nu metoda finala.

---

## 4. Profilul Fiecarei Metode: Cum Arata Reteaua, Cum a Fost Implementata, Puncte Tari si Limitari

### 4.1 `spectral_gating`

- **Tip**:
  metoda clasica, non-neuronala.
- **Implementare**:
  `sebench/postfilters.py`.
- **Structura**:
  STFT + estimare `noise_floor` + masca soft sigmoid + iSTFT.
- **Puncte tari**:
  cost foarte mic, determinism, robustete pentru deploy, excelent ca baseline si guidance.
- **Limitari**:
  plafon de calitate mai mic; nu invata din date.

### 4.2 `aTENNuate`

- **Tip**:
  model waveform-domain.
- **Implementare**:
  `attenuate/model.py`, adaptat prin `AtennuateAdapter` in `sebench/models.py`.
- **Structura**:
  encoder-decoder temporal, varianta mica cu canale `[16, 32, 64, 96, 128]`, `repeat=8`, `num_coeffs=12`, resampling multi-stage.
- **Puncte tari**:
  simplu de antrenat, STOI decent, inferenta rapida pe GPU.
- **Limitari**:
  PESQ semnificativ sub modelele STFT mai puternice; nu a fost un candidat bun pentru deploy embedded final.

### 4.3 `MP-SENet`

- **Tip**:
  model spectral cu ramuri separate pentru magnitudine si faza.
- **Implementare**:
  `MPSENet` in `sebench/models.py`.
- **Structura**:
  doua ramuri conv2d; una produce `mag_mask`, cealalta `delta_phase`.
- **Puncte tari**:
  trateaza explicit faza, latenta host mica.
- **Limitari**:
  in setarile din repo a ramas sub `aTENNuate` la PESQ si mult sub `FullSubNet+`.

### 4.4 `CMGAN-small`

- **Tip**:
  model spectral cu encoder conv si Transformer.
- **Implementare**:
  `CMGANSmall` in `sebench/models.py`.
- **Structura**:
  conv2d encoder pe `[real, imag, |X|]`, apoi Transformer encoder pe secvente per frecventa, apoi complex mask head.
- **Puncte tari**:
  crestere clara fata de `aTENNuate` si `MP-SENet`.
- **Limitari**:
  latenta host mare fata de `FullSubNet+`; nu a fost competitiv pentru deploy MCU.

### 4.5 `FullSubNet+`

- **Tip**:
  model spectral cu combinatie sub-band + full-band.
- **Implementare**:
  `FullSubNetPlus` in `sebench/models.py`.
- **Structura**:
  conv2d sub-branch pe `[real, imag, mag]`, GRU bidirectional full-band, fuziune si complex mask head.
- **Puncte tari**:
  cel mai bun baseline neural antrenat local dintre familiile clasice din sweep-ul principal.
- **Limitari**:
  tot sub teacher-ul pretrained `MetricGAN+`; bidirectional, deci nefavorabil pentru embedded.

### 4.6 `MetricGAN+ raw`

- **Tip**:
  baseline / teacher pretrained de calitate maxima.
- **Implementare**:
  `MetricGANPlusAdapter` in `sebench/models.py`, wrapper peste `speechbrain/metricgan-plus-voicebank`.
- **Structura**:
  retea de tip spectral masking bazata pe BLSTM + straturi liniare, folosita exact prin bundle-ul SpeechBrain.
- **Puncte tari**:
  cea mai buna referinta wideband la `16 kHz`; baza pentru distillation.
- **Limitari**:
  nu este deployable pe MCU; model mare, bidirectional, cerinte de memorie ridicate.

### 4.7 `MetricGAN+ + spectral gating`

- **Tip**:
  baseline neural plus post-filtru clasic.
- **Implementare**:
  `PostFilterEnhancer` + `SpectralGateConfig` in `sebench/postfilters.py`.
- **Structura**:
  iesirea teacher-ului este trecuta prin `spectral_gate_waveform`.
- **Puncte tari**:
  foarte usor de testat si de integrat in pipeline.
- **Limitari**:
  in campania canonica nu a depasit `MetricGAN+ raw`; toate variantele au scazut fata de raw.

### 4.8 `MetricGAN+ refiner`

- **Tip**:
  cascada in doua etape.
- **Implementare**:
  `MetricGANPlusRefiner` in `sebench/models.py`.
- **Structura**:
  `MetricGAN+` inghetat pentru stage-1, apoi `ResidualWaveRefiner` cu conv1d reziduale pe `[noisy, stage1, residual]`.
- **Puncte tari**:
  idee arhitecturala rezonabila pentru refinement.
- **Limitari**:
  in practica a ramas sub `MetricGAN+ raw`; raport slab intre cost si castig.

### 4.9 `tiny_stm32_fc`

- **Tip**:
  student embedded foarte mic.
- **Implementare**:
  `TinySTM32FC` in `sebench/stm32_models.py`.
- **Structura**:
  features ERB + energie, context 5 cadre, MLP `165 -> 128 -> 64 -> 32`, reconstructie in STFT cu faza semnalului zgomotos.
- **Puncte tari**:
  cel mai simplu student neuronal deployable; trece inclusiv profilul de referinta `STM32L4`.
- **Limitari**:
  calitate mai mica decat `tiny_stm32_hybrid_sg` si `teacher-lite`.

### 4.10 `tiny_stm32_hybrid_sg`

- **Tip**:
  student embedded cu guidance clasic.
- **Implementare**:
  `TinySTM32HybridSG` in `sebench/stm32_models.py`.
- **Structura**:
  MLP `325 -> 160 -> 80 -> 32`, cu intrare concatenata `log-ERB noisy + spectral_gating guidance + energy`.
- **Puncte tari**:
  cel mai bun student tiny din prima generatie; bate clar baseline-ul clasic.
- **Limitari**:
  nu trece profilul `STM32L4` in auditul curent; pentru low-power are nevoie de MCU-uri ceva mai puternice decat `STM32L4`.

### 4.11 `MetricGAN+ native8k`

- **Tip**:
  teacher de maxima calitate pe ramura `8 kHz`.
- **Implementare**:
  `MetricGANLikeEnhancer` in `sebench/models.py`.
- **Structura**:
  STFT `256/80/160`, 2 straturi BiLSTM, linear1, linear2, learnable sigmoid mask head.
- **Puncte tari**:
  cea mai buna calitate din proiect pe ramura `8 kHz`; cuantizarea dinamica aproape nu ii reduce PESQ.
- **Limitari**:
  audit-only; aproximativ `1.67 MB` weights, `2.23 MB` SRAM peak, `500 ms` lookahead; nerecomandat pentru produs MCU. Fara constrangerea de putere, ramane realist doar ca demo pe `STM32N6` si `i.MX RT700`, nu ca model de produs.

### 4.12 `MetricGAN+ native8k causal_xs`

- **Tip**:
  teacher-lite cauzal.
- **Implementare**:
  `MetricGANCausalLiteEnhancer` in `sebench/models.py`.
- **Structura**:
  GRU unidirectional, 1 strat, hidden 64, linear 96, lookahead `16 ms`.
- **Puncte tari**:
  deployabil pe toate profilele low-power din shortlist, calitate foarte buna.
- **Limitari**:
  mai slab decat `causal_s`; doar varianta de compresie mai agresiva.

### 4.13 `MetricGAN+ native8k causal_s`

- **Tip**:
  metoda finala teacher-lite.
- **Implementare**:
  `MetricGANCausalLiteEnhancer` in `sebench/models.py`, antrenata prin distillation si apoi QAT.
- **Structura**:
  GRU unidirectional, 1 strat, hidden 96, linear 128, lookahead `16 ms`, front-end `8 kHz / 256 / 80 / 160`.
- **Puncte tari**:
  cel mai bun compromis intre calitate si deploy;
  bate atat `tiny_stm32_hybrid_sg`, cat si `spectral_gating`;
  trece constrangerile `on-MCU + real-time + <50 mW`.
- **Limitari**:
  nu trece profilul de referinta `STM32L4`; necesita o clasa de MCU low-power mai moderna (`STM32U5`, `nRF54H20`, `Apollo4 Blue+`).

---

## 5. Cea Mai Buna Metoda Deployable pe Baza Constrangerilor si Metricilor

### 5.1 Constrangeri considerate

Conditia de produs folosita in auditul final a fost:

- `on-MCU`;
- `real-time`;
- `power < 50 mW`;
- pentru ramura teacher-lite:
  `lookahead <= 80 ms`.

### 5.2 Castigatorul actual

**Cea mai buna metoda deployable finala este:**

`metricgan_plus_native8k_causal_s-small-lr0.0002-seg16000-lossD2-seed0`

Adica varianta `teacher-lite causal_s` dupa `QAT`.

### 5.3 Metrici finale ale metodei deployable

| Metoda deployable | Protocol | PESQ | STOI | SI-SDR | Delta SNR | Latenta host | Lookahead | MCU deployable |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `MetricGAN+ native8k causal_s` (QAT) | `8 kHz`, `val_select` | **2.8608** | **0.8538** | **13.3801** | **7.4184** | 0.00368 s / 10 s | 16 ms | Da |

Comparativ:

- vs `tiny_stm32_hybrid_sg`:
  `+0.2049 PESQ`
- vs `spectral_gating`:
  `+0.4847 PESQ`

### 5.4 MCU-uri suportate pentru metoda castigatoare

Re-auditata direct prin simulatorul curent, `MetricGAN+ native8k causal_s` trece toate profilele non-reference din shortlist:

- `STM32U5`
- `nRF54H20`
- `Apollo4 Blue+`
- `i.MX RT700`
- `STM32N6`
- `RA8P1`

Nu trece profilul de referinta `STM32L4`.

### 5.5 Rezumat per-chip pentru metoda castigatoare

| MCU profile | Real-time | Power < 50 mW | Recommended MHz | Avg. power (mW) | Verdict |
|---|---|---:|---:|---:|---|
| `Apollo4 Blue+` | Da | Da | 20 | **5.22** | Cel mai bun profil low-power |
| `nRF54H20` | Da | Da | 17 | 7.92 | Foarte bun |
| `STM32U5` | Da | Da | 23 | 7.96 | Foarte bun |
| `i.MX RT700` | Da | Da | 5 | 22.38 | Bun, dar overpowered pentru produs low-power |
| `STM32N6` | Da | Da | 4 | 30.18 | Bun pentru max-quality on MCU |
| `RA8P1` | Da | Da | 4 | 28.14 | Bun pentru max-quality on MCU |
| `STM32L4` | Nu | Da | 33 | 15.51 | Pica pe real-time |

### 5.6 Alternative deployable

Daca obiectivul se schimba de la `maxim PESQ` la `maxim conservatorism embedded`, alternativele sunt:

- `tiny_stm32_fc`
  - `PESQ = 2.5990`
  - trece inclusiv `STM32L4`
  - recomandat cand se cere compatibilitate cu MCU foarte restrans.
- `tiny_stm32_hybrid_sg`
  - `PESQ = 2.6559`
  - mai bun decat `tiny_stm32_fc`, dar nu la fel de bun ca `causal_s`.
- `spectral_gating`
  - `PESQ = 2.3762`
  - fallback clasic, cel mai simplu de portat si validat.

---

## 6. Concluzii

1. **Datasetul central** al proiectului este `VoiceBank+DEMAND`, cu ramura `16 kHz` pentru calitate si ramura `8 kHz` pentru deploy embedded.
2. **Cel mai bun baseline clasic** este `spectral_gating`.
3. **Cel mai bun baseline neuronal de referinta** este `MetricGAN+ raw`, cu `test PESQ = 3.1245` la `16 kHz`.
4. **Cel mai bun model ca acuratete bruta** pe `8 kHz` este `MetricGAN+ native8k`, dar acesta nu este deployable pe MCU.
5. **Cea mai buna metoda deployable** este in prezent `MetricGAN+ native8k causal_s` dupa `QAT`, cu `PESQ = 2.8608`, `STOI = 0.8538`, `SI-SDR = 13.3801`, `lookahead = 16 ms`, si compatibilitate real-time sub `50 mW` pe mai multe MCU-uri low-power moderne.
6. **Recomandarea de produs** este:
   - `Apollo4 Blue+` pentru consum minim;
   - `STM32U5` sau `nRF54H20` pentru un compromis foarte bun;
   - `STM32N6` / `i.MX RT700` doar daca obiectivul se muta spre `max quality on MCU` fara presiune serioasa de consum.
