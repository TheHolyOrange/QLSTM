# QLSTM — Variational Quantum LSTM for Time-Series Forecasting

A quantum LSTM (QLSTM) in which each of the four classical LSTM gates is
replaced by a variational quantum circuit (VQC), evaluated against a
capacity-matched classical LSTM baseline on two hourly forecasting datasets:
**ETTh1** (electricity transformer oil temperature) and **UniELoad**
(university campus substation load).

Reference architecture adapted from
[QCL-PKNU/SPP-QLSTM](https://github.com/QCL-PKNU/SPP-QLSTM)
(Kea et al., *A Hybrid Quantum-Classical Model for Stock Price Prediction
Using Quantum-Enhanced Long Short-Term Memory*, Entropy 2024), generalized
here to arbitrary feature counts, batched quantum circuit execution, and a
train/val/test pipeline with proper scaling and early stopping.

## Architecture

```
concat(h_{t-1}, x_t) --Linear--> y_t (dim = n_qubits)
y_t --AngleEmbedding + entangling/rotation ansatz (VQC)--> quantum expectations
quantum expectations --Linear (shared across gates)--> gate pre-activation (dim = hidden_size)
gate pre-activation --sigmoid/tanh--> f_t, i_t, g_t, o_t
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

Each gate has its own VQC (angle encoding + CNOT-ring entanglement at
offsets 1 and 2 + trainable `R_X R_Y R_Z` rotations, `n_qlayers` times), but
all four gates share one output projection back to classical space.

## Structure

```
data/ETTh1.csv                     raw ETTh1 dataset (zhouhaoyi/ETDataset)
data/GSTU_Data_Final_With_Weather.csv   raw UniELoad dataset (Mendeley d8r95wmzms/3)
src/preprocessing.py                ETTh1: cyclical time features, chronological split, scaling, windowing
src/unieload_preprocessing.py       UniELoad: target construction, same split/scale/window harness
src/qlstm_model.py                  QLSTM cell (4-gate VQC) + regression head
src/lstm_model.py                   classical LSTM baseline, same interface
src/train.py                        shared train/eval loop, early stopping, optional LR schedule + grad clip
src/hyperparam_search.py            Optuna search over qubits/layers/hidden size/lr (ETTh1)
src/unieload_qlstm_tune.py          QLSTM lr/qubit/layer search on UniELoad
src/quantum_analysis.py             expressibility, entangling capability, barren-plateau scan,
                                     loss landscape, shot/depolarising-noise robustness
src/utils.py                        seeding, RMSE/MAE/MAPE
run_pipeline.py                     end-to-end on ETTh1: preprocess -> HPO -> final training -> evaluation -> plots
run_classical_baseline.py           classical LSTM on ETTh1, same harness and hidden size as QLSTM
run_unieload.py                     LSTM + QLSTM on UniELoad, reusing the ETTh1-selected config
run_unieload_qlstm_v2.py            QLSTM-S on UniELoad: adds LR-on-plateau schedule + gradient clipping
results/                            metrics.json, hpo_results.json, model_config.json, plots/, checkpoints/
results/unieload/                   UniELoad metrics/plots/checkpoints + analysis/ (quantum diagnostics JSONs)
```

## Reproducing

```bash
pip install -r requirements.txt
python run_pipeline.py              # QLSTM + HPO on ETTh1 -> results/
python run_classical_baseline.py    # classical LSTM on ETTh1 -> results/
python run_unieload.py              # LSTM + QLSTM on UniELoad -> results/unieload/
python run_unieload_qlstm_v2.py     # QLSTM-S (scheduler + grad clip) on UniELoad
```

Key settings (see top of `run_pipeline.py`): 3,500-hour subsample, 24-hour
lookback window, 8-trial / 4-epoch Optuna search over
`n_qubits∈{4,6}, n_qlayers∈{1,2}, hidden_size∈{8,16}, lr∈{0.005,0.01}`, final
training up to 30 epochs with early stopping (patience 8). These keep
quantum circuit simulation (PennyLane `default.qubit`) tractable on CPU.

## Results

Selected configuration: `n_qubits=6, n_qlayers=2, hidden_size=8, lr=0.01`
(329 trainable parameters vs. 681 for the classical LSTM at the same hidden
size).

| Dataset | Model | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|---|
| ETTh1 (°C) | LSTM | 0.632 | 0.402 | 5.15% |
| ETTh1 (°C) | QLSTM | 0.592 | 0.391 | 4.89% |
| UniELoad (kW) | LSTM | 30.66 | 23.89 | 9.63% |
| UniELoad (kW) | QLSTM | 31.09 | 24.88 | 9.79% |
| UniELoad (kW) | QLSTM-S | 30.50 | 24.00 | 9.55% |

`src/quantum_analysis.py` additionally reports, for the trained UniELoad
QLSTM gate circuit: expressibility (KL-to-Haar 0.0020), Meyer–Wallach
entangling capability (0.947 ± 0.035), a barren-plateau gradient-variance
scan vs. qubit count/depth, a 2D loss-landscape slice around the trained
parameters, and robustness to finite-shot and depolarising-noise simulation
— see `results/unieload/analysis/`.
