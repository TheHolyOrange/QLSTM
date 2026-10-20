================================================================================
 QLSTM — Variational Quantum LSTM for Time-Series Forecasting
 README
================================================================================

--------------------------------------------------------------------------------
A. DATASET SOURCE / URL
--------------------------------------------------------------------------------

Two datasets are used, both already included under data/ so no download is
required to run the code.

1. ETTh1 (Electricity Transformer Temperature, hourly)
   File   : data/ETTh1.csv
   Source : https://github.com/zhouhaoyi/ETDataset
   Content: Hourly readings (July 2016 - June 2018) of six load channels
            (HUFL, HULL, MUFL, MULL, LUFL, LULL) and the oil temperature
            (OT) of an electricity transformer. OT is the forecasting
            target. Only the most recent 3,500 hours are used (quantum
            circuit simulation is expensive), chosen by
            src/preprocessing.py.

2. UniELoad (university campus substation load + weather, hourly)
   File   : data/GSTU_Data_Final_With_Weather.csv
   Source : Mendeley Data, v3 - https://data.mendeley.com/datasets/d8r95wmzms/3
   Content: Hourly electrical measurements (voltage, three-phase current,
            power factor) and weather data for a university campus
            substation. The target, active load in kW, is computed from
            the raw measurements by src/unieload_preprocessing.py
            (three-phase power formula). As with ETTh1, the most recent
            3,500 hours are used.

--------------------------------------------------------------------------------
B. SOFTWARE AND HARDWARE REQUIREMENTS
--------------------------------------------------------------------------------

Software
--------
  - Operating system : Windows, Linux or macOS (no OS-specific code; the
                        quantum simulator is pure Python/CPU).
  - Python           : 3.10 or later (developed and tested with Python 3.13).
  - Python packages   : listed in requirements.txt -
                           torch        >= 2.13
                           pennylane    >= 0.45
                           pandas       >= 3.0
                           scikit-learn >= 1.9
                           matplotlib   >= 3.11
                           optuna       >= 4.9
                           numpy        >= 2.5
                           scipy        >= 1.18
                        Install all of them with:
                           pip install -r requirements.txt

Hardware
--------
  - CPU   : A modern multi-core CPU. All quantum-circuit simulation
            (PennyLane's "default.qubit" statevector simulator) runs on
            CPU only - this is PennyLane's own limitation, not a choice
            made by this code, so a GPU gives no speed-up for the QLSTM
            parts and is not required.
  - GPU   : Not required. (If a CUDA GPU is present, the purely classical
            LSTM baseline may use it automatically via PyTorch, but the
            QLSTM's quantum layers are pinned to CPU regardless.)
  - RAM   : 8 GB or more is comfortable. The datasets and models are small
            (a few MB of CSV data, under 1 MB per model checkpoint).
  - Disk  : ~50 MB for the code and data, plus a few hundred MB of free
            space for checkpoints, logs and plots written during training.
  - Time  : Quantum circuit simulation on CPU is the main cost, not the
            model size. Approximate wall-clock times on a typical laptop
            CPU, for reference:
               run_pipeline.py              : ~2-2.5 hours
                 (Optuna search ~75-80 min + final training ~50-55 min)
               run_classical_baseline.py    : a few seconds
               run_unieload.py lstm         : a few seconds
               run_unieload.py qlstm        : ~2-2.5 hours
               run_unieload_qlstm_v2.py     : ~5-6 hours
            All four scripts checkpoint after every epoch and resume
            automatically if interrupted and re-run, so a run does not
            have to complete in one sitting.

--------------------------------------------------------------------------------
C. HOW TO EXECUTE THE SOURCE CODE
--------------------------------------------------------------------------------

1) Set up the environment (once)
   ------------------------------
   From the project's root folder (the folder this file is in):

       pip install -r requirements.txt

2) Run the scripts, IN THIS ORDER
   --------------------------------
   The later scripts read configuration/results written by the earlier
   ones, so they must be run in this order the first time. All commands
   are run from the project's root folder.

   Step 1 - ETTh1: hyperparameter search + train the QLSTM
       python run_pipeline.py

       What it does: preprocesses ETTh1, runs an 8-trial Optuna search
       (4 epochs/trial) over qubit count / layers / hidden size /
       learning rate, retrains the best configuration for up to 30
       epochs with early stopping, and evaluates it on the test set.

       Output: results/model_config.json, results/metrics.json,
       results/hpo_results.json, results/checkpoints/qlstm_best.pt,
       results/plots/loss_curve.png, results/plots/predictions_vs_actual.png,
       results/training_log.txt.

   Step 2 - ETTh1: train the classical LSTM baseline
       python run_classical_baseline.py

       What it does: trains a classical LSTM on ETTh1 with the same
       hidden size that Step 1 selected for the QLSTM, under an
       identical training harness, for a fair comparison.

       Output: results/classical_lstm_metrics.json,
       results/checkpoints/classical_lstm_best.pt,
       results/plots/classical_lstm_loss_curve.png,
       results/plots/classical_lstm_predictions_vs_actual.png,
       results/plots/qlstm_vs_classical_vs_actual.png,
       results/classical_lstm_log.txt.

   Step 3 - UniELoad: train the classical LSTM
       python run_unieload.py lstm

   Step 4 - UniELoad: train the QLSTM
       python run_unieload.py qlstm

       Both Step 3 and Step 4 reuse the ETTh1-selected configuration
       from results/model_config.json (no new search is run on
       UniELoad). Output goes to results/unieload/ (metrics, logs,
       plots, checkpoints for each model).

   Step 5 - UniELoad: combined comparison plot (optional)
       python run_unieload.py compare

       Builds a side-by-side plot and summary table from the
       checkpoints saved in Steps 3-4. Run this only after both.

   Step 6 - UniELoad: QLSTM with LR schedule + gradient clipping
       python run_unieload_qlstm_v2.py

       What it does: retrains the QLSTM on UniELoad with a
       reduce-on-plateau learning-rate schedule and gradient clipping,
       which fixes an oscillating-validation-loss issue seen with a
       fixed learning rate in Step 4. Needs results/model_config.json
       (Step 1) and, for its comparison plot, results/unieload/
       lstm_metrics.json (Step 3).

       Output: results/unieload/qlstm_v2_metrics.json,
       results/unieload/checkpoints/qlstm_v2_*,
       results/unieload/plots/qlstm_v2_*.

   Step 7 - UniELoad: quantum-circuit diagnostics (optional)
       python -m src.quantum_analysis barren
       python -m src.quantum_analysis landscape
       python -m src.quantum_analysis noise
       python -m src.quantum_analysis metrics

       What it does: analyses the QLSTM gate circuit trained in Step 4
       (results/unieload/checkpoints/qlstm_best.pt) - "barren" scans
       gradient variance vs. qubit count/depth (barren plateaus),
       "landscape" samples a 2D slice of the validation-loss landscape
       around the trained parameters, "noise" evaluates accuracy under
       finite-shot and depolarising noise, and "metrics" computes
       expressibility and Meyer-Wallach entangling capability. The
       results already committed under results/unieload/analysis/ were
       produced this way; re-running is only needed to reproduce or
       extend them.

3) Resuming an interrupted run
   -----------------------------
   Every training script saves a checkpoint (model, optimiser, training
   history) after every epoch. If a script is stopped partway through
   (e.g. Ctrl+C, machine restart), simply re-run the exact same command;
   it will detect the existing checkpoint under results/ and resume
   from the last completed epoch instead of starting over. The same
   applies to the Optuna search in Step 1 (results/optuna_study.db).

4) Where to look at the results
   -------------------------------
   - results/metrics.json, results/classical_lstm_metrics.json,
     results/unieload/*_metrics.json : RMSE / MAE / MAPE, parameter
     counts and training time for each model and dataset.
   - results/plots/, results/unieload/plots/ : loss curves and
     forecast-vs-actual plots (PNG).
   - results/unieload/analysis/ : quantum-circuit diagnostics for the
     trained UniELoad QLSTM - expressibility, entangling capability,
     barren-plateau scan, loss landscape and noise robustness
     (JSON, produced by src/quantum_analysis.py).

================================================================================
