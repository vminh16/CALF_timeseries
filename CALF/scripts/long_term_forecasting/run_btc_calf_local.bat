@echo off
setlocal enabledelayedexpansion

REM ==============================
REM Local CALF experiment script
REM Dataset: BTCUSDT 15m
REM GPU: RTX 2050 safe config
REM ==============================

set DATA_ROOT=./dataset/clean/
set DATA_FILE=BTCUSDT_15m_calf.csv

set SEQ_LEN=96
set LABEL_LEN=0

set BATCH_SIZE=128
set EPOCHS=100
set ITR=2

set D_MODEL=768
set D_FF=768
set ENC_IN=7
set DEC_IN=7
set C_OUT=7
set GPT_LAYERS=6

set LR=0.0001
set GPU_ID=0

set LOG_DIR=./logs_btc
if not exist %LOG_DIR% mkdir %LOG_DIR%

set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo ==============================
echo Start CALF BTCUSDT experiments
echo batch_size=%BATCH_SIZE%
echo epochs=%EPOCHS%
echo itr=%ITR%
echo ==============================

for %%P in (16 32 96) do (
    for %%A in (ori dropAttn_keepWE llm_to_attn llm_to_trsf) do (

        echo.
        echo ==========================================
        echo Running pred_len=%%P ablation=%%A
        echo ==========================================

        call python run.py ^
          --root_path %DATA_ROOT% ^
          --data_path %DATA_FILE% ^
          --is_training 1 ^
          --task_name long_term_forecast ^
          --model_id BTCUSDT_%SEQ_LEN%_%%P_%%A ^
          --data custom ^
          --features M ^
          --target close ^
          --freq 15min ^
          --seq_len %SEQ_LEN% ^
          --label_len %LABEL_LEN% ^
          --pred_len %%P ^
          --batch_size %BATCH_SIZE% ^
          --learning_rate %LR% ^
          --train_epochs %EPOCHS% ^
          --d_model %D_MODEL% ^
          --d_ff %D_FF% ^
          --enc_in %ENC_IN% ^
          --dec_in %DEC_IN% ^
          --c_out %C_OUT% ^
          --gpt_layers %GPT_LAYERS% ^
          --itr %ITR% ^
          --model GPT4TS ^
          --cos 1 ^
          --gpu %GPU_ID% ^
          --tmax 20 ^
          --patience 5 ^
          --bootstrap_eval 0 ^
          --num_workers 0 ^
          --use_amp ^
          --log_fine_name BTCUSDT_15m_result.txt ^
          > %LOG_DIR%/BTCUSDT_%SEQ_LEN%_%%P_%%A.log 2>&1

        if errorlevel 1 (
            echo Failed at pred_len=%%P ablation=%%A
            echo Check log: %LOG_DIR%/BTCUSDT_%SEQ_LEN%_%%P_%%A.log
            exit /b 1
        )

        echo Finished pred_len=%%P ablation=%%A
    )
)

echo.
echo All experiments finished.
endlocal