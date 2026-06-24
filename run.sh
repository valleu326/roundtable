#!/usr/bin/env bash
# 用法：source run.sh   （在已 conda activate roundtable 的终端里）
# 直接以当前环境拉起 Chainlit web 应用，并在子目录里跑，不影响你当前所在目录。
cd "$(dirname "${BASH_SOURCE[0]}")"
conda activate roundtable
chainlit run app.py
conda deactivate
