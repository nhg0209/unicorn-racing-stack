#!/usr/bin/env bash
# libcartographer_ros.so의 collect_metrics gflags 중복 정의 패치
# 사용법: conda activate unicorn && bash patch_cartographer_so.sh

set -euo pipefail
: "${CONDA_PREFIX:?conda env 미활성화. 'conda activate unicorn' 먼저.}"

SO_PATH="$CONDA_PREFIX/lib/libcartographer_ros.so"
[ -f "$SO_PATH" ] || { echo "ERROR: $SO_PATH 없음"; exit 1; }

if grep -aq "collect_metricz" "$SO_PATH"; then
echo "[skip] 이미 패치됨: $SO_PATH"
exit 0
fi

cp -n "$SO_PATH" "${SO_PATH}.bak"
LC_ALL=C sed -i 's/collect_metrics/collect_metricz/g' "$SO_PATH"
echo "[done] 패치 완료: $SO_PATH"
echo "        백업:    ${SO_PATH}.bak"