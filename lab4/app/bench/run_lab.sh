#!/bin/bash
# Лабораторна 4, Частина 1: усі вимірювання одним запуском (macOS, Apple Silicon).
#   bash lab4/app/bench/run_lab.sh   (з кореня репозиторію або з будь-якої папки)
# Триває ~20–30 хв. Результати: lab4/bench_results.csv і lab4/app/results/ (hardware.txt, answers_*.md,
# memory_*.txt, test_results.md)
set -u
BENCH="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$BENCH")"; LAB="$(dirname "$APP")"
mkdir -p "$APP/results"; cd "$APP/results"
LOG=run_lab.log; exec > >(tee -a "$LOG") 2>&1
BEFORE=qwen3:4b-instruct-2507-q8_0
AFTER=qwen3:4b-instruct-2507-q4_K_M
PORT=11435   # окремий сервер з контрольованими налаштуваннями (не заважає застосунку Ollama на 11434)
URL=http://127.0.0.1:$PORT

echo "== 1. Залізо"
{
  echo "# system_profiler SPHardwareDataType (без серійного номера)"
  system_profiler SPHardwareDataType | grep -v -i -E "serial|UUID|Provisioning"
  echo "# GPU"
  system_profiler SPDisplaysDataType | grep -E "Chipset|Total Number of Cores|Metal|Type"
  echo "# sysctl"
  sysctl -n machdep.cpu.brand_string hw.memsize hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null
  echo "# GPU wired memory limit (iogpu.wired_limit_mb, 0 = дефолт ≈ 2/3 RAM)"; sysctl -n iogpu.wired_limit_mb 2>/dev/null
  sw_vers
} > hardware.txt
cat hardware.txt

echo "== 2. Ollama"
if ! command -v ollama >/dev/null; then
  echo "Ollama не встановлено. Встановіть: https://ollama.com/download (або brew install ollama) і запустіть скрипт ще раз."; exit 1
fi
ollama --version

echo "== 3. Python-оточення"
if [ ! -d "$LAB/.venv" ]; then python3 -m venv "$LAB/.venv" || { echo "Не вдалося створити venv"; exit 1; }; fi
. "$LAB/.venv/bin/activate"
pip install -q --upgrade pip >/dev/null; pip install -q openai || { echo "pip install openai не вдався"; exit 1; }

start_server() {  # $1 = опис, далі env
  echo "-- запускаю ollama serve на $PORT ($1)"
  shift
  env OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=10m "$@" ollama serve > "ollama_$PORT.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 30); do curl -s $URL/api/version >/dev/null && return 0; sleep 1; done
  echo "Сервер не стартував, див. ollama_$PORT.log"; exit 1
}
stop_server() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; sleep 2; }

echo "== 4. BEFORE: $BEFORE, дефолтні налаштування (NUM_PARALLEL=1, без flash attention, KV-кеш f16)"
start_server "before" OLLAMA_NUM_PARALLEL=1
OLLAMA_HOST=127.0.0.1:$PORT ollama pull $BEFORE
OLLAMA_HOST=127.0.0.1:$PORT ollama pull $AFTER
python "$BENCH/bench.py" --base-url $URL/v1 --model $BEFORE --label before --concurrency 1 2 4 --output "$LAB/bench_results.csv"
OLLAMA_HOST=127.0.0.1:$PORT ollama ps | tee memory_before.txt
python "$BENCH/eval_prompts.py" --base-url $URL/v1 --model $BEFORE --out answers_before.md
stop_server

echo "== 5. AFTER: $AFTER + NUM_PARALLEL=4 + flash attention + KV-кеш q8_0"
start_server "after" OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
python "$BENCH/bench.py" --base-url $URL/v1 --model $AFTER --label after --concurrency 1 2 4 --output "$LAB/bench_results.csv"
OLLAMA_HOST=127.0.0.1:$PORT ollama ps | tee memory_after.txt
python "$BENCH/eval_prompts.py" --base-url $URL/v1 --model $AFTER --out answers_after.md

echo "== 6. Тестові сценарії застосунку"
python "$APP/test_scenarios.py" --base-url $URL/v1 --model $AFTER
stop_server
echo "== ГОТОВО. Напишіть Claude «готово»."
