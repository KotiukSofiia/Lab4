"""Бенчмарк OpenAI-сумісного сервера з мовною моделлю.

Надсилає стрімінгові запити з різною кількістю паралельних клієнтів і для кожного рівня
рахує TTFT (p50 і p95), швидкість генерації на одного користувача й throughput сервера.
Працює з будь-яким OpenAI-сумісним сервером: Ollama, llama-server, LM Studio, vLLM
або вашим власним бекендом.

Встановлення:
    pip install openai

Приклади:
    # Ollama
    python bench.py --base-url http://localhost:11434/v1 --model qwen3:1.7b --reasoning-effort none

    # llama-server, з міткою для порівняння до й після оптимізації
    python bench.py --base-url http://localhost:8080/v1 --model local --label before
    python bench.py --base-url http://localhost:8080/v1 --model local --label after

Результати друкуються таблицею й дописуються в CSV (за замовчуванням bench_results.csv),
тож кілька запусків з різними --label потрапляють в один файл.
"""

import argparse
import concurrent.futures
import csv
import os
import time

from openai import OpenAI

DEFAULT_PROMPTS = [
    "Яка столиця Австралії?",
    "Поясни в трьох реченнях, що таке квантизація моделі.",
    "Напиши короткий лист викладачеві з проханням перенести здачу лабораторної на тиждень.",
    "Explain in two sentences why the sky is blue.",
    "Склади план підготовки до іспиту з програмування на 5 днів.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Перелічи три переваги й три недоліки локального запуску моделей.",
    "Translate into Ukrainian: The meeting has been moved to Thursday.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Бенчмарк OpenAI-сумісного сервера з мовною моделлю.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="адреса API, разом із /v1")
    parser.add_argument("--model", required=True, help="назва моделі на сервері")
    parser.add_argument("--api-key", default="local", help="ключ API; локальні сервери його зазвичай не перевіряють")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="кількість паралельних клієнтів для кожного рівня")
    parser.add_argument("--requests", type=int, default=16, help="кількість запитів на кожен рівень")
    parser.add_argument("--max-tokens", type=int, default=200, help="максимум токенів у відповіді")
    parser.add_argument("--prompts", help="файл із промптами, по одному на рядок (інакше — вбудований набір)")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"],
                        help='для моделей з міркуванням (напр. qwen3 в Ollama) передайте "none", '
                             "інакше TTFT включатиме час міркування")
    parser.add_argument("--timeout", type=float, default=120, help="таймаут одного запиту, секунди")
    parser.add_argument("--label", default="", help="мітка запуску в CSV, напр. before / after")
    parser.add_argument("--output", default="bench_results.csv", help="CSV, у який дописуються результати")
    return parser.parse_args()


def percentile(values, q):
    """Перцентиль з лінійною інтерполяцією, як numpy.percentile."""
    values = sorted(values)
    position = (len(values) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def measure(client, args, prompt):
    """Один стрімінговий запит; повертає TTFT, tokens/s, кількість токенів і чи модель міркувала."""
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    start = time.perf_counter()
    first = usage = None
    reasoning = False
    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            # Ollama надсилає міркування в полі reasoning, llama-server і vLLM — у reasoning_content
            if (delta.model_extra or {}).get("reasoning") or (delta.model_extra or {}).get("reasoning_content"):
                reasoning = True
            if first is None and delta.content:
                first = time.perf_counter()
        if chunk.usage:
            usage = chunk.usage
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("модель не повернула жодного токена в content — "
                           "якщо вона міркує, запустіть з --reasoning-effort none")
    tokens = usage.completion_tokens if usage else None
    return {
        "ttft": first - start,
        "tokens": tokens,
        "tokens_per_s": (tokens - 1) / (end - first) if tokens and end > first else None,
        "reasoning": reasoning,
    }


def load_test(client, args, prompts, concurrency):
    batch = [prompts[i % len(prompts)] for i in range(args.requests)]
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(measure, client, args, p) for p in batch]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
    wall = time.perf_counter() - start

    if not results:
        raise SystemExit(f"Усі запити завершилися помилкою, остання: {errors[-1]}")
    ttft = [r["ttft"] for r in results]
    speeds = [r["tokens_per_s"] for r in results if r["tokens_per_s"]]
    tokens = sum(r["tokens"] or 0 for r in results)
    return {
        "label": args.label,
        "model": args.model,
        "concurrency": concurrency,
        "requests": args.requests,
        "errors": len(errors),
        "ttft_p50_s": round(percentile(ttft, 50), 3),
        "ttft_p95_s": round(percentile(ttft, 95), 3),
        "tokens_per_s_per_user_p50": round(percentile(speeds, 50), 1) if speeds else None,
        "throughput_tokens_per_s": round(tokens / wall, 1),
        "reasoning": any(r["reasoning"] for r in results),
    }


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    print(f"Прогрів: {args.model} на {args.base_url}")
    measure(client, args, "Привіт!")  # перший запит включає завантаження моделі — не рахуємо

    rows = []
    header = f"{'клієнтів':>9} {'помилок':>8} {'TTFT p50':>9} {'TTFT p95':>9} {'tok/s на кор.':>14} {'throughput':>11}"
    print(header)
    for concurrency in args.concurrency:
        row = load_test(client, args, prompts, concurrency)
        rows.append(row)
        print(f"{row['concurrency']:>9} {row['errors']:>8} {row['ttft_p50_s']:>8.2f}с {row['ttft_p95_s']:>8.2f}с "
              f"{row['tokens_per_s_per_user_p50'] or 0:>14.1f} {row['throughput_tokens_per_s']:>11.1f}")
    if any(row["reasoning"] for row in rows):
        print("УВАГА: модель міркувала перед відповіддю. TTFT включає час міркування, а tokens/s завищений, "
              "бо токени міркування теж пораховано. Для коректних вимірювань запустіть з --reasoning-effort none.")

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Результати дописано в {args.output}")


if __name__ == "__main__":
    main()
