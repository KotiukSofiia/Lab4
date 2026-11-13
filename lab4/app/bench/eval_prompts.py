"""Проганяє 10 типових для застосунку промптів через модель і зберігає відповіді в Markdown.

    python app/bench/eval_prompts.py --model qwen3:4b-instruct-2507-q8_0 --out app/results/answers_before.md
"""
import argparse, os, sys, time
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
from openai import OpenAI
from study_helper import SYSTEM_PROMPT, QUIZ_INSTRUCTIONS, PARAMS, parse_quiz

TEXT = open(os.path.join(APP_DIR, "sample_lecture.txt"), encoding="utf-8").read()
CTX = f'Навчальний текст:\n"""\n{TEXT}\n"""'

PROMPTS = [  # (режим, промпт, що вважаємо правильним)
    ("summary", "Зроби стислий конспект тексту: 5–7 маркованих пунктів, кожен — одне речення з ключовою думкою.", "5–7 пунктів, без вигаданих фактів"),
    ("ask", "Де саме в клітині відбувається фотосинтез?", "у хлоропластах (листки)"),
    ("ask", "Які дві стадії має фотосинтез і де кожна відбувається?", "світлова — мембрани тилакоїдів; темнова (цикл Кальвіна) — строма"),
    ("ask", "Що утворюється в результаті фотосинтезу?", "глюкоза і кисень"),
    ("ask", "Як називається ключовий фермент циклу Кальвіна?", "рубіско"),
    ("ask", "Від яких факторів залежить інтенсивність фотосинтезу?", "освітлення, концентрація CO2, температура"),
    ("ask", "Хто вперше відкрив фотосинтез і в якому році?", "«У тексті цього немає» (немає в тексті)"),
    ("ask", "Поясни простими словами для школяра 5 класу, навіщо рослинам хлорофіл.", "поглинає світло; просто й коректно"),
    ("ask", "Explain in English in two sentences what the Calvin cycle does.", "фіксує CO2 у глюкозу за рахунок АТФ/НАДФ·Н"),
    ("quiz", "QUIZ", "валідний JSON з 3 питаннями, 4 варіанти, правильні відповіді"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=180)
    lines = [f"# Відповіді моделі `{a.model}` на 10 типових промптів\n", f"Контекст — `app/sample_lecture.txt`.\n"]
    for i, (mode, prompt, expected) in enumerate(PROMPTS, 1):
        if mode == "quiz":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": QUIZ_INSTRUCTIONS.format(n=3, text=TEXT)}]
            prompt = "/quiz 3 (генерація тесту у JSON)"
        elif mode == "summary":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + "\n\n" + CTX}]
        else:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + CTX}, {"role": "user", "content": prompt}]
        t = time.perf_counter()
        r = client.chat.completions.create(model=a.model, messages=msgs, temperature=PARAMS[mode]["temperature"],
                                           max_tokens=PARAMS[mode]["max_tokens"])
        dt = time.perf_counter() - t
        ans = r.choices[0].message.content or ""
        extra = ""
        if mode == "quiz":
            try:
                extra = f"\n\n✅ JSON валідний, питань: {len(parse_quiz(ans))}"
            except ValueError as e:
                extra = f"\n\n❌ JSON невалідний: {e}"
        toks = r.usage.completion_tokens if r.usage else "?"
        lines.append(f"\n## {i}. {prompt}\n\n*Очікується:* {expected}  \n*Час:* {dt:.1f} с, токенів: {toks}\n\n{ans.strip()}{extra}\n")
        print(f"[{i}/10] {dt:.1f} s")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print("saved", a.out)


if __name__ == "__main__":
    main()
