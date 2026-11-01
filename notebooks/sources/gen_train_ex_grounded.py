# train_grounded.jsonl の各質問を言い回しを変えた 3 問ずつ増幅し、
# train_ex_grounded.jsonl を生成する。回答（assistant の content）は原文と全く同じにする。
# 使い方: python gen_train_ex_grounded.py （この sources/ ディレクトリで実行）
#
# 注意: test/valid に使われている言い回し（例「Xについて簡単に説明してください。」）と
# 重複しないよう、生成した全質問を既存ファイルと突合してチェックする。

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "train_grounded.jsonl"
OUT = HERE / "train_ex_grounded.jsonl"
N_EX = 3  # 各質問ごとの増幅数

# 元の「「X」と是何ですか？」とも、test/valid の「Xについて簡単に説明してください。」とも
# 違う言い回し。連続する語で同じ組み合わせが続かないよう 6 種類を回して使う。
PATTERNS = [
    "「{t}」とはどういうものですか？",
    "「{t}」の意味を教えてください。",
    "「{t}」について教えてください。",
    "「{t}」とはどのようなものですか？",
    "「{t}」を説明してください。",
    "「{t}」が何なのか教えてください。",
]


def load_questions(name):
    qs = set()
    with open(HERE / name, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qs.add(json.loads(line)["messages"][0]["content"])
    return qs


def main():
    originals = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                originals.append(json.loads(line))

    # 既存の質問（元 train、test、valid）と重複してはいけない
    used = (
        {d["messages"][0]["content"] for d in originals}
        | load_questions("test_grounded.jsonl")
        | load_questions("valid_grounded.jsonl")
    )

    out_lines = []
    for i, d in enumerate(originals):
        term = re.fullmatch(r"「(.+)」とは何ですか？", d["messages"][0]["content"]).group(1)
        assistant = d["messages"][1]  # 回答はそのまま流用
        for j in range(N_EX):
            p = PATTERNS[(3 * i + j) % len(PATTERNS)]
            q = p.format(t=term)
            assert q not in used, f"言い回しが既存と重複: {q}"
            used.add(q)
            row = {"messages": [{"role": "user", "content": q}, assistant]}
            out_lines.append(json.dumps(row, ensure_ascii=False))

    OUT.write_text("\n".join(out_lines), encoding="utf-8", newline="\n")

    # 検証: 件数・回答同一・重複なし
    assert len(out_lines) == len(originals) * N_EX
    with open(OUT, encoding="utf-8") as f:
        gen = [json.loads(line) for line in f if line.strip()]
    assert len(gen) == len(originals) * N_EX
    for k, d in enumerate(originals):
        for j in range(N_EX):
            g = gen[k * N_EX + j]
            assert g["messages"][1] == d["messages"][1], f"回答が一致しない: {k}"
    assert len({g["messages"][0]["content"] for g in gen}) == len(gen), "質問が重複"
    print(f"OK: {len(originals)} 質問 × {N_EX} = {len(gen)} 件 → {OUT.name}")
    for g in gen[:6]:
        print("  ", g["messages"][0]["content"])


if __name__ == "__main__":
    main()
