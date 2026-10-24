"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import mlflow
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai.types import GenerateContentConfig
from typing import Callable, List, Dict

from app.config import MODEL, api_error_message, get_api_key, get_tracking_server_url, get_temperature

load_dotenv()

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

tracking_server_url = get_tracking_server_url()
if tracking_server_url:
    mlflow.gemini.autolog()
    mlflow.set_tracking_uri(tracking_server_url)


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    temperature = get_temperature()
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=GenerateContentConfig(
            temperature=temperature
        )
    )

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）
    usage = response.usage_metadata
    print(f"temperature: {temperature}")
    print(f"入力トークン数: {usage.prompt_token_count}")
    print(f"出力トークン数: {usage.candidates_token_count}")
    print(f"合計トークン数: {usage.total_token_count}")

    return response.text


def chat(renderer: Callable[[str], None], greeting: str = "") -> List[Dict[str, str]]:
    client = genai.Client(api_key=get_api_key())
    session_id = f"{uuid.uuid4().hex}"
    renderer(f"セッション開始： {session_id}")
    history = []
    with mlflow.tracing.context(session_id=session_id, user="default"):
        chat = client.chats.create(model=MODEL)
        if greeting:
            renderer(f"{greeting}\n")
        while True:
            query = input("> ")
            if not query:
                continue
            if query.strip().lower() in ["/quit", "/exit"]:
                break
            response = chat.send_message(query)
            renderer(f"{response.text}\n")
            history.append({"query": query, "response": response.text})

    renderer(f"セッション終了: {session_id}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "-c", "--chat", help="チャットの開始", action="store_true")
    args = parser.parse_args()

    try:
        print(f"[model] {MODEL}")
        if args.chat:
            chat(greeting="Hello!", renderer=lambda s: print(s))
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
