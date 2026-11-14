from datasets import load_dataset


def show_instance(target_id: str):
    # SWE-bench Verified をロード（2回目以降はローカルキャッシュから高速に読み込まれます）
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # 指定した instance_id に一致するデータを抽出
    matched = dataset.filter(lambda x: x["instance_id"] == target_id)

    if len(matched) == 0:
        print(f"指定された ID '{target_id}' は見つかりませんでした。")
        return

    item = matched[0]

    # 見やすく整形して出力
    print("=" * 60)
    print(f"Instance ID : {item['instance_id']}")
    print(f"Repository  : {item['repo']}")
    print(f"Base Commit : {item['base_commit']}")
    print(f"Created At  : {item['created_at']}")
    print("=" * 60)

    print("\n【問題文 (problem_statement)】")
    print(item["problem_statement"])

    # ヒントが存在する場合のみ表示
    if item.get("hints_text"):
        print("\n【ヒント (hints_text)】")
        print(item["hints_text"])

    print("\n【正解パッチ (patch)】")
    print(item["patch"])

    print("\n【テストパッチ (test_patch)】")
    print(item["test_patch"])


if __name__ == "__main__":
    # 確認したい instance_id を指定
    target_instance_id = "pydata__xarray-6992"
    show_instance(target_instance_id)
