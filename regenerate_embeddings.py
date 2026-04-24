import csv
import json
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

INPUT_CSV = "faq_rows.csv"
OUTPUT_CSV = "faq_rows_new.csv"
BATCH_SIZE = 100
MODEL = "text-embedding-3-small"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": MODEL, "input": texts},
        timeout=60.0
    )
    response.raise_for_status()
    data = response.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def update_supabase(row_id: str, embedding: list[float]):
    httpx.patch(
        f"{SUPABASE_URL}/rest/v1/faq?id=eq.{row_id}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        },
        json={"embedding": embedding},
        timeout=30.0
    )


def main():
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"總共 {len(rows)} 筆，開始生成 embedding...")
    total = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        questions = [row["question"] for row in batch]
        print(f"處理 {i+1}~{min(i+BATCH_SIZE, total)}/{total}...")

        embeddings = get_embeddings(questions)

        for row, emb in zip(batch, embeddings):
            row["embedding"] = json.dumps(emb)

        # 每批次寫入 CSV，避免中途失敗遺失進度
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  已存至 {OUTPUT_CSV}")
        time.sleep(0.5)  # 避免 rate limit

    print(f"完成！請把 {OUTPUT_CSV} 改名成 faq_rows.csv 後重新 deploy。")


if __name__ == "__main__":
    main()
