import csv
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_CSV = "faq_rows.csv"
OUTPUT_CSV = "faq_rows_new.csv"
BATCH_SIZE = 64
MODEL_NAME = "text-embedding-3-small"


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"總共 {len(rows)} 筆，使用 OpenAI {MODEL_NAME} 生成 embedding...")
    total = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        questions = [row["question"] for row in batch]
        print(f"處理 {i+1}~{min(i+BATCH_SIZE, total)}/{total}...")

        response = client.embeddings.create(model=MODEL_NAME, input=questions)
        embeddings = [d.embedding for d in response.data]

        for row, emb in zip(batch, embeddings):
            row["embedding"] = json.dumps(emb)

        # 每批次寫入 CSV，避免中途失敗遺失進度
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  已存至 {OUTPUT_CSV}")

    print(f"完成！請把 {OUTPUT_CSV} 改名成 faq_rows.csv 後重新 deploy。")


if __name__ == "__main__":
    main()
