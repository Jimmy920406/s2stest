import csv
import json
from sentence_transformers import SentenceTransformer

INPUT_CSV = "faq_rows.csv"
OUTPUT_CSV = "faq_rows_new.csv"
BATCH_SIZE = 64
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    model = SentenceTransformer(MODEL_NAME)

    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"總共 {len(rows)} 筆，使用本地模型 {MODEL_NAME} 生成 embedding...")
    total = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        questions = [row["question"] for row in batch]
        print(f"處理 {i+1}~{min(i+BATCH_SIZE, total)}/{total}...")

        embeddings = model.encode(questions, normalize_embeddings=True, batch_size=BATCH_SIZE)

        for row, emb in zip(batch, embeddings):
            row["embedding"] = json.dumps(emb.tolist())

        # 每批次寫入 CSV，避免中途失敗遺失進度
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  已存至 {OUTPUT_CSV}")

    print(f"完成！請把 {OUTPUT_CSV} 改名成 faq_rows.csv 後重新 deploy。")


if __name__ == "__main__":
    main()
