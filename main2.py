import json
with open("./test.json", "r") as f:
  test = json.load(f)
import openai
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from time import time
import pandas as pd
import time


test_data = json.load(open('test.json'))
openai.api_key = json.load(open(".openai_api_key.json"))["key"]

start_idx = 842
summary_results = [str(i) for i in range(start_idx, len(test), 4)] # 여기에 test document에 대한 summary를 넣어주세요
CORE_NUM = 8

def split_text(text):
    max_chunk_size = 1024 
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_summary(text):
    input_chunks = split_text(text)
    output_chunks = []
    for chunk in input_chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": chunk+"\n한글로 요약해줘 :"}],
            temperature=0.7,
            max_tokens=1024,
            n = 1,
            stop=None
        )
        summary = str(response['choices'][0]['message']['content']).strip()
        output_chunks.append(summary)
    return " ".join(output_chunks)

def api_call(multi):
    for test_set_idx in tqdm(range(0, len(test_data[start_idx:]), CORE_NUM)):
        test_set = test_data[start_idx + multi + test_set_idx]
        while True:
            try:
                summary_results[multi + test_set_idx] = generate_summary(test_set.get("document"))
                break
            except Exception as e:
                print(f"ERROR! In {start_idx + test_set_idx} : {e}")
                time.sleep(5)

        print(start_idx + multi + test_set_idx)
        print(summary_results[multi + test_set_idx])

def main():
    fe_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    fe_model = AutoModel.from_pretrained("monologg/kobert")
    pooler_outputs = []
    fe_model.eval()

    for s in tqdm(summary_results):
        input_ids = fe_tokenizer(s, return_tensors="pt").input_ids
        pooler_output = fe_model(input_ids).pooler_output.squeeze(0)
        pooler_outputs.append(pooler_output) 

    pooler_outputs = torch.stack(pooler_outputs)
    row_id = [i for i in range(768*start_idx, 768*1500, CORE_NUM)]
    group_id = []
    for i in range(start_idx, 1500, CORE_NUM):
        for _ in range(768):
            group_id.append(i)
    pooler_outputs = pooler_outputs.reshape(-1)
    pooler_outputs = pooler_outputs.detach().numpy()
    pred = pd.DataFrame({"row_id": row_id, "val": pooler_outputs.T})
    pred.set_index("row_id")

    pred.to_csv('./pred_1.csv', index=False)
    print("SUCCESS!")

if __name__ == "__main__":
    import multiprocessing
    pool = multiprocessing.Pool(processes=CORE_NUM)
    pool.map(api_call, [i for i in range(CORE_NUM)])
    pool.close()
    pool.join()
    main()
