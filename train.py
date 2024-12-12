from tqdm import tqdm
import evaluate
import pandas as pd
from dataset import IAMDataset
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW

cer_metric = evaluate.load("cer")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


def main():

    df = pd.read_fwf("./dataset/IAM/gt_test_small.txt", header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    df["file_name"] = df["file_name"].apply(
        lambda x: x + "g" if x.endswith("jp") else x
    )
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=25)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_dataset = IAMDataset(
        root_dir="./dataset/IAM/image/", df=train_df, processor=processor
    )
    eval_dataset = IAMDataset(
        root_dir="./dataset/IAM/image/", df=test_df, processor=processor
    )
    print("Number of training", len(train_dataset))
    print("Number of validation", len(eval_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 32
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))

            # evaluate
            model.eval()
            valid_cer = 0.0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    outputs = model.generate(batch["pixel_values"].to(device))
                    # compute metrics
                    cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                    valid_cer += cer

            print("Validation CER:", valid_cer / len(eval_dataloader))

    model.save_pretrained(".")


if __name__ == "__main__":
    main()
