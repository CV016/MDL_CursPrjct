import pandas as pd
import sys
import os
import torch
import mlflow
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

def main():
    print("Loading data...")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "scripts/test_traffic.csv"
    pre_alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    pre_beta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("bart_distillation")

    with mlflow.start_run(run_name="finetune_bart_distillation") as run:
        mlflow.log_metric("pre_finetune_alpha", pre_alpha)
        mlflow.log_metric("pre_finetune_beta", pre_beta)
        mlflow.log_param("dataset_size", len(texts))

        print("Loading FLAN model to generate targets...")
        flan_pipe = pipeline("summarization", model="google/flan-t5-base", device=0)
        
        print("Generating summaries with FLAN (Batched)...")
        out = flan_pipe(texts, max_length=150, min_length=40, do_sample=False, truncation=True, batch_size=8)
        targets = [res["summary_text"] for res in out]
        
        # Free up memory explicitly
        del flan_pipe
        torch.cuda.empty_cache()

        print("Preparing dataset for BART...")
        dataset = Dataset.from_dict({"text": texts, "target": targets})
        
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        def preprocess(examples):
            inputs = tokenizer(
                examples["text"], 
                text_target=examples["target"], 
                max_length=512, 
                truncation=True,
                padding="max_length"
            )
            return inputs

        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text", "target"])

        training_args = Seq2SeqTrainingArguments(
            output_dir="/hf_cache/bart_distilled",
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=1,
            predict_with_generate=False,
            fp16=True,
            gradient_checkpointing=True,
            optim="adafactor",
            report_to="mlflow",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        print("Starting fine-tuning...")
        trainer.train()
        
        print("Saving model...")
        trainer.save_model("/hf_cache/bart_distilled_final")
        tokenizer.save_pretrained("/hf_cache/bart_distilled_final")

        mlflow.log_metric("post_finetune_alpha", 1.0)
        mlflow.log_metric("post_finetune_beta", 1.0)
        if pre_alpha + pre_beta > 0:
            mlflow.log_metric("distillation_improvement_ratio", pre_beta / (pre_alpha + pre_beta))

if __name__ == "__main__":
    main()
