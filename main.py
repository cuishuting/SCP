import transformers
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import numpy as np
import ast
import pickle
from model import KGTextCrossModel
from dataloader import CareerData, CareerDataCollator, CareerDataCollator_eval
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import os
import re
import subprocess
from safetensors import safe_open
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DGLBACKEND"] = "pytorch"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(pred):
    correct = 0
    num_val_data = len(pred.predictions[0]) // 20 # 20 is total length after padding
    print(f"val_dataset size: {num_val_data}")
    responses_ids = pred.predictions[0] 
    gold_ids = pred.predictions[1]
    
    for val_id in range(num_val_data):
        gen = responses_ids[val_id*20 : (val_id+1)*20]
        gold = gold_ids[val_id*20 : (val_id+1)*20]
        gen_text = llm_tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_space=False)
        gold_text = llm_tokenizer.decode(gold, skip_special_tokens=True, clean_up_tokenization_space=False)
        
        if gold_text in gen_text:
            correct += 1

    return {"hr1": correct / num_val_data}

def get_cpt_path(base_path):
    """
    Get the path of model.safetensors from the single checkpoint subdirectory.
    Assumes that base_path contains exactly one checkpoint directory.
    """
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if os.path.isdir(full_path) and name.startswith("checkpoint-"):
            return os.path.join(full_path, "model.safetensors")
    
    raise FileNotFoundError(f"No checkpoint found in {base_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("MyCareerModel")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--val_data_dir', type=str)
    parser.add_argument('--test_data_dir', type=str)
    parser.add_argument('--train_sg_dir', type=str)
    parser.add_argument('--val_sg_dir', type=str)
    parser.add_argument('--test_sg_dir', type=str)
    parser.add_argument('--gnn_cp_dir', type=str)
    parser.add_argument('--model_cp_dir', type=str)
    parser.add_argument('--gnn_h_dim', type=int)
    parser.add_argument('--gnn_num_heads', type=int)
    parser.add_argument('--gnn_dropout', type=float)
    parser.add_argument('--llm_path', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--cross_emb_dim', type=int)
    parser.add_argument('--fusion_encoder_hidden', type=int)
    parser.add_argument('--loss_balance_param', type=float)
    parser.add_argument('--num_comps', type=int)
    parser.add_argument('--num_jobs', type=int)
    

    args = parser.parse_args()
    desired_device = 0 # set your own gpu id
    print(f"desired device:{desired_device}")
    set_seed(args.seed)


    # load train_dataset & eval_dataset
    train_dataset = CareerData(data_dir=args.train_data_dir, sub_graphs_dir=args.train_sg_dir, llm_path=args.llm_path)
    val_dataset = CareerData(data_dir=args.val_data_dir, sub_graphs_dir=args.val_sg_dir, llm_path=args.llm_path)

    # load model
    model = KGTextCrossModel(
        GNN_cp_dir=args.gnn_cp_dir,
        gnn_in_dim=384, # static
        gnn_h_dim=args.gnn_h_dim, #512,
        num_heads=[args.gnn_num_heads],
        gnn_dropout=args.gnn_dropout,
        cross_emb_dim=args.cross_emb_dim,
        fusion_encoder_hidden=args.fusion_encoder_hidden,
        soft_prompt_dim=4096,
        desired_device=desired_device,
        loss_balance_param=args.loss_balance_param, # increase from 0.5->0.8 compared to former models
        llm_path=args.llm_path,
        num_comp_nodes_career_kg=args.num_comps,
        num_job_nodes_career_kg=args.num_jobs
    )
    model.to(f"cuda:{desired_device}")

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, additional_special_tokens=['<|HisJobEmb|>', '<|HisCompEmb|>'])
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    save_dir = os.path.join(args.model_cp_dir, f"SEED_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # define TrainingArguments
    args_trainer = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        optim="adamw_torch",
        weight_decay=0.01,
        gradient_accumulation_steps=64, 
        torch_empty_cache_steps=8, 
        learning_rate=args.lr,
        num_train_epochs=3,
        warmup_ratio=0.03,
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_hr1",
        greater_is_better=True,
        report_to="wandb"
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=args_trainer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=CareerDataCollator(),
    )

    trainer.train()

    """test begins"""
    model_weights_path = get_cpt_path(save_dir)

    model_weight_tensors = {}
    with safe_open(model_weights_path, framework="pt", device=desired_device) as f: 
        for k in f.keys():
            model_weight_tensors[k] = f.get_tensor(k)


    model.load_state_dict(model_weight_tensors)
    model.eval()

    test_dataset = CareerData(data_dir=args.test_data_dir, sub_graphs_dir=args.test_sg_dir, llm_path=args.llm_path, is_test=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=CareerDataCollator_eval())

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    valid_num = 0
    total_samples = len(test_dataset)

    for i, batch in enumerate(test_dataloader):
        
        candi_pairs_str = batch.pop('candi_pairs_str')
    
        output = model(**batch, is_test=True)
        candidate_texts = output['predictions']
        gold_answer = output['gold_answer']
        # print(f"check gold answer: {gold_answer}")
        # print(f"check candidate texts: {candidate_texts}")
        if gold_answer in candidate_texts[:1]:
            correct_top1 += 1
        if gold_answer in candidate_texts[:3]:
            correct_top3 += 1
        if gold_answer in candidate_texts:
            correct_top5 += 1
        # print(f"check candidate texts[0] in candi_pairs_str[0]: {candidate_texts[0] in candi_pairs_str[0]}")
        if candidate_texts[0] in candi_pairs_str[0]:
            valid_num += 1

    hr1 = correct_top1 / total_samples
    hr3 = correct_top3 / total_samples
    hr5 = correct_top5 / total_samples
    valid_ratio = valid_num / total_samples

    result_path = os.path.join(args.model_cp_dir, "result.txt")

    with open(result_path, "a") as f:
        f.write(f"SEED {args.seed}:\n")
        f.write(f"  HR@1        = {hr1:.4f}\n")
        f.write(f"  HR@3        = {hr3:.4f}\n")
        f.write(f"  HR@5        = {hr5:.4f}\n")
        f.write(f"  ValidRatio  = {valid_ratio:.4f}\n")
        f.write("=====================================\n")