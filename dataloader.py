import torch.utils.data as data
import pandas as pd
import ast
import re
from dgl.data.utils import load_graphs
import torch
from typing import Dict, Optional, Sequence, List
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig, BertModel, BertTokenizer


class CareerData(data.Dataset):
    def __init__(self, data_dir, sub_graphs_dir, llm_path, is_test=False):
        self.df = pd.read_csv(data_dir)
        self.df['candi_pairs_20'] = self.df['candi_pairs_20'].apply(ast.literal_eval)
        self.df['his_jobs'] = self.df['his_jobs'].apply(ast.literal_eval)
        self.df['his_jobs'] = self.df['his_jobs'].apply(lambda x: [job.strip() for job in x if job])

        self.df['his_comps'] = self.df['his_comps'].apply(ast.literal_eval)
        self.df['his_comps'] = self.df['his_comps'].apply(lambda x: [comp.strip() for comp in x if comp])


        self.df['his_jobs_id_list'] = self.df['his_jobs_id_list'].apply(ast.literal_eval)
        self.df['his_comps_id_list'] = self.df['his_comps_id_list'].apply(ast.literal_eval)
        self.graphs_dir = sub_graphs_dir

        self.text_input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, additional_special_tokens=['<|HisJobEmb|>', '<|HisCompEmb|>'])
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.is_test = is_test

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i):
        sample = self.df.loc[i]
        candi_pairs = sample['candi_pairs_20']
        candi_pairs = [eval(pair) for pair in candi_pairs]
        # list of varied-length input text tokens 
        tokenized_his_jobs = [self.text_input_tokenizer(job, return_tensors='pt', padding=True, truncation=True) for job in sample['his_jobs']]
        tokenized_his_comps = [self.text_input_tokenizer(comp, return_tensors='pt', padding=True, truncation=True) for comp in sample['his_comps']]

        his_jobs_id = [id for id in sample['his_jobs_id_list']]
        his_comps_id = [id for id in sample['his_comps_id_list']]

        # get sub graph
        sub_graph, _ = load_graphs(self.graphs_dir, [i]) # At this point, self.graphs_dir corresponds one-to-one with the rows of self.df
        sub_graph = sub_graph[0]

        # get instruction prompt
        sample_his_careers = sample['prep_career_desc']
        sample_careers_num = sample['careers_num']
        pos_last_career = sample_his_careers.find('Career '+str(sample_careers_num))
        input_work_exp = sample_his_careers[:pos_last_career] # work_exp_desc in instruction prompt
        job_pattern = r"(job title: [^;]+)(;)"
        comp_pattern = r"(company name: [^;]+)(;)"
        input_work_exp = re.sub(job_pattern, r"\1<|HisJobEmb|>\2", input_work_exp)
        input_work_exp = re.sub(comp_pattern, r"\1<|HisCompEmb|>\2", input_work_exp)
        candi_pairs_str_instr_prompt = [f"({pair[0]}, {pair[1]})" for pair in candi_pairs]
        candi_pairs_str_instr_prompt = (", ").join(candi_pairs_str_instr_prompt)
        instruction_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a career path planner tasked with helping an individual find new job opportunities. <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYou have access to this person's historical careers description and education background. Based on these information, your goal is to identify the most suitable next job from a set of candidate positions. The candidate jobs set is presented as a list of tuples, where the first element in each tuple is the job title and the second element is the corresponding company name. The person's historical careers description is: {input_work_exp} The person's education background is: {sample['edu_desc']} The candidate positions are: {candi_pairs_str_instr_prompt}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        instruction = self.llm_tokenizer(instruction_prompt, add_special_tokens=False, return_tensors='pt')

        # get response prompt
        sample_label = sample['next_pair']
        sample_label = sample_label.replace("'", "")
        response_prompt = f"{sample_label}<|eot_id|>"
        response = self.llm_tokenizer(response_prompt, add_special_tokens=False, return_tensors='pt')

        labels = torch.cat((torch.tensor([-100] * len(instruction["input_ids"][0])), response["input_ids"][0], torch.tensor([self.llm_tokenizer.pad_token_id])), dim=0).unsqueeze(0) # [1, seq_len]

        if not self.is_test:
            return {
                "sub_graph": sub_graph, # input for graph encoder 
                "his_jobs_ids": his_jobs_id, # his jobs id list
                "his_comps_ids": his_comps_id, # his comps id list
                "instructions": instruction, # tokenized chat template instruction prompt with special token added
                "responses": response, # tokenized chat template response prompt with special token added
                "tokenized_his_jobs": tokenized_his_jobs,
                "tokenized_his_comps": tokenized_his_comps,
                "labels": labels
            }

        else:
            return {
                "sub_graph": sub_graph, # input for graph encoder 
                "his_jobs_ids": his_jobs_id, # his jobs id list
                "his_comps_ids": his_comps_id, # his comps id list
                "instructions": instruction, # tokenized chat template instruction prompt with special token added
                "responses": response, # tokenized chat template response prompt with special token added
                "tokenized_his_jobs": tokenized_his_jobs,
                "tokenized_his_comps": tokenized_his_comps,
                "labels": labels,
                "candi_pairs_str": candi_pairs_str_instr_prompt
            }
        

class CareerDataCollator(object):
    def __call__(self, instances):
        his_jobs_ids, his_comps_ids, instructions, responses, tokenized_his_jobs, tokenized_his_comps, labels = tuple([instance[key] for instance in instances] for key in ("his_jobs_ids", "his_comps_ids", "instructions", "responses", "tokenized_his_jobs", "tokenized_his_comps", "labels"))
        batch = dict(
            his_jobs_ids=his_jobs_ids,
            his_comps_ids=his_comps_ids,
            instructions=instructions,
            responses=responses,
            tokenized_his_jobs=tokenized_his_jobs,
            tokenized_his_comps=tokenized_his_comps,
            labels=labels
        )
        batch['sub_graph'] = [instance['sub_graph'] for instance in instances]
        
        return batch


class CareerDataCollator_eval(object):
    def __call__(self, instances):
        his_jobs_ids, his_comps_ids, instructions, responses, tokenized_his_jobs, tokenized_his_comps, labels, candi_pairs_str = tuple([instance[key] for instance in instances] for key in ("his_jobs_ids", "his_comps_ids", "instructions", "responses", "tokenized_his_jobs", "tokenized_his_comps", "labels", "candi_pairs_str"))
        batch = dict(
            his_jobs_ids=his_jobs_ids,
            his_comps_ids=his_comps_ids,
            instructions=instructions,
            responses=responses,
            tokenized_his_jobs=tokenized_his_jobs,
            tokenized_his_comps=tokenized_his_comps,
            labels=labels,
            candi_pairs_str=candi_pairs_str
        )
        batch['sub_graph'] = [instance['sub_graph'] for instance in instances]
        
        return batch