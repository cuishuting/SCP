import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig, BertModel, BertTokenizer
from peft import LoraConfig, TaskType, get_peft_model
import bitsandbytes as bnb
import dgl.nn.pytorch as dglnn
from utils import get_retrieve_score
from GNNplus_for_node_emb import HAN


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class KGTextCrossModel(nn.Module):
    def __init__(self, GNN_cp_dir, gnn_in_dim, gnn_h_dim, num_heads, gnn_dropout, cross_emb_dim, fusion_encoder_hidden, soft_prompt_dim, desired_device, loss_balance_param, llm_path, num_comp_nodes_career_kg=10774, num_job_nodes_career_kg=13686):
        super().__init__()
        self.desired_device = desired_device

        # load pre-trained text encoder (frozen)
        self.text_input_encoder = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa") # frozen
        self.text_input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        for param in self.text_input_encoder.parameters():
            param.requires_grad = False

        # load pre-trained gnn encoder (frozen): only load the multi_path_HAN part
        gnn_encoder_state_dict_all = torch.load(GNN_cp_dir)
        multi_HAN_state_dict = {
            k.replace("multi_metapath_HAN.", ""): v  # Remove the prefix
            for k, v in gnn_encoder_state_dict_all.items()
            if k.startswith("multi_metapath_HAN.")
        }
        self.gnn_encoder = HAN(
            comp_meta_paths=[['has', 'belongsto'], ["has", "transtojob", "belongsto"]], 
            job_meta_paths=[['belongsto', 'has'], ['belongsto', 'transtocomp', 'has']],
            in_size=gnn_in_dim, # 384
            hidden_size=gnn_h_dim, # 512
            num_heads=num_heads, # [8]
            dropout=gnn_dropout # 0.2
        )
        self.gnn_encoder.load_state_dict(multi_HAN_state_dict)
        for param in self.gnn_encoder.parameters():
            param.requires_grad = False
        
        self.num_comps = num_comp_nodes_career_kg
        self.num_jobs = num_job_nodes_career_kg
        # learnable nodes' weights related to retrieval score
        self.node_weights = nn.Parameter(torch.randn(self.num_comps+self.num_jobs, 1)) 

        # bi-directional cross-modal interaction adapter related (trainable)
        self.job_text_wq = nn.Linear(768, cross_emb_dim) # 768 is the token embedding size of BertModel
        self.job_text_wk = nn.Linear(768, cross_emb_dim)
        self.job_text_wv = nn.Linear(768, cross_emb_dim)

        self.comp_text_wq = nn.Linear(768, cross_emb_dim)
        self.comp_text_wk = nn.Linear(768, cross_emb_dim)
        self.comp_text_wv = nn.Linear(768, cross_emb_dim)

        g_output_dim = num_heads[0] * gnn_h_dim
        self.job_graph_wq = nn.Linear(g_output_dim, cross_emb_dim)
        self.job_graph_wk = nn.Linear(g_output_dim, cross_emb_dim)
        self.job_graph_wv = nn.Linear(g_output_dim, cross_emb_dim)

        self.comp_graph_wq = nn.Linear(g_output_dim, cross_emb_dim)
        self.comp_graph_wk = nn.Linear(g_output_dim, cross_emb_dim)
        self.comp_graph_wv = nn.Linear(g_output_dim, cross_emb_dim)

        self.cross_emb = cross_emb_dim

        # attn-fusion encoder related
        self.job_cross_modal_attn = nn.Linear(cross_emb_dim, 1)
        self.comp_cross_modal_attn = nn.Linear(cross_emb_dim, 1)
        self.fusion_encoder = nn.Sequential(
            nn.Linear(cross_emb_dim, fusion_encoder_hidden), 
            nn.Linear(fusion_encoder_hidden, soft_prompt_dim)
        ) # soft_prompt_dim = 4096, static for Llama-3.1-8B-Instruct

        # LLM encoder related
        # 1. Load QLoRA configuration
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, additional_special_tokens=['<|HisJobEmb|>', '<|HisCompEmb|>'])
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        load_in_4bit = True
        bnb_4bit_compute_dtype = torch.bfloat16
        bnb_4bit_quant_type = "nf4"
        bnb_4bit_use_double_quant = True # False
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit, 
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map=torch.cuda.set_device(desired_device), quantization_config=bnb_config, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))

        self.llm.config.use_cache = False
        self.llm.enable_input_require_grads()

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj"],
            inference_mode=False, # training mode
            r=16, # Lora rank
            lora_alpha=32, 
            lora_dropout=0.1
        )
        self.llm = get_peft_model(self.llm, config)
        self.loss_balance_param = loss_balance_param
    
    def forward(self, tokenized_his_jobs, tokenized_his_comps, his_jobs_ids, his_comps_ids, sub_graph, instructions, responses, labels, is_test=False):
        tokenized_his_jobs = tokenized_his_jobs[0]
        tokenized_his_comps = tokenized_his_comps[0]
        his_jobs_ids = his_jobs_ids[0]
        his_comps_ids = his_comps_ids[0]
        sub_graph = sub_graph[0] 
        instruction = instructions[0]
        response = responses[0]
        labels = labels[0]

        # transfer input data into cuda device
        for his_job_tokens in tokenized_his_jobs:
            for key in his_job_tokens:
                his_job_tokens[key] = his_job_tokens[key].to(f'cuda:{self.desired_device}')
        for his_comp_tokens in tokenized_his_comps:
            for key in his_comp_tokens:
                his_comp_tokens[key] = his_comp_tokens[key].to(f'cuda:{self.desired_device}')
        for key in instruction:
            instruction[key] = instruction[key].to(f'cuda:{self.desired_device}')
        for key in response:
            response[key] = response[key].to(f'cuda:{self.desired_device}')

        his_jobs_emb_list = [self.text_input_encoder(**tokenized_his_job).last_hidden_state for tokenized_his_job in tokenized_his_jobs]
        his_comps_emb_list = [self.text_input_encoder(**tokenized_his_comp).last_hidden_state for tokenized_his_comp in tokenized_his_comps]

        his_jobs_mean_emb_list = [] # here "mean" is avg_emb on num_tokens for each his_job
        his_comps_mean_emb_list = []
        for job in his_jobs_emb_list:
            his_jobs_mean_emb_list.append(torch.mean(job[0], dim=0))
        for comp in his_comps_emb_list:
            his_comps_mean_emb_list.append(torch.mean(comp[0], dim=0)) 

        his_jobs_emb = torch.stack(his_jobs_mean_emb_list, dim=0).to(torch.float32) # [num_his_jobs, 768]
        his_comps_emb = torch.stack(his_comps_mean_emb_list, dim=0).to(torch.float32) # [num_his_comps, 768]
        q_text_job = self.job_text_wq(his_jobs_emb)  # [num_his_jobs, cross_emb_dim]
        q_text_comp = self.comp_text_wq(his_comps_emb) # [num_his_comps, cross_emb_dim]

        comp_nodes_feature = sub_graph.nodes['company'].data['name_emb'] # [n_nodes, emb_size]
        job_nodes_feature = sub_graph.nodes['job'].data['title_emb'] # [n_nodes, emb_size]

        gnn_input = {
            'company': comp_nodes_feature,
            'job': job_nodes_feature
        }

        sub_graph = sub_graph.to(f'cuda:{self.desired_device}')
        for key in gnn_input:
            gnn_input[key] = gnn_input[key].to(f'cuda:{self.desired_device}')

        gnn_output = self.gnn_encoder(sub_graph, gnn_input)

        jobs_nid = sub_graph.ndata[dgl.NID]['job']
        comps_nid = sub_graph.ndata[dgl.NID]['company']
        comp_nodes_weights = self.node_weights[:self.num_comps][comps_nid]
        job_nodes_weights = self.node_weights[self.num_comps:][jobs_nid]
        all_nodes_weights = torch.softmax(torch.cat([comp_nodes_weights, job_nodes_weights],dim=0), dim=0) # [n_sg_nodes, 1]
        comp_nodes_weights = all_nodes_weights[:len(comps_nid)]
        job_nodes_weights = all_nodes_weights[len(comps_nid):]
        
        all_comp_nodes_emb = comp_nodes_weights * gnn_output['company'] # [n_comp_nodes, gnn_output_dim]
        all_job_nodes_emb = job_nodes_weights * gnn_output['job'] # [n_job_nodes, gnn_output_dim]

        k_graph_job = self.job_graph_wk(all_job_nodes_emb)
        v_graph_job = self.job_graph_wv(all_job_nodes_emb) # all with shape: [n_job_nodes, cross_emb_dim]

        k_graph_comp = self.comp_graph_wk(all_comp_nodes_emb)
        v_graph_comp = self.comp_graph_wv(all_comp_nodes_emb) # all with shape: [n_comp_nodes, cross_emb_dim]

        # text-query cross attn embedding 
        his_jobs_text_q_cross_emb = torch.matmul(F.softmax(torch.matmul(q_text_job, torch.transpose(k_graph_job, 0, 1)) / (self.cross_emb ** 0.5), dim=-1), v_graph_job) # [num_his_jobs, cross_emb_dim]
        his_comps_text_q_cross_emb = torch.matmul(F.softmax(torch.matmul(q_text_comp, torch.transpose(k_graph_comp, 0, 1)) / (self.cross_emb ** 0.5), dim=-1), v_graph_comp) # [num_his_comps, cross_emb_dim]

        # graph-query cross-attn emb
        q_graph_job = self.job_graph_wq(all_job_nodes_emb) # [n_job_nodes, cross_emb_dim]
        q_graph_comp = self.comp_graph_wq(all_comp_nodes_emb) # [n_comp_nodes, cross_emb_dim]

        k_text_job = self.job_text_wk(his_jobs_emb) # [n_his_jobs, cross_emb_dim]
        v_text_job = self.job_text_wv(his_jobs_emb) # [n_his_jobs, cross_emb_dim]
        k_text_comp = self.comp_text_wk(his_comps_emb) # [n_his_comps, cross_emb_dim]
        v_text_comp = self.comp_text_wv(his_comps_emb) # [n_his_jobs, cross_emb_dim]

        job_graph_q_cross_emb = torch.matmul(F.softmax(torch.matmul(q_graph_job, torch.transpose(k_text_job, 0, 1)) / (self.cross_emb ** 0.5), dim=-1), v_text_job) # [n_job_nodes, cross_emb_dim]
        comp_graph_q_cross_emb = torch.matmul(F.softmax(torch.matmul(q_graph_comp, torch.transpose(k_text_comp, 0, 1)) / (self.cross_emb ** 0.5), dim=-1), v_text_comp) # [n_comp_nodes, cross_emb_dim]
        # extract his career entities' graph_query_emb
        org_job_nid_list = sub_graph.ndata[dgl.NID]['job']
        org_comp_nid_list = sub_graph.ndata[dgl.NID]['company']
        his_jobs_positions = [torch.where(org_job_nid_list == i)[0].item() for i in his_jobs_ids] # current batch_size == 1, check CareerDataCollator()
        his_comps_positions = [torch.where(org_comp_nid_list == i)[0].item() for i in his_comps_ids]
        
        his_jobs_graph_q_cross_emb = torch.cat([job_graph_q_cross_emb[p] for p in his_jobs_positions], dim=0).reshape((-1, self.cross_emb)) # [num_his_jobs, cross_emb_dim]
        his_comps_graph_q_cross_emb = torch.cat([comp_graph_q_cross_emb[p] for p in his_comps_positions], dim=0).reshape((-1, self.cross_emb)) # [num_his_comps, cross_emb_dim]

        # attn-based fusion encoder
        # job:
        job_text_modal_w = self.job_cross_modal_attn(his_jobs_text_q_cross_emb) # [n_his_jobs, 1]
        job_graph_modal_w = self.job_cross_modal_attn(his_jobs_graph_q_cross_emb) # [n_his_jobs, 1]
        
        job_att_fusion_w = torch.stack([job_text_modal_w, job_graph_modal_w], dim=-1) # [n_his_jobs, 1, 2]
        job_att_fusion_w = F.softmax(job_att_fusion_w, dim=-1) 
        job_text_att, job_graph_att = job_att_fusion_w.split([1, 1], dim=-1) # both are [n_h_j, 1, 1]
        fused_his_jobs_emb = job_text_att.squeeze(-1) * his_jobs_text_q_cross_emb + job_graph_att.squeeze(-1) * his_jobs_graph_q_cross_emb # [num_his_jobs, cross_emb_dim]

        # comp:
        comp_text_modal_w = self.comp_cross_modal_attn(his_comps_text_q_cross_emb)
        comp_graph_modal_w = self.comp_cross_modal_attn(his_comps_graph_q_cross_emb)
        comp_att_fusion_w = torch.stack([comp_text_modal_w, comp_graph_modal_w], dim=-1)
        comp_att_fusion_w = F.softmax(comp_att_fusion_w, dim=-1)
        comp_text_att, comp_graph_att = comp_att_fusion_w.split([1, 1], dim=-1)
        fused_his_comps_emb = comp_text_att.squeeze(-1) * his_comps_text_q_cross_emb + comp_graph_att.squeeze(-1) * his_comps_graph_q_cross_emb # [num_his_comps, cross_emb_dim]

        # get his job/comp soft prompt
        soft_prompt_his_jobs = self.fusion_encoder(fused_his_jobs_emb) # [num_his_jobs, 4096]
        soft_prompt_his_comps = self.fusion_encoder(fused_his_comps_emb) # [num_his_comps, 4096]

        special_token_id_his_job = self.llm_tokenizer(f"<|HisJobEmb|>", add_special_tokens=False, return_tensors='pt')['input_ids'][0].item()
        special_token_id_his_comp = self.llm_tokenizer(f"<|HisCompEmb|>", add_special_tokens=False, return_tensors='pt')['input_ids'][0].item()

        if self.training:
            input_ids = torch.cat((instruction["input_ids"][0], response["input_ids"][0], torch.tensor([self.llm_tokenizer.pad_token_id]).to(f"cuda:{self.desired_device}")), dim=0).unsqueeze(0) # [1, input_ids_len]
            attention_mask = torch.cat((instruction["attention_mask"][0], response["attention_mask"][0], torch.tensor([1]).to(f"cuda:{self.desired_device}")), dim=0).unsqueeze(0).to('cuda:0')
            labels = labels.to(f"cuda:{self.desired_device}")
        else: # val/test mode
            input_ids = torch.cat((instruction["input_ids"][0], torch.tensor([self.llm_tokenizer.pad_token_id]).to(f"cuda:{self.desired_device}")), dim=0).unsqueeze(0) # [1, input_ids_len]
            attention_mask = torch.cat((instruction["attention_mask"][0], torch.tensor([1]).to(f"cuda:{self.desired_device}")), dim=0).unsqueeze(0).to('cuda:0')

        pos_his_jobs = torch.nonzero(instruction['input_ids'][0] == special_token_id_his_job, as_tuple=True)[0]
        pos_his_comps = torch.nonzero(instruction['input_ids'][0] == special_token_id_his_comp, as_tuple=True)[0]

        prompt_emb = self.llm.get_input_embeddings()(input_ids) # [1, token_seq_len, 4096]

        assert len(pos_his_jobs) == len(soft_prompt_his_jobs)
        assert len(pos_his_comps) == len(soft_prompt_his_comps)

        with torch.no_grad():
            for i, pos in enumerate(pos_his_jobs):
                prompt_emb[0, pos] = soft_prompt_his_jobs[i]
            for i, pos in enumerate(pos_his_comps):
                prompt_emb[0, pos] = soft_prompt_his_comps[i]

        if self.training:
            outputs = self.llm(inputs_embeds=prompt_emb, attention_mask=attention_mask, labels=labels)
            first_part_loss = outputs['loss']
            retrieval_score = -1 * get_retrieve_score(his_jobs_positions, his_comps_positions, gnn_output['job'], gnn_output['company'], job_nodes_weights, comp_nodes_weights)

            result = {
                "loss": first_part_loss + self.loss_balance_param * retrieval_score, # dict with keys: 'loss', 'logits'
                "logits": outputs['logits']
            }

            return result
        
        elif not self.training and is_test: #  test mode , return 3 metrics
            outputs = self.llm.generate(**instruction, pad_token_id=self.llm_tokenizer.pad_token_id, max_new_tokens=20, num_beams=5, num_return_sequences=5, return_dict_in_generate=True, output_scores=True)
            candidate_sequences = outputs.sequences
            candidate_sequences = [seq[instruction["input_ids"].shape[1]:] for seq in candidate_sequences]

            candidate_texts = [
                self.llm_tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_space=False)
                for seq in candidate_sequences
            ]
            
            gold_answer = self.llm_tokenizer.decode(
                response["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_space=False
            )


            return {
                'predictions': candidate_texts, # list
                'gold_answer': gold_answer, # str
            }



        else: # eval mode (called by Trainer)
            target_length = 20 # total pad length: calculate hit@1

            gold_ids = response["input_ids"][0]

            """calculate hit@1 & evaluation during training (based on hit@1 metric to early stop)"""
            outputs_ids = self.llm.generate(**instruction, pad_token_id=self.llm_tokenizer.pad_token_id, max_new_tokens=len(gold_ids))[0][instruction["input_ids"].shape[1]:] 
            
            outputs_ids = torch.cat([outputs_ids, torch.tensor([self.llm_tokenizer.pad_token_id] * (target_length - len(outputs_ids)), dtype=torch.long).to(f"cuda:{self.desired_device}")], dim=0)

            gold_ids = torch.cat([gold_ids, torch.tensor([self.llm_tokenizer.pad_token_id]*(target_length-len(gold_ids)), dtype=torch.long).to(f"cuda:{self.desired_device}")], dim=0)

            response_text = self.llm_tokenizer.decode(outputs_ids, skip_special_tokens=True, clean_up_tokenization_space=False)
            gold_answer = self.llm_tokenizer.decode(gold_ids, skip_special_tokens=True, clean_up_tokenization_space=False)
            # print(f"check response text: {response_text}")
            # print(f"check gold answer: {gold_answer}")

            if gold_answer in response_text:
                correct = torch.tensor(1.0)
            else:
                correct = torch.tensor(0.0)
            # print(f"outputs_ids:{outputs_ids.shape}")
            # print(f"label_ids:{gold_ids.shape}")
            return {
                'loss': correct,
                'predictions': outputs_ids,
                'label_ids': gold_ids
            }

            """For case study"""
            # if response_text == gold_answer:
            #     # self.node_weights = nn.Parameter(torch.randn(self.num_comps+self.num_jobs, 1)) 
            #     case_dict = {
            #         'his_jobs_ids': his_jobs_ids,
            #         'his_comps_ids': his_comps_ids,
            #         'gold_answer': gold_answer,
            #     }
            #     all_nodes_weights = torch.cat([comp_nodes_weights, job_nodes_weights], dim=0) # [n_sg_nodes, 1]
            #     sg_comp_nodes_num = comp_nodes_weights.shape[0]
            #     if sg_comp_nodes_num > 20:
                    
            #         top20_nodes_idx = torch.topk(all_nodes_weights.squeeze(), k=20).indices.tolist()
            #         top20_comp_ids = [id for id in top20_nodes_idx if id < sg_comp_nodes_num]
            #         top20_job_ids = [id - sg_comp_nodes_num for id in top20_nodes_idx if id >= sg_comp_nodes_num]

            #         # print(f"all: {top20_nodes_idx}") # [20, ]
            #         # print(f"comp: {top20_comp_ids}")
            #         # print(f"job: {top20_job_ids}")

            #         org_job_nid_list = sub_graph.ndata[dgl.NID]['job'].tolist()
            #         org_comp_nid_list = sub_graph.ndata[dgl.NID]['company'].tolist()
            #         # print(f"org comp nid list:{org_comp_nid_list}")
            #         # print(f"org job nid list:{org_job_nid_list}")

            #         impt_jobs = [org_job_nid_list[i] for i in top20_job_ids]
            #         impt_comps = [org_comp_nid_list[i] for i in top20_comp_ids]
            #         case_dict['impt_jobs'] = impt_jobs
            #         case_dict['impt_comps'] = impt_comps
            #         # print(f"impt jobs: {impt_jobs}")
            #         # print(f"impt comps: {impt_comps}") # career_kg node_id

            #         impt_nodes = {
            #             'job': top20_job_ids,
            #             'company': top20_comp_ids
            #         }

            #         impt_subgraph = dgl.node_subgraph(sub_graph, impt_nodes, relabel_nodes=True, store_ids=True)

            #         for etype in impt_subgraph.canonical_etypes:
            #             src, dst = impt_subgraph.edges(etype=etype)

            #             # Step 2: 从 impt_subgraph 的节点 ID 转换为 sub_graph 的节点 ID
            #             src_subgraph_ids = impt_subgraph.nodes[etype[0]].data[dgl.NID][src]
            #             dst_subgraph_ids = impt_subgraph.nodes[etype[2]].data[dgl.NID][dst]

            #             # Step 3: 从 sub_graph 的节点 ID 转换为 career_kg 的节点 ID
            #             src_career_kg_ids = sub_graph.nodes[etype[0]].data[dgl.NID][src_subgraph_ids]
            #             dst_career_kg_ids = sub_graph.nodes[etype[2]].data[dgl.NID][dst_subgraph_ids] # career_kg node_id

            #             case_dict[etype[1]+'_src'] = src_career_kg_ids.tolist() 
            #             case_dict[etype[1]+ '_dst'] = dst_career_kg_ids.tolist()
            #             # 打印结果
            #             # print(f"Edge type: {etype}")
            #             # print(f"Source IDs in career_kg: {src_career_kg_ids.tolist()}")
            #             # print(f"Destination IDs in career_kg: {dst_career_kg_ids.tolist()}")

            #         # print(case_dict)
            #         # print("$$$$$$$$$$$$$")
            #         if self.num_comps == 7638: # IT dataset
            #             with open("/home/cst/LLM_CP_pred/KG_enhanced_Career_Rec_2025/case_study/cases_it.txt", 'a') as f:
            #                 f.write(str(case_dict) + "\n")
            #         else: # FIN dataset
            #             with open("/home/cst/LLM_CP_pred/KG_enhanced_Career_Rec_2025/case_study/cases_fin.txt", 'a') as f:
            #                 f.write(str(case_dict) + "\n")
            """end of case study"""

            """calculate hit@3 / hit@5"""
            # outputs = self.llm.generate(**instruction, pad_token_id=self.llm_tokenizer.pad_token_id, max_new_tokens=20, num_beams=5, num_return_sequences=5, return_dict_in_generate=True, output_scores=True)
            # candidate_sequences = outputs.sequences
            # candidate_sequences = [seq[instruction["input_ids"].shape[1]:] for seq in candidate_sequences]

            # candidate_texts = [
            #     self.llm_tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_space=False)
            #     for seq in candidate_sequences
            # ]
            
            # gold_answer = self.llm_tokenizer.decode(
            #     response["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_space=False
            # )


            # return {
            #     'predictions': candidate_texts, # list
            #     'gold_answer': gold_answer, # str
            # }