# Preliminaries:
* Python version: Python 3.11.9
* Required packages: Specified in requirements.txt

# Step1:  MMP-GAT pre-training for graph-modal embeddings

* Run the following command to pre-train MMP-GAT and obtain graph-modal node embeddings. The best checkpoint will be automatically saved under the directory: ./Multi_HAN_results/, with the following filename format: early_stop_cp\_{seed}\_{date}\_{hour}\_{minute}\_{second}_{dataset}.pth

```bash
python GNNplus_for_node_emb.py \
  --KG_dir /abs/path/to/your_kg.bin \ # path to the KG 
  --data_type fin \ # dataset type, in our case, fin/it
  --hidden_dim 512 \ # hidden dimension of MMP-GAT
  --dropout 0.2 \ 
  --num_heads 8 # num_heads of MMP-GAT
```



# Step2: Train & Evaluate SCP 

* Run the following command to train and evaluate SCP.

  ``` bash
  python main.py \
    --seed ${SEED} \
    --train_data_dir path/to/dataset/data_train.csv \
    --val_data_dir path/to/dataset/data_val.csv \
    --test_data_dir path/to/FIN_dataset/data_test.csv \
    --train_sg_dir path/to/career_KG_FIN/train_2hop_sub_g.bin \
    --val_sg_dir path/to/career_KG_FIN/val_2hop_sub_g.bin \
    --test_sg_dir path/to/career_KG_FIN/test_2hop_sub_g.bin \
    --gnn_cp_dir path/to/MMP-GAT_checkpoint.pth \ # pre-trained MMP-GAT checkpoint from step 1 
    --model_cp_dir path/to/save_checkpoints \ # the directory to save SCP checkpoint
    --gnn_h_dim 512 \ # hidden dimension of pre-trained MMP-GAT
    --gnn_num_heads 8 \ # num_heads of pre-trained MMP-GAT
    --gnn_dropout 0.2 \
    --cross_emb_dim 1024 \ # cross-modality embedding dimension
    --fusion_encoder_hidden 2048 \ 
    --loss_balance_param 1.0 \
    --lr 5e-4 \
    --num_comps 5365 \ # number of companies in current career KG
    --num_jobs 8464 \ # number of jobs in current career KG
    --llm_path path/to/llm_backbone # path to the backbone LLM
  ```

  

# Dataset Description

Due to privacy and legal constraints, we are unable to publicly release the original LinkedIn user data used in our experiments. The dataset contains sensitive personal career trajectories and educational histories collected from a professional networking platform. To facilitate reproducibility and understanding of the data structure, we provide anonymized schema descriptions and illustrative examples below, which reflect the exact format and semantics of the data used in our experiments.

Our data consists of two main parts: the tabular data and the career KG.

* Tabular data

  | Column Name       | Description                                               | example                                                      |
  | ----------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
  | user_code         | Anonymized user identifier.                               | jerry-b53686121                                              |
  | careers_num       | Total number of recorded career stages.                   | 2                                                            |
  | prep_career_desc  | Preprocessed textual career description used in prompts.  | Career 1: job title: senior software specialist engineer analyst; company name: nationwide; start time: 1985/2; end time: 2002/7; job address: nan; job content: Develop, maintain and support commercial insurance applications using Cobol (mainframe and PC applications), DB2 database admin, MQSeries administrator and C developer on Windows and RS6000 servers, developed data warehouse applications with Microstrategy tools, Teradata platform, developed a web based agent locator on Sun Solariemweb server using Java script and Perl CGI\n. Career 2: job title: senior software engineer analyst; company name: palmetto gba; start time: 2002/7; end time: 2019/2; job address: nan; job content: Maintain Medicare C applications on RS6000, Oracle admin and developer on RS6000, developed web applications on the Windows platform using Window DotNet developer tools (ASP.Net) for Palmetto operational Medicare units, currently provide mainframe systems support for Medicare testing units. |
  | his_jobs          | List of historical job titles (excluding last career).    | [senior software specialist engineer analyst]                |
  | his_comps         | List of historical company names (excluding last career). | [nationwide]                                                 |
  | his_jobs_id_list  | Job node IDs in Career KG (historical).                   | [8199]                                                       |
  | his_comps_id_list | Company node IDs in Career KG (historical).               | [43]                                                         |
  | edu_desc          | Textual description of education background.              | Education 1: school name: The Ohio State University; major: BS Wildlife Management, Natural Resouces, Wildlife Management; start time: 1975; end time: 1978. Education 2: school name: Columbus Technical Institute; major: AS Data Processing, Computer Programming; start time: 1981; end time: 1985. |
  | next_pair         | Ground-truth next career hop (job, company).              | (senior software engineer analyst, palmetto gba)             |
  | candi_pairs_20    | Candidate set of 20 (job, company) pairs.                 | [(financial account manager, standard bank namibia), (field network engineer, acuative), (system analyst, zensar technologies), (senior development engineer, walmart labs), (market director, national futures association), (senior system administrator, micro focus), (senior technology specialist, synechron), (manager, small information technology and services company), (associate, booz allen hamilton), (senior software engineer analyst, palmetto gba), (business analysis manager, first data corporation), (technology recruiter leader, small information technology and services company), (mobile application development leader, small information technology and services company), (assistant manager, small investment banking company), (cloud engineer, small information technology and services company), (senior testing engineer, orion health), (account commercial manager, small insurance company), (assistant manager, noor bank), (technologist, small real estate company), (software testing integration engineer, evosoft)] |

* Career KG

  The statistics of the constructed career KG can be found in the table below. Each type of node has its own textual attributes. We store the name and the type for the company nodes and job title
  for the job nodes. We utilize Sentence Transformer to get both the company names’ and job titles’ initial embeddings with a fixed size of 384.

  

|  Entities and Relationships   | Finance Knowledge Graph | Finance Knowledge Graph |
| :---------------------------: | :---------------------: | :---------------------: |
|            Company            |          5,365          |          7,638          |
|              Job              |          8,464          |          7,885          |
|    Job-belongs_to-Company     |         23,835          |         26,759          |
|        Company-has-Job        |         23,835          |         26,759          |
| Company-trans_to_comp-Company |          3,405          |          4,526          |
|     Job-trans_to_job-Job      |          5,424          |          6,425          |

