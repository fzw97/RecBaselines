# RecBaselines
PyTorch implementations of some Recommendation System papers

## Requirements
* scipy==1.1.0
* networkx==2.1
* pandas==0.23.4
* numpy==1.15.1
* joblib==1.0.1
* torch==1.8.0

## Run the demo
```bash
cd runs
python run_bprmf.py
```
## Data
Yelp: https://www.kaggle.com/yelp-dataset/yelp-dataset

## Results
We followed the evaluation strategy in SASRec. Specifically, for each user, we randomly sample 100 negative items and rank these items with the ground-truth item. HR and NDCG are estimated based on the ranking results. 

| Test Result |  Recall@10 |  NDCG@10   |
| ----------- | ---------- | ---------- |
| BPRMF       | 76.51±0.26 | 55.77±0.16 |
| NeuMF       |  |  | 
| GRec        |  |  | 
| DGRec       |  |  | 
| SocialMF    |  |  |  
| SoRec       |  |  | 
| LightGCN    |  |  | 
| SASRec      |  |  | 
| TransRec    |  |  | 


## Model & Paper
This repo contains the following models:
* BPRMF: https://arxiv.org/abs/1205.2618
* NeuMF: https://arxiv.org/abs/1708.05031
* GRec: https://dl.acm.org/doi/abs/10.1145/3308558.3313488
* DGRec: https://dl.acm.org/doi/abs/10.1145/3289600.3290989
* SocialMF: https://dl.acm.org/doi/abs/10.1145/1864708.1864736
* SoRec: https://dl.acm.org/doi/abs/10.1145/1458082.1458205
* LightGCN: https://dl.acm.org/doi/abs/10.1145/3397271.3401063
* SASRec: https://arxiv.org/abs/1808.09781
* TransRec: https://dl.acm.org/doi/abs/10.1145/3109859.3109882


