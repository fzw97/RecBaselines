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

We followed the Leave-One-Out evaluation strategy in SASRec. Specifically, for each user, we randomly sample 100 negative items and rank these items with the ground-truth item. HR and NDCG are estimated based on the ranking results. 

| Test Result |  Recall@10 |  NDCG@10   |
| ----------- | ---------- | ---------- |
| BPRMF       | 76.51±0.26 | 55.77±0.16 |
| NeuMF       | 79.35±0.12 | 59.06±0.24 | 
| GRec        | 81.55±0.17 | 55.74±0.18 | 
| DGRec       | 86.57±0.18 | 63.55±0.26 | 
| SocialMF    | 76.27±0.28 | 53.42±0.21 |  
| SoRec       | 81.45±0.04 | 58.15±0.07 | 
| LightGCN    | 84.39±0.07 | 60.80±0.19 | 
| SASRec      | 81.66±0.08 | 57.21±0.37 | 
| ASASRec     | 84.53±0.04 | 60.53±0.09 | 
| TransRec    | 80.19±0.20 | 64.00±0.15 | 


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


