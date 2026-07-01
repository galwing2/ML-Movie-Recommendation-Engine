🎬 Movie Recommendation Engine (C | Matrix Factorization)








A high-performance collaborative filtering recommendation system written in pure C, implementing stochastic gradient descent (SGD) to learn latent user and movie representations.

The system predicts unseen ratings and generates personalized top-N movie recommendations.

📌 Key Features
⚙️ Matrix Factorization recommender (latent embeddings)
📉 SGD-based optimization with L2 regularization
🧠 User & movie bias learning
📊 Train/test split with RMSE evaluation
🎯 Top-10 personalized recommendations
🧹 Movie title normalization (articles + year cleanup)
💾 Fully manual memory management (no ML libraries)
🧠 Model Overview

The model predicts ratings using:

r^ui=μ+bu+bi+Pu⋅Qi
	​
Where:
μ = global mean rating (3.53)
b_u, b_i = user/movie biases
P_u, Q_i = latent feature vectors (dimension K = 20)
⚙️ Hyperparameters
Parameter	Value	Description
K	20	Latent feature dimension
EPOCHS	50	Training iterations
ALPHA	0.005	Learning rate
LAMBDA	0.01	Regularization strength
MEAN	3.53	Global rating mean
📂 Dataset Format
Ratings File
user_id movie_id rating timestamp

Example:

1 10 4.0 881250949
Movies File
movie_id|movie_title

Example:

1|Toy Story (1995)
🚀 Build Instructions

Compile using GCC:

gcc main.c -o recommender -lm

-lm is required for math operations (sqrt)

▶️ Usage
./recommender <ratings_file> <movies_file> <user_id>
Example:
./recommender ratings.txt movies.txt 42
📊 Output Example
[INFO] Loaded 100000 ratings. Found 943 users and 1682 movies.
Finished Epoch 50 with RMSE 0.89

[RESULT] True Testing RMSE on 20000 unseen ratings: 0.91

===========================================
=== TOP 10 RECOMMENDATIONS FOR USER 42 ===
1. The Matrix (1999)         | Predicted Rating: 4.82
2. Inception (2010)          | Predicted Rating: 4.79
3. Fight Club (1999)         | Predicted Rating: 4.76
...
===========================================
📸 Screenshots / Demo
Training Performance

Recommendation Output

Replace images by adding them to:

/assets/training.png
/assets/recommendations.png
🏗️ System Architecture
Ratings Data ──► Train/Test Split ──► SGD Training
                                      │
                                      ▼
                           Latent Matrices P, Q
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
             RMSE Evaluation                 Top-N Recommendations
🧪 Evaluation
Random 80/20 train-test split
RMSE computed on unseen ratings
Full-dataset SGD optimization

🧩 Core Modules
📥 Data Loader
Dynamic array expansion
Index normalization (0-based IDs)
🧠 Training Engine
SGD updates for:
user vectors P
item vectors Q
biases b_u, b_i
📊 Evaluation
Computes RMSE over test set
🎯 Recommendation Engine
Scores unseen movies
Sorts predictions
Outputs top-10 results

⚠️ Limitations
No cold-start handling
No implicit feedback
CPU-only training
Fixed hyperparameters (no tuning loop)

🚀 Future Improvements
Adam / RMSProp optimization
Mini-batch SGD
Parallel training (OpenMP)
Model persistence (save/load weights)
Web API wrapper (FastAPI)
Hybrid recommender (content + collaborative)
🛠️ Dependencies

Standard C libraries only:
stdio.h
stdlib.h
string.h
math.h
time.h
stdbool.h
👤 Author

Built as a systems-level ML project demonstrating:

Matrix factorization from scratch
Optimization via SGD
Low-level memory management in C
Recommender system design
📜 License

This project is licensed under the MIT License.
