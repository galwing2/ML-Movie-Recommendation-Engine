#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h> 



#define INITIAL_CAPACITY 1000
#define EPOCHS 50
#define K 20
#define ALPHA 0.005f
#define LAMBDA 0.01f
#define MEAN 3.53f

void normalize_title(char* title);

typedef struct Rating{
    int MovieID;
    int UserID;
    float MovieRating;
} Rating;

typedef struct Prediction{
    int MovieID;
    float PredictedRating;
} Prediction;

typedef struct Movie{
    int MovieID;
    char MovieTitle[256];
} Movie;


Rating* load_data(const char* filename, int* total_ratings,int* num_users, int* num_movies){
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }
    int max_user_id = 0;
    int max_movie_id = 0;

    int capacity = INITIAL_CAPACITY;
    *total_ratings = 0;
    Rating* dataset = malloc(capacity*sizeof(Rating));
    if (!dataset) {
        fclose(file);
        return NULL;
    }

    char line[256];
    int user_id, movie_id;
    float r;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%d %d %f %*d", &user_id, &movie_id, &r) == 3) {

            if (user_id > max_user_id) max_user_id = user_id;
            if (movie_id > max_movie_id) max_movie_id = movie_id;
            if(*total_ratings>=capacity){
                capacity *= 2;
                Rating* temp = realloc(dataset, capacity * sizeof(Rating));
                if (!temp) {
                    printf("Memory reallocation failed!\n");
                    free(dataset);
                    fclose(file);
                    return NULL;
                }
                dataset = temp;
            }
            dataset[*total_ratings].MovieID = movie_id-1;
            dataset[*total_ratings].UserID = user_id-1;
            dataset[*total_ratings].MovieRating = r;

            (*total_ratings) ++;
        }
    }
    *num_users = max_user_id;
    *num_movies = max_movie_id;
    fclose(file);
    Rating* final_dataset = realloc(dataset, (*total_ratings) * sizeof(Rating));
    if (final_dataset) {
        dataset = final_dataset;
    }    
    return dataset;
}

Movie* load_movie_titles(const char* filename, int num_movies) {
    Movie* movies = calloc(num_movies, sizeof(Movie));
    FILE* file = fopen(filename, "r");
    if (!file) return movies;

    char line[512];
    while (fgets(line, sizeof(line), file)) {
        int id;
        char title[256];
        if (sscanf(line, "%d|%255[^|]", &id, title) == 2) {
            if (id <= num_movies) {
                movies[id - 1].MovieID = id - 1;
                strncpy(movies[id - 1].MovieTitle, title, 255);
                normalize_title(movies[id - 1].MovieTitle);
            }
        }
    }
    fclose(file);
    return movies;
}

void normalize_title(char* title) {
    // Find and temporarily remove the year "(YYYY)" from the end
    char year[16] = "";
    char* year_start = strrchr(title, '(');
    if (year_start != NULL) {
        // Copy the year part, trim the space before it, then cut the string
        strncpy(year, year_start, sizeof(year) - 1);
        char* p = year_start;
        while (p > title && *(p-1) == ' ') p--;
        *p = '\0';
    }

    // Now normalize the article on the year-stripped title
    const char* suffixes[] = {", The", ", A", ", An"};
    for (int i = 0; i < 3; i++) {
        char* pos = strstr(title, suffixes[i]);
        if (pos == NULL) continue;

        // Must be at the end of the (now year-stripped) title
        char* after = pos + strlen(suffixes[i]);
        while (*after == ' ' || *after == '\n' || *after == '\r') after++;
        if (*after != '\0') continue;

        // Extract article and rebuild
        char article[8];
        strncpy(article, suffixes[i] + 2, sizeof(article) - 1);
        article[sizeof(article) - 1] = '\0';

        char temp[256];
        *pos = '\0';
        snprintf(temp, sizeof(temp), "%s %s", article, title);
        strncpy(title, temp, 255);
        title[255] = '\0';
        break;
    }

    // Re-append the year
    if (year[0] != '\0') {
        strncat(title, " ", 255 - strlen(title));
        strncat(title, year, 255 - strlen(title));
    }
}
float* allocate_feature_matrix(int num_of_objects){
    float* matrix = malloc(num_of_objects*K*sizeof(float));
    if(matrix==NULL){
        perror("Failed to allocate feature matrix.\n");
        return NULL;
    }

    for(int i=0; i<num_of_objects*K;i++){
        matrix[i] = ((float)rand() / (float)(RAND_MAX)) * 0.1f;
    }
    return matrix;
}

void train_model(Rating* dataset, int total_ratings, float* P, float* Q,int num_users,int num_movies,float* b_u,float* b_i){ //P users, Q movies
    printf("[INFO] Starting SGD Training for %d Epochs...\n", EPOCHS);
    fflush(stdout); 
    for(int epoch = 0; epoch<EPOCHS; epoch++){
        float squared_error_sum = 0.0;
        for(int i = 0; i < total_ratings; i++){
            int u = dataset[i].UserID;
            int m = dataset[i].MovieID;
            if (u < 0 || u >= num_users || m < 0 || m >= num_movies) {
                continue; // Skip anomalous data points
            }
            float y = dataset[i].MovieRating;
            float dot_product = 0.0f;
            for(int k=0;k<K;k++){
                dot_product+=P[u*K+k]*Q[m*K+k];
            }
            float pred = MEAN + b_u[u] + b_i[m] + dot_product;
            float error_ui = y-pred;
            squared_error_sum += (error_ui * error_ui);
            b_u[u] += ALPHA * (error_ui - LAMBDA * b_u[u]);
            b_i[m] += ALPHA * (error_ui - LAMBDA * b_i[m]);

            for(int k = 0; k < K; k++){
                int p_index = u*K+k;
                int q_index = m*K+k;
                float temp_p = P[p_index];
                P[p_index] += ALPHA*(error_ui*Q[q_index]-LAMBDA*P[p_index]);
                Q[q_index] += ALPHA*(error_ui*temp_p-LAMBDA*Q[q_index]);
            }
        }
        float rmse = sqrt(squared_error_sum / total_ratings);
        printf("Finished Epoch %d with RMSE %f\n", epoch + 1,rmse);
        fflush(stdout);
    }
}

void evaluate_model(Rating* test_dataset, int test_count, float* P, float* Q, float* b_u, float* b_i) {
    float squared_error_sum = 0.0f;
    
    for (int i = 0; i < test_count; i++) {
        int u = test_dataset[i].UserID;
        int m = test_dataset[i].MovieID;
        float actual = test_dataset[i].MovieRating;

        float dot_product = 0.0f;
        for (int k = 0; k < K; k++) {
            dot_product += P[u * K + k] * Q[m * K + k];
        }
        
        float pred = MEAN + b_u[u] + b_i[m] + dot_product;
        float error = actual - pred;
        squared_error_sum += (error * error);
    }
    
    float test_rmse = sqrt(squared_error_sum / test_count);
    printf("\n===========================================\n");
    printf("[RESULT] True Testing RMSE on %d unseen ratings: %.4f\n", test_count, test_rmse);
    printf("===========================================\n\n");
}


int compare_predictions(const void* a, const void* b) {
    Prediction* p1 = (Prediction*)a;
    Prediction* p2 = (Prediction*)b;
    
    // Sort in descending order (highest rating first)
    if (p1->PredictedRating < p2->PredictedRating) return 1;
    if (p1->PredictedRating > p2->PredictedRating) return -1;
    return 0;
}

void predict_top_movies(int target_userId,Rating* dataset, int total_ratings,float* P_trained, float* Q_trained,int num_movies, Movie* movie_data,float* b_u,float* b_i){
    bool* visited = calloc(num_movies, sizeof(bool));
    for(int i=0;i<total_ratings;i++){
        if(dataset[i].UserID==target_userId){
            visited[dataset[i].MovieID] = 1;
        } 
    }
    Prediction* recs = malloc(num_movies * sizeof(Prediction));
    int rec_count=0;
    for(int i=0;i<num_movies;i++){
        if(visited[i]==0){
            float dot_product = 0.0f;
            for(int k=0;k<K;k++){
                dot_product += P_trained[target_userId*K+k]*Q_trained[i*K+k];
            }
            float prediction = MEAN + b_u[target_userId] + b_i[i] + dot_product;

            recs[rec_count].MovieID = i;
            recs[rec_count].PredictedRating = prediction;
            rec_count+=1;

        }
    }
    qsort(recs, rec_count, sizeof(Prediction), compare_predictions);
    printf("\n=== TOP 10 RECOMMENDATIONS FOR USER %d ===\n", target_userId + 1);
    fflush(stdout);
    for (int i = 0; i < 10 && i < rec_count; i++) {
        printf("%2d. %-40s | Predicted Rating: %.2f\n", 
               i + 1, movie_data[recs[i].MovieID].MovieTitle, recs[i].PredictedRating);
        fflush(stdout);
    }
    printf("===========================================\n");
    fflush(stdout);

    // 6. Clean up
    free(visited);
    free(recs);
}

void print_user_history(int target_user, Rating* dataset, int total_ratings, Movie* movie_data) {
    printf("\n=== RATING HISTORY FOR USER %d ===\n", target_user + 1);
    int count = 0;
    
    // Iterate through all ratings, filtering for our target user
    for (int i = 0; i < total_ratings; i++) {
        if (dataset[i].UserID == target_user) {
            int m_id = dataset[i].MovieID;
            printf(" - Rated %.1f/5.0 | %s\n", dataset[i].MovieRating, movie_data[m_id].MovieTitle);
            count++;
        }
    }
    
    if (count == 0) {
        printf(" No rating history found for this user.\n");
    } else {
        printf(" Total Movies Rated: %d\n", count);
    }
    printf("===================================\n\n");
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <dataset_file> <target_user_id>\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(time(NULL)); // Seed random number generator

    const char* filename = argv[1];
    int target_user = atoi(argv[3]) - 1; 

    int total_ratings = 0;
    int num_users = 0;
    int num_movies = 0;

    // 1. Load Data
    Rating* dataset = load_data(filename, &total_ratings, &num_users, &num_movies);
    if (!dataset) return EXIT_FAILURE;

    Movie* movie_data = load_movie_titles(argv[2], num_movies);
    if (!movie_data) return EXIT_FAILURE;

    
    printf("[INFO] Loaded %d ratings. Found %d users and %d movies.\n", 
           total_ratings, num_users, num_movies);

    // Validate User ID
    if (target_user < 0 || target_user >= num_users) {
        printf("[ERROR] User ID must be between 1 and %d.\n", num_users);
        free(dataset);
        free(movie_data);
        return EXIT_FAILURE;
    }

        // 2. Train / Test Split Logic
    Rating* train_set = malloc(total_ratings * sizeof(Rating));
    Rating* test_set = malloc(total_ratings * sizeof(Rating));
    int train_count = 0;
    int test_count = 0;

    for (int i = 0; i < total_ratings; i++) {
        // 20% chance to put the rating in the hidden test vault
        if ((rand() % 100) < 20) {
            test_set[test_count++] = dataset[i];
        } else {
            train_set[train_count++] = dataset[i];
        }
    }

    printf("[INFO] Data Split Complete: %d for Training, %d for Testing.\n", train_count, test_count);

    // 2. Allocate Matrices and Biases
    float* P = allocate_feature_matrix(num_users);
    float* Q = allocate_feature_matrix(num_movies);
    float* b_u = calloc(num_users, sizeof(float));
    float* b_i = calloc(num_movies, sizeof(float));
    if(b_u==NULL || b_i==NULL){
        perror("Failed to allocate biases.\n");
    }

    //print_user_history(target_user , dataset, total_ratings, movie_data);

    // 3. Train
    train_model(train_set, train_count, P, Q, num_users, num_movies,b_u,b_i);

    evaluate_model(test_set, test_count, P, Q, b_u, b_i);


    // 4. Predict
    predict_top_movies(target_user, dataset, total_ratings, P, Q, num_movies,movie_data,b_u,b_i);

    // 5. Clean up
    free(P);
    free(Q);
    free(dataset);
    free(movie_data);
    free(train_set); 
    free(test_set);


    return EXIT_SUCCESS;
}