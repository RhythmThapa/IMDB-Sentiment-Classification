# IMDB Movie Reviews - Sentiment Classification

Classifies movie reviews as positive or negative using TF-IDF vectorization and classic ML models, on the IMDB Movie Reviews dataset.

## What I did
- Loaded and inspected the review text and sentiment labels
- Cleaned the text and removed duplicate reviews
- Vectorized reviews with TF-IDF
- Split into train/validation/test (70/15/15)
- Trained Multinomial Naive Bayes and Logistic Regression, compared on validation
- Selected the best model by F1-score, then evaluated once on the held-out test set

## Results
| Model | Val Accuracy | Val Precision | Val Recall | Val F1 | Training Time |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.876 | 0.867 | 0.889 | 0.878 | 0.08s |
| Logistic Regression | 0.907 | 0.894 | 0.924 | 0.909 | 4.16s |

Logistic Regression was selected on validation F1. Final test metrics: accuracy 0.909, precision 0.902, recall 0.919, F1 0.910, ROC-AUC 0.971.

## Structure
- `notebooks/` - main notebook
- `results/` - confusion matrix and ROC curve plots
- `models/` - saved model (`sentiment_model.joblib`) and TF-IDF vectorizer (`tfidf_vectorizer.joblib`)
