# Naive Bayes Fake News Classifier (By Hand)

This project implements a Naive Bayes classifier **from scratch (by hand)** using Python to distinguish between **real** and **fake** news articles. No NLP libraries (e.g., TfidfVectorizer or CountVectorizer) are used for model building—only core Python data structures.

---

## Dataset

We use the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

- `True.csv`: Real news articles  
- `Fake.csv`: Fake news articles  
- Columns: `title`, `text`, `subject`, `date`

Both files are combined with an added binary label:

df_real['RealNews?'] = True
df_fake['RealNews?'] = False
df = df_real.append(df_fake)

## Preprocessing Steps

1.  **Combine** `title` and `text` into a single `document` column
    
2.  Convert all text to **lowercase**
    
3.  **Tokenize** using regex `re.split(r"\W+", document)`
    
4.  **Split** data into 80% train / 20% test
    

----------

## Naive Bayes Implementation (By Hand)

-   **Binary Classification**: Real (`1`) vs. Fake (`0`)
    
-   Uses **Laplace (add-1) smoothing**
    
-   Core logic:
    
    -   Calculate **prior probabilities**
        
    -   Calculate **word likelihoods** per class
        
    -   Compute **log probabilities** for prediction
        

----------

## Evaluation Metrics

Evaluation performed on the test set using:

-   **Precision**
    
-   **Recall**
    
-   **F1 Score**
