# Titanic Survival Project

A Streamlit dashboard for exploring Titanic passenger survival patterns, comparing three trained classifiers, and generating a majority-vote prediction for a passenger profile.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard expects `Titanic_Dataset.csv` and the trained model files (`lregression.pkl`, `knn.pkl`, `svm_model.pkl`, and `scaler.pkl`) in the project directory.

## Project files

- `app.py`: Streamlit dashboard and passenger prediction interface.
- `Titanic Survival Project.py`: model training, evaluation, and exploration workflow.
- `Trainmodel.py`: command-line passenger prediction script.
- `Titanic_Dataset.csv`: Titanic passenger dataset.
- `*.pkl`: trained classifiers and feature scaler used by the prediction scripts.