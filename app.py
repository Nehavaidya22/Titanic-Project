from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Titanic_Dataset.csv"
FEATURE_COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]
MODEL_FILES = {
    "Logistic Regression": "lregression.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Support Vector Machine": "svm_model.pkl",
}


st.set_page_config(page_title="Titanic | Survival Lab", page_icon="🚢", layout="wide")


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    return data


@st.cache_resource
def load_models():
    models = {
        name: joblib.load(BASE_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return models, scaler


def make_features(data):
    features = pd.get_dummies(
        data[["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "Sex", "Embarked"]],
        columns=["Sex", "Embarked"],
        drop_first=True,
        dtype=int,
    )
    return features.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def model_scores(data, models, scaler):
    features = make_features(data)
    target = data["Survived"]
    _, test_features, _, test_target = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    scaled_features = scaler.transform(test_features)
    scores = []
    for name, model in models.items():
        predictions = model.predict(scaled_features)
        scores.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(test_target, predictions),
                "Precision": precision_score(test_target, predictions, zero_division=0),
                "Recall": recall_score(test_target, predictions, zero_division=0),
                "F1 score": f1_score(test_target, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(scores).set_index("Model")


st.title("Titanic | Survival Lab")
st.caption("Explore the passenger data, compare the trained classifiers, and test a passenger profile.")

try:
    passengers = load_dataset()
    models, scaler = load_models()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error.filename}")
    st.stop()

survival_rate = passengers["Survived"].mean()
survivors = int(passengers["Survived"].sum())
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Passengers", f"{len(passengers):,}")
metric_col2.metric("Survived", f"{survivors:,}")
metric_col3.metric("Overall survival", f"{survival_rate:.1%}")
metric_col4.metric("Models in vote", len(models))

overview_tab, performance_tab, prediction_tab = st.tabs(
    ["Passenger overview", "Model performance", "Try a passenger"]
)

with overview_tab:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Survival by passenger class")
        class_rates = passengers.groupby("Pclass")["Survived"].mean().rename("Survival rate")
        st.bar_chart(class_rates, y="Survival rate", color="#168C83")
    with chart_col2:
        st.subheader("Survival by sex")
        sex_rates = passengers.groupby("Sex")["Survived"].mean().rename("Survival rate")
        st.bar_chart(sex_rates, y="Survival rate", color="#E27A52")

    age_col, family_col = st.columns(2)
    with age_col:
        st.subheader("Survival across age groups")
        age_data = passengers.assign(
            AgeGroup=pd.cut(
                passengers["Age"],
                bins=[0, 12, 18, 35, 60, float("inf")],
                labels=["0-12", "13-18", "19-35", "36-60", "60+"],
            )
        )
        age_rates = age_data.groupby("AgeGroup", observed=False)["Survived"].mean()
        st.bar_chart(age_rates.rename("Survival rate"), y="Survival rate", color="#D2A53B")
    with family_col:
        st.subheader("Survival by family size")
        family_rates = passengers.groupby("FamilySize")["Survived"].mean().rename("Survival rate")
        st.bar_chart(family_rates, y="Survival rate", color="#5575A6")

    with st.expander("Browse passenger records"):
        st.dataframe(
            passengers[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]],
            use_container_width=True,
            hide_index=True,
        )

with performance_tab:
    st.subheader("Held-out test set")
    st.caption("Scores use the same stratified 80/20 split and random seed as the training notebook.")
    scores = model_scores(passengers, models, scaler)
    st.bar_chart(scores, y=["Accuracy", "Precision", "Recall", "F1 score"])
    st.dataframe(scores.style.format("{:.1%}"), use_container_width=True)

with prediction_tab:
    st.subheader("Passenger profile")
    with st.form("passenger_form"):
        input_col1, input_col2, input_col3 = st.columns(3)
        with input_col1:
            passenger_class = st.selectbox("Passenger class", [1, 2, 3], index=2)
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
            sex = st.selectbox("Sex", ["female", "male"])
        with input_col2:
            siblings_spouses = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0)
            parents_children = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0)
            embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])
        with input_col3:
            fare = st.number_input("Fare paid", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        submitted = st.form_submit_button("Predict survival", type="primary")

    if submitted:
        family_size = siblings_spouses + parents_children + 1
        profile = pd.DataFrame(
            [{
                "Pclass": passenger_class,
                "Age": age,
                "SibSp": siblings_spouses,
                "Parch": parents_children,
                "Fare": fare,
                "FamilySize": family_size,
                "IsAlone": int(family_size == 1),
                "Sex": sex,
                "Embarked": embarked,
            }]
        )
        scaled_profile = scaler.transform(make_features(profile))
        votes = {name: int(model.predict(scaled_profile)[0]) for name, model in models.items()}
        survived_votes = sum(votes.values())
        survived = survived_votes >= 2
        result_col, vote_col = st.columns([1, 2])
        with result_col:
            if survived:
                st.success("Predicted to survive")
            else:
                st.error("Predicted not to survive")
            st.metric("Survival votes", f"{survived_votes} / {len(votes)}")
        with vote_col:
            st.subheader("Classifier votes")
            vote_table = pd.DataFrame(
                {"Prediction": {name: "Survived" if vote else "Did not survive" for name, vote in votes.items()}}
            )
            st.dataframe(vote_table, use_container_width=True)