"""Ex.No 8: Ensembling techniques - bagging, boosting, stacking and voting.

Compares four ensembles against their single base models on the
Wisconsin breast-cancer dataset (built into scikit-learn, no download needed).
"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def base_models():
    return [
        ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ]


def ensemble_models():
    return {
        "Bagging (50 trees)": BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=50, random_state=42
        ),
        "Boosting (AdaBoost)": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Stacking": StackingClassifier(
            base_models(), final_estimator=LogisticRegression(max_iter=2000)
        ),
        "Voting (hard)": VotingClassifier(base_models(), voting="hard"),
    }


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
    )

    print("Single base models")
    print("-" * 52)
    for name, model in base_models():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<22} test accuracy = {acc:.4f}")

    print("\nEnsemble models")
    print("-" * 52)
    for name, model in ensemble_models().items():
        cv = cross_val_score(model, X_train, y_train, cv=5).mean()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:<22} 5-fold CV = {cv:.4f}   test accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
