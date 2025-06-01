import warnings
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", module="joblib.externals.loky")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Hanya ambil yang Status-nya bukan "Enrolled"
    df = df[df['Status'] != 'Enrolled']
    # Map Status menjadi 1 (Dropout) dan 0 (Graduate)
    df['Status'] = df['Status'].map({'Dropout': 1, 'Graduate': 0})
    # Drop kolom yang memang tidak dipakai untuk modeling
    drop_cols = [
        'Nacionality', 'Application_order', 'Unemployment_rate', 
        'Inflation_rate', 'GDP', 'Mothers_qualification', 
        'Fathers_qualification', 'Mothers_occupation', 
        'Fathers_occupation', 'Debtor'
    ]
    df.drop(columns=drop_cols, inplace=True)
    return df


def build_preprocessors(cat_features, num_features):
    """
    Mengembalikan dua preprocessors:
     1) 'preprocessing' = hanya OneHotEncoder untuk model RFC/XGB
     2) 'robust_pre'   = OneHotEncoder + RobustScaler untuk Logistic Regression
    """
    # Untuk RandomForest & XGBoost (hanya OneHot + pass-through numeric)
    preprocessing = ColumnTransformer(
        transformers=[
            (
                'onehot',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                cat_features
            )
        ],
        remainder='passthrough'
    )

    # Untuk Logistic Regression (OneHot + RobustScaler untuk numeric saja)
    robust_pre = ColumnTransformer(
        transformers=[
            (
                'onehot',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                cat_features
            ),
            (
                'scale',
                RobustScaler(),
                num_features
            )
        ],
        remainder='drop'
    )

    return preprocessing, robust_pre


def resampling_scores(model, X, y, preprocessing, folds=5):
    """
    Menghitung cross-validated F2-score untuk sebuah model dengan SMOTE.
    """
    resampler = SMOTE(random_state=42)
    scorer = make_scorer(fbeta_score, beta=2)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    pipeline = ImbPipeline(
        steps=[
            ('prep', preprocessing),
            ('resample', resampler),
            ('model', model)
        ]
    )
    scores = cross_val_score(
        pipeline,
        X, y,
        cv=skf,
        scoring=scorer,
        n_jobs=-1
    )
    print(f"{model.__class__.__name__} CV Mean F2: {scores.mean():.4f}  Std: {scores.std():.4f}")


def tune_logistic(X_train, y_train, preprocessor):
    """
    Melakukan GridSearchCV untuk mencari hyperparameter terbaik Logistic Regression.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f2_scorer = make_scorer(fbeta_score, beta=2)
    resampler = SMOTE(random_state=42)

    pipe = ImbPipeline(
        steps=[
            ('prep', preprocessor),
            ('resample', resampler),
            ('model', LogisticRegression(random_state=42))
        ]
    )

    param_grid = {
        'model__penalty': ['l1', 'l2', 'elasticnet', None],
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['saga'],
        'model__l1_ratio': [0, 0.5, 1],
        'model__max_iter': [200]
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=skf,
        scoring=f2_scorer,
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X_train, y_train)

    print(f"Best F2: {grid.best_score_:.4f}")
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Fit model, lalu tampilkan classification report untuk Train dan Test.
    """
    model.fit(X_train, y_train)

    for split_name, (X_split, y_split) in zip(
        ['Train', 'Test'], 
        [(X_train, y_train), (X_test, y_test)]
    ):
        preds = model.predict(X_split)
        print(f"\n{split_name} Classification Report:\n", classification_report(y_split, preds))


def save_model(model, filename: str):
    """
    Simpan model ke file pickle.
    """
    pickle.dump(model, open(filename, 'wb'))
    print(f"Model saved to {filename}")


def main():
    # 1) Load data
    data_path = 'students_performance_cleaned.csv'
    model_path = 'students_performance_logreg.sav'

    df = load_data(data_path)

    # 2) Definisikan fitur:  
    cat_features = [
        'Marital_status', 'Application_mode', 'Daytime_evening_attendance',
        'Previous_qualification', 'Displaced', 'Educational_special_needs',
        'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
        'International', 'Course'
    ]

    num_features = [
        'Admission_grade', 'Previous_qualification_grade', 'Age_at_enrollment',
        'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade',
        'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_without_evaluations'
    ]

    # 3) Build preprocessors
    preprocessing, robust_pre = build_preprocessors(cat_features, num_features)

    # 4) Split data
    X = df.drop('Status', axis=1)
    y = df['Status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Baseline resampling evaluation (contoh dengan LogisticRegression)
    resampling_scores(LogisticRegression(random_state=42), X_train, y_train, robust_pre)

    # 6) Hyperparameter tuning untuk LogisticRegression
    best_logreg = tune_logistic(X_train, y_train, robust_pre)

    # 7) Evaluasi akhir
    evaluate_model(best_logreg, X_train, y_train, X_test, y_test)

    # 8) Simpan model terbaik
    save_model(best_logreg, model_path)


if __name__ == '__main__':
    main()
