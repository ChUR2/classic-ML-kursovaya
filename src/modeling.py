#Общий каркас моделирования для всех задач проекта.

#Модуль задаёт единый протокол эксперимента, чтобы результаты семи задач
#были сопоставимы между собой:

#* групповое разбиение train/test и групповая кросс-валидация
#  (группа = уникальный набор дескрипторов, см. ``data_utils.molecule_groups``);
#* единый набор моделей-кандидатов с сетками гиперпараметров;
#* подбор гиперпараметров через ``RandomizedSearchCV`` с фиксированным seed;
#* единый набор метрик и графиков.


from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 30
TEST_SIZE = 0.2


# Разбиение данных

def group_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    groups: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """Разбить выборку на train/test так, чтобы группы не пересекались.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, groups_train)``.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(features, target, groups))
    return (
        features.iloc[train_idx],
        features.iloc[test_idx],
        target.iloc[train_idx],
        target.iloc[test_idx],
        groups[train_idx],
    )


def make_cv(stratified: bool = False):
    """Вернуть объект групповой кросс-валидации нужного типа."""
    if stratified:
        return StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
    return GroupKFold(n_splits=N_SPLITS)


# --------------------------------------------------------------------------
# Пайплайны
# --------------------------------------------------------------------------
def scaled_pipeline(estimator) -> Pipeline:
    """Пайплайн для моделей, чувствительных к масштабу признаков."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def tree_pipeline(estimator) -> Pipeline:
    """Пайплайн для древесных моделей: шкалирование не требуется."""
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    )


# --------------------------------------------------------------------------
# Каталог моделей
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """Модель-кандидат вместе с сеткой гиперпараметров."""

    name: str
    pipeline: Pipeline
    param_dist: dict[str, Any] = field(default_factory=dict)
    comment: str = ""


def regression_candidates() -> list[Candidate]:
    """Набор моделей для задач регрессии."""
    return [
        Candidate(
            "Baseline (медиана)",
            tree_pipeline(DummyRegressor(strategy="median")),
            {},
            "нижняя граница качества: константный прогноз",
        ),
        Candidate(
            "Ridge",
            scaled_pipeline(Ridge(random_state=RANDOM_STATE)),
            {"model__alpha": loguniform(1e-2, 1e5)},
            "линейная модель с L2-регуляризацией",
        ),
        Candidate(
            "Lasso",
            scaled_pipeline(Lasso(random_state=RANDOM_STATE, max_iter=20000)),
            {"model__alpha": loguniform(1e-4, 1e1)},
            "L1: одновременно выполняет отбор признаков",
        ),
        Candidate(
            "ElasticNet",
            scaled_pipeline(
                ElasticNet(random_state=RANDOM_STATE, max_iter=20000)
            ),
            {
                "model__alpha": loguniform(1e-4, 1e1),
                "model__l1_ratio": uniform(0.05, 0.9),
            },
            "компромисс L1/L2, устойчив к мультиколлинеарности",
        ),
        Candidate(
            "kNN",
            scaled_pipeline(KNeighborsRegressor()),
            {
                "model__n_neighbors": randint(3, 40),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "метрическая модель, проверка локальной структуры данных",
        ),
        Candidate(
            "SVR (RBF)",
            scaled_pipeline(SVR(kernel="rbf")),
            {
                "model__C": loguniform(1e-1, 1e3),
                "model__gamma": loguniform(1e-4, 1e0),
                "model__epsilon": uniform(0.01, 0.5),
            },
            "нелинейная модель на ядре, устойчива при высокой размерности",
        ),
        Candidate(
            "Random Forest",
            tree_pipeline(
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
            ),
            {
                "model__n_estimators": randint(200, 700),
                "model__max_depth": [None, 6, 10, 16, 24],
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", 0.3, 0.6],
            },
            "бэггинг деревьев, устойчив к переобучению и коллинеарности",
        ),
        Candidate(
            "Extra Trees",
            tree_pipeline(
                ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)
            ),
            {
                "model__n_estimators": randint(200, 700),
                "model__max_depth": [None, 8, 14, 22],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", 0.3, 0.6],
            },
            "усиленная рандомизация разбиений, меньше дисперсия",
        ),
        Candidate(
            "LightGBM",
            tree_pipeline(
                LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
            ),
            {
                "model__n_estimators": randint(150, 900),
                "model__learning_rate": loguniform(5e-3, 2e-1),
                "model__num_leaves": randint(7, 64),
                "model__min_child_samples": randint(5, 50),
                "model__subsample": uniform(0.6, 0.4),
                "model__subsample_freq": [1],
                "model__colsample_bytree": uniform(0.4, 0.6),
                "model__reg_lambda": loguniform(1e-3, 1e2),
            },
            "градиентный бустинг, основной кандидат на лучшую модель",
        ),
        Candidate(
            "XGBoost",
            tree_pipeline(
                XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0,
                )
            ),
            {
                "model__n_estimators": randint(150, 900),
                "model__learning_rate": loguniform(5e-3, 2e-1),
                "model__max_depth": randint(2, 9),
                "model__min_child_weight": randint(1, 15),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.4, 0.6),
                "model__reg_lambda": loguniform(1e-3, 1e2),
            },
            "альтернативная реализация бустинга для перекрёстной проверки",
        ),
    ]


def classification_candidates() -> list[Candidate]:
    """Набор моделей для задач бинарной классификации."""
    return [
        Candidate(
            "Baseline (частый класс)",
            tree_pipeline(
                DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
            ),
            {},
            "нижняя граница качества: ROC-AUC = 0.5 по построению",
        ),
        Candidate(
            "Logistic Regression",
            scaled_pipeline(
                LogisticRegression(
                    max_iter=5000, random_state=RANDOM_STATE,
                    solver="liblinear",
                )
            ),
            {
                "model__C": loguniform(1e-3, 1e2),
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
            "линейный baseline с регуляризацией",
        ),
        Candidate(
            "kNN",
            scaled_pipeline(KNeighborsClassifier()),
            {
                "model__n_neighbors": randint(3, 45),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "метрическая модель",
        ),
        Candidate(
            "SVC (RBF)",
            scaled_pipeline(SVC(kernel="rbf", probability=True,
                                random_state=RANDOM_STATE)),
            {
                "model__C": loguniform(1e-1, 1e3),
                "model__gamma": loguniform(1e-4, 1e0),
                "model__class_weight": [None, "balanced"],
            },
            "нелинейная модель на ядре",
        ),
        Candidate(
            "Random Forest",
            tree_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
            ),
            {
                "model__n_estimators": randint(200, 700),
                "model__max_depth": [None, 6, 10, 16, 24],
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", 0.3],
                "model__class_weight": [None, "balanced"],
            },
            "бэггинг деревьев",
        ),
        Candidate(
            "Extra Trees",
            tree_pipeline(
                ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)
            ),
            {
                "model__n_estimators": randint(200, 700),
                "model__max_depth": [None, 8, 14, 22],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", 0.3],
                "model__class_weight": [None, "balanced"],
            },
            "усиленная рандомизация разбиений",
        ),
        Candidate(
            "LightGBM",
            tree_pipeline(
                LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
            ),
            {
                "model__n_estimators": randint(150, 900),
                "model__learning_rate": loguniform(5e-3, 2e-1),
                "model__num_leaves": randint(7, 64),
                "model__min_child_samples": randint(5, 50),
                "model__subsample": uniform(0.6, 0.4),
                "model__subsample_freq": [1],
                "model__colsample_bytree": uniform(0.4, 0.6),
                "model__reg_lambda": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
            "градиентный бустинг, основной кандидат",
        ),
        Candidate(
            "XGBoost",
            tree_pipeline(
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    eval_metric="logloss",
                    verbosity=0,
                )
            ),
            {
                "model__n_estimators": randint(150, 900),
                "model__learning_rate": loguniform(5e-3, 2e-1),
                "model__max_depth": randint(2, 9),
                "model__min_child_weight": randint(1, 15),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.4, 0.6),
                "model__reg_lambda": loguniform(1e-3, 1e2),
            },
            "альтернативная реализация бустинга",
        ),
    ]


# --------------------------------------------------------------------------
# Подбор гиперпараметров
# --------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "cache"


def _cache_path(task: str, model_name: str) -> Path:
    """Путь к файлу кэша результата поиска для пары (задача, модель)."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in model_name)
    return CACHE_DIR / f"{task}__{safe}.json"


def search_candidates(
    candidates: list[Candidate],
    features: pd.DataFrame,
    target: pd.Series,
    groups: np.ndarray,
    scoring: str,
    stratified: bool = False,
    n_iter: int = N_ITER,
    verbose: bool = True,
    task: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Подобрать гиперпараметры каждой модели групповой кросс-валидацией.

    Если задан ``task``, результат подбора для каждой модели кэшируется в
    ``results/cache``. Повторный запуск переиспользует найденные
    гиперпараметры и заново обучает только финальную модель, что делает
    ноутбук быстро воспроизводимым. Для полного пересчёта достаточно
    очистить каталог кэша.

    Returns
    -------
    tuple
        Таблица со сводкой по каждой модели и словарь обученных
        лучших оценщиков.
    """
    cv = make_cv(stratified=stratified)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {}

    for cand in candidates:
        cached = None
        if task is not None:
            path = _cache_path(task, cand.name)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))

        if cached is not None:
            estimator = cand.pipeline.set_params(
                **{f"model__{k}": v for k, v in cached["params"].items()}
            ).fit(features, target)
            score, std = cached["score"], cached["std"]
            params, n_fits = cached["params"], cached["n_fits"]
        elif cand.param_dist:
            search = RandomizedSearchCV(
                cand.pipeline,
                cand.param_dist,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
                error_score=np.nan,
            )
            search.fit(features, target, groups=groups)
            estimator = search.best_estimator_
            score, std = search.best_score_, float(
                search.cv_results_["std_test_score"][search.best_index_]
            )
            params = {
                k.replace("model__", ""): _round(v)
                for k, v in search.best_params_.items()
            }
            n_fits = n_iter * cv.get_n_splits()
        else:
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(
                cand.pipeline, features, target, groups=groups,
                cv=cv, scoring=scoring, n_jobs=-1,
            )
            estimator = cand.pipeline.fit(features, target)
            score, std = float(scores.mean()), float(scores.std())
            params, n_fits = {}, cv.get_n_splits()

        if task is not None and cached is None:
            _cache_path(task, cand.name).write_text(
                json.dumps(
                    {"score": float(score), "std": float(std),
                     "params": params, "n_fits": int(n_fits)},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        best[cand.name] = estimator
        rows.append(
            {
                "модель": cand.name,
                f"CV {scoring}": score,
                "std": std,
                "лучшие гиперпараметры": params,
                "число обучений": n_fits,
                "комментарий": cand.comment,
            }
        )
        if verbose:
            print(f"{cand.name:24s} CV {scoring} = {score:7.4f} (±{std:.4f})")

    table = pd.DataFrame(rows).sort_values(f"CV {scoring}", ascending=False)
    return table.reset_index(drop=True), best


def _round(value):
    """Округлить числовые гиперпараметры для компактного вывода."""
    if isinstance(value, float):
        return round(value, 5)
    return value


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, inverse_log: bool = True
) -> dict[str, float]:
    """Метрики регрессии в логарифмической и, опционально, исходной шкале."""
    result = {
        "R2": r2_score(y_true, y_pred),
        "RMSE (log10)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE (log10)": mean_absolute_error(y_true, y_pred),
    }
    if inverse_log:
        true_orig = 10.0 ** np.asarray(y_true)
        pred_orig = 10.0 ** np.asarray(y_pred)
        result["MedAE (исх. ед.)"] = float(
            np.median(np.abs(true_orig - pred_orig))
        )
        result["медианная ошибка, раз"] = float(
            np.median(10.0 ** np.abs(np.asarray(y_true) - np.asarray(y_pred)))
        )
    return result


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Полный набор метрик бинарной классификации."""
    return {
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced acc.": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }


def evaluate_on_test(
    estimators: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "regression",
) -> pd.DataFrame:
    """Оценить все обученные модели на отложенной тестовой выборке."""
    rows = []
    for name, estimator in estimators.items():
        if task == "regression":
            metrics = regression_metrics(
                y_test.to_numpy(), estimator.predict(X_test)
            )
        else:
            proba = estimator.predict_proba(X_test)[:, 1]
            metrics = classification_metrics(
                y_test.to_numpy(), estimator.predict(X_test), proba
            )
        rows.append({"модель": name, **metrics})
    key = "R2" if task == "regression" else "ROC-AUC"
    return (
        pd.DataFrame(rows)
        .sort_values(key, ascending=False)
        .reset_index(drop=True)
    )


# Визуализация

def plot_cv_vs_test(
    cv_table: pd.DataFrame,
    test_table: pd.DataFrame,
    cv_col: str,
    test_col: str,
    title: str,
    save_path: str | None = None,
) -> None:
    """Столбчатая диаграмма: качество на кросс-валидации против теста."""
    merged = cv_table[["модель", cv_col]].merge(
        test_table[["модель", test_col]], on="модель"
    )
    merged = merged.sort_values(test_col)
    y = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(merged) + 1.6))
    ax.barh(y - 0.2, merged[cv_col], height=0.38, label="кросс-валидация",
            color="#4C72B0")
    ax.barh(y + 0.2, merged[test_col], height=0.38, label="отложенный тест",
            color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["модель"])
    ax.set_xlabel(test_col)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    save_path: str | None = None,
) -> None:
    """Диагностика регрессии: факт-прогноз, остатки, их распределение."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    lims = [min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred))]
    axes[0].scatter(y_true, y_pred, s=14, alpha=.55, color="#4C72B0")
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set(xlabel="факт (log10)", ylabel="прогноз (log10)",
                title=f"{label}: факт vs прогноз\n"
                      f"R2 = {r2_score(y_true, y_pred):.3f}")

    axes[1].scatter(y_pred, residuals, s=14, alpha=.55, color="#55A868")
    axes[1].axhline(0, color="r", ls="--", lw=1)
    axes[1].set(xlabel="прогноз (log10)", ylabel="остаток",
                title="Остатки против прогноза")

    axes[2].hist(residuals, bins=35, color="#DD8452", edgecolor="white")
    axes[2].axvline(0, color="r", ls="--", lw=1)
    axes[2].set(xlabel="остаток (log10)", ylabel="частота",
                title="Распределение остатков\n"
                      f"смещение = {residuals.mean():+.3f}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_classification_diagnostics(
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str,
    save_path: str | None = None,
) -> None:
    """ROC- и PR-кривые нескольких моделей на одном полотне."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, proba in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, lw=1.6,
                     label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
        precision, recall, _ = precision_recall_curve(y_true, proba)
        axes[1].plot(recall, precision, lw=1.6,
                     label=f"{name} (AP="
                           f"{average_precision_score(y_true, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="случайная модель")
    axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC-кривые: {title}")
    axes[0].legend(fontsize=8, loc="lower right")
    baseline = float(np.mean(y_true))
    axes[1].axhline(baseline, color="k", ls="--", lw=1,
                    label=f"доля класса 1 = {baseline:.2f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title=f"PR-кривые: {title}")
    axes[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_permutation_importance(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str,
    title: str,
    top_n: int = 20,
    n_repeats: int = 8,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Оценить и построить permutation importance лучшей модели.

    Permutation importance предпочтительнее встроенной важности деревьев:
    она измеряет реальное падение качества на отложенных данных и не
    завышает вклад признаков с большим числом уникальных значений.
    """
    result = permutation_importance(
        estimator, X_test, y_test, n_repeats=n_repeats,
        random_state=RANDOM_STATE, scoring=scoring, n_jobs=-1,
    )
    table = (
        pd.DataFrame(
            {
                "признак": X_test.columns,
                "падение качества": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("падение качества", ascending=False)
        .reset_index(drop=True)
    )
    top = table.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 0.34 * top_n + 1.4))
    plt.barh(top["признак"], top["падение качества"],
             xerr=top["std"], color="#8172B3", error_kw={"lw": .8})
    plt.xlabel(f"падение метрики {scoring} при перемешивании признака")
    plt.title(title)
    plt.grid(axis="x", alpha=.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    return table


def stability_check(
    estimator,
    features: pd.DataFrame,
    target: pd.Series,
    groups: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Проверить устойчивость модели к выбору разбиения train/test."""
    from sklearn.base import clone

    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te, _ = group_train_test_split(
            features, target, groups, random_state=seed
        )
        model = clone(estimator).fit(X_tr, y_tr)
        rows.append({"seed": seed, "метрика": scoring(y_te, model, X_te)})
    frame = pd.DataFrame(rows)
    print(f"Среднее = {frame['метрика'].mean():.4f}, "
          f"std = {frame['метрика'].std():.4f}, "
          f"размах = {frame['метрика'].max() - frame['метрика'].min():.4f}")
    return frame
