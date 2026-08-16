#Утилиты загрузки и предобработки данных курсового проекта.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

TARGET_IC50 = "IC50, mM"
TARGET_CC50 = "CC50, mM"
TARGET_SI = "SI"
TARGETS = (TARGET_IC50, TARGET_CC50, TARGET_SI)

ID_COLUMN = "Unnamed: 0"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset.xlsx"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Прочитать исходный файл с данными химиков.

    Parameters
    ----------
    path : Path | str
        Путь к xlsx-файлу.

    Returns
    -------
    pd.DataFrame
        Датафрейм со всеми исходными колонками, служебный индексный
        столбец удалён.
    """
    frame = pd.read_excel(path)
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
    return frame


def constant_columns(frame: pd.DataFrame) -> list[str]:
    """Вернуть список признаков с единственным уникальным значением."""
    return frame.columns[frame.nunique(dropna=False) <= 1].tolist()


def duplicated_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Найти полностью дублирующиеся колонки.

    Returns
    -------
    dict[str, str]
        Отображение "колонка-дубликат -> первая встреченная колонка".
    """
    duplicates: dict[str, str] = {}
    seen: dict[bytes, str] = {}
    for column in frame.columns:
        key = np.ascontiguousarray(
            frame[column].to_numpy(dtype="float64")
        ).round(12).tobytes()
        if key in seen:
            duplicates[column] = seen[key]
        else:
            seen[key] = column
    return duplicates


def near_constant_columns(
    frame: pd.DataFrame, threshold: float = 0.995
) -> list[str]:
    """Признаки, у которых доля самого частого значения выше порога."""
    shares = frame.apply(lambda col: col.value_counts(normalize=True).max())
    return shares[shares > threshold].index.tolist()


def correlated_columns(
    frame: pd.DataFrame, threshold: float = 0.98
) -> list[str]:
    """Отобрать колонки для удаления из-за почти полной коллинеарности.

    Из каждой пары признаков с |r| > threshold удаляется вторая колонка
    (в порядке следования), что сохраняет по одному представителю от
    каждой группы дублирующей информации.
    """
    corr = frame.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if (upper[col] > threshold).any()]


def clean_features(
    frame: pd.DataFrame,
    drop_correlated: bool = False,
    corr_threshold: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Очистить признаковое пространство от неинформативных колонок.

    Returns
    -------
    tuple
        Очищенный датафрейм и словарь с описанием удалённых групп.
    """
    features = frame.drop(columns=[t for t in TARGETS if t in frame.columns])
    report: dict[str, list[str]] = {}

    report["constant"] = constant_columns(features)
    features = features.drop(columns=report["constant"])

    report["duplicated"] = list(duplicated_columns(features))
    features = features.drop(columns=report["duplicated"])

    report["near_constant"] = near_constant_columns(features)
    features = features.drop(columns=report["near_constant"])

    if drop_correlated:
        report["collinear"] = correlated_columns(features, corr_threshold)
        features = features.drop(columns=report["collinear"])
    else:
        report["collinear"] = []

    return features, report


def build_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Сформировать все целевые переменные проекта.

    Регрессионные таргеты логарифмируются по основанию 10: исходные
    распределения сильно скошены вправо (log-normal), а десятичный
    логарифм соответствует принятой в фармакологии шкале pIC50.
    """
    targets = pd.DataFrame(index=frame.index)
    targets["ic50"] = frame[TARGET_IC50]
    targets["cc50"] = frame[TARGET_CC50]
    targets["si"] = frame[TARGET_SI]
    targets["log_ic50"] = np.log10(frame[TARGET_IC50])
    targets["log_cc50"] = np.log10(frame[TARGET_CC50])
    targets["log_si"] = np.log10(frame[TARGET_SI])
    targets["ic50_gt_median"] = (
        frame[TARGET_IC50] > frame[TARGET_IC50].median()
    ).astype(int)
    targets["cc50_gt_median"] = (
        frame[TARGET_CC50] > frame[TARGET_CC50].median()
    ).astype(int)
    targets["si_gt_median"] = (
        frame[TARGET_SI] > frame[TARGET_SI].median()
    ).astype(int)
    targets["si_gt_8"] = (frame[TARGET_SI] > 8).astype(int)
    return targets


def molecule_groups(features: pd.DataFrame) -> np.ndarray:
    """Присвоить каждому объекту идентификатор уникальной молекулы.

    В выборке 1001 строка описывает лишь 804 уникальных набора дескрипторов:
    одно и то же вещество (или неразличимые 2D-дескрипторами изомеры)
    встречается несколько раз с разными экспериментальными значениями.
    Идентификатор используется как ``groups`` в групповой кросс-валидации,
    чтобы копии одной молекулы не попадали одновременно в обучение и контроль.
    """
    key = features.fillna(-999.0).round(8).astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


def make_preprocessor(
    columns: list[str], scale: bool = True
) -> ColumnTransformer:
    """Собрать препроцессор: импутация и опциональное шкалирование."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), columns)],
        remainder="drop",
    )


def load_dataset(
    path: Path | str = DATA_PATH,
    drop_correlated: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    """Полный цикл подготовки данных.

    Returns
    -------
    tuple
        ``(X, y, groups, report)``  матрица признаков, таблица всех целевых
        переменных, идентификаторы уникальных молекул и отчёт об очистке.
    """
    raw = load_raw(path)
    groups = molecule_groups(raw.drop(columns=[t for t in TARGETS
                                               if t in raw.columns]))
    features, report = clean_features(raw, drop_correlated=drop_correlated)
    targets = build_targets(raw)
    return features, targets, groups, report
