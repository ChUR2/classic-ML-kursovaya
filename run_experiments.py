"""Предварительный расчёт подбора гиперпараметров для всех семи задач.

Скрипт последовательно перебирает задачи и модели-кандидаты, сохраняя
результат каждого ``RandomizedSearchCV`` в ``results/cache``. Расчёт
резюмируемый: при повторном запуске уже посчитанные пары
(задача, модель) пропускаются. Ноутбуки используют тот же кэш, поэтому
выполняются за секунды, а полный пересчёт запускается очисткой каталога
``results/cache``.

Использование::

    python run_experiments.py            # обработать все задачи
    python run_experiments.py --budget 150   # остановиться через 150 секунд
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src import data_utils as du  # noqa: E402
from src import modeling as mdl  # noqa: E402

warnings.filterwarnings("ignore")

TASKS = [
    ("ic50_reg", "log_ic50", "regression", "r2"),
    ("cc50_reg", "log_cc50", "regression", "r2"),
    ("si_reg", "log_si", "regression", "r2"),
    ("si_reg_nocensored", "log_si", "regression", "r2"),
    ("ic50_clf", "ic50_gt_median", "classification", "roc_auc"),
    ("cc50_clf", "cc50_gt_median", "classification", "roc_auc"),
    ("si_clf", "si_gt_median", "classification", "roc_auc"),
    ("si8_clf", "si_gt_8", "classification", "roc_auc"),
]


def main() -> None:
    """Пройти по всем задачам и моделям, дозаполняя кэш результатов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=1e9,
                        help="ограничение по времени работы, секунд")
    parser.add_argument("--n-iter", type=int, default=mdl.N_ITER,
                        help="число итераций RandomizedSearchCV")
    args = parser.parse_args()

    started = time.time()
    features, targets, groups, _ = du.load_dataset()
    remaining = 0

    for task, column, kind, scoring in TASKS:
        subset = slice(None)
        if task == "si_reg_nocensored":
            keep = (targets["si"] - 1.0).abs() > 1e-9
            subset = keep.to_numpy()

        X_task = features[subset] if subset is not slice(None) else features
        y_task = targets.loc[subset, column] if subset is not slice(None) \
            else targets[column]
        g_task = groups[subset] if subset is not slice(None) else groups

        X_train, _, y_train, _, g_train = mdl.group_train_test_split(
            X_task, y_task, g_task
        )
        candidates = (
            mdl.regression_candidates() if kind == "regression"
            else mdl.classification_candidates()
        )

        for cand in candidates:
            if mdl._cache_path(task, cand.name).exists():
                continue
            if time.time() - started > args.budget:
                remaining += 1
                continue
            step = time.time()
            mdl.search_candidates(
                [cand], X_train, y_train, g_train, scoring,
                stratified=(kind == "classification"),
                n_iter=args.n_iter, verbose=False, task=task,
            )
            print(f"[{task:18s}] {cand.name:24s} готово за "
                  f"{time.time() - step:6.1f} с", flush=True)

    total = sum(
        1 for task, *_ in TASKS
        for cand in (mdl.regression_candidates()
                     if task.endswith("reg") or "reg" in task
                     else mdl.classification_candidates())
    )
    exists = mdl.CACHE_DIR.exists()
    done = len(list(mdl.CACHE_DIR.glob("*.json"))) if exists else 0
    print(f"\nГотово файлов кэша: {done}. Осталось незапущенных: {remaining}.")
    print(f"Общее время: {time.time() - started:.1f} с "
          f"(ориентир задач: {total})")


if __name__ == "__main__":
    main()
