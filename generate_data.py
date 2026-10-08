"""Генерация учебного датасета stocks.csv: 20 акций российского рынка за 2024 год.

Данные синтетические (np.random, фиксированный seed), но устроены правдоподобно:
  * у каждого сектора свой типичный уровень волатильности (IT рискованнее нефтегаза);
  * у каждой бумаги свой типичный объём торгов, от месяца к месяцу он колеблется;
  * цена идёт случайным блужданием, размах которого задаётся волатильностью;
  * иногда случаются «шоковые» месяцы с резким скачком волатильности (выбросы).

Первые два месяца для пяти бумаг из методички (GAZP, SBER, LKOH, YNDX, GMKN)
заданы точно так, как в условии задания.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# тикер -> (сектор, стартовая цена, руб.; типичный месячный объём, шт.)
TICKERS: dict[str, tuple[str, float, float]] = {
    # Нефтегаз
    "GAZP": ("Нефтегаз", 165.40, 12_500_000),
    "LKOH": ("Нефтегаз", 6780.00, 2_500_000),
    "ROSN": ("Нефтегаз", 560.00, 4_200_000),
    "TATN": ("Нефтегаз", 650.00, 3_100_000),
    "SNGS": ("Нефтегаз", 22.50, 15_000_000),
    # Финансы
    "SBER": ("Финансы", 285.60, 18_000_000),
    "VTBR": ("Финансы", 0.024, 9_000_000),
    "MOEX": ("Финансы", 225.00, 3_800_000),
    "TCSG": ("Финансы", 3100.00, 1_900_000),
    "CBOM": ("Финансы", 6.20, 11_000_000),
    # IT
    "YNDX": ("IT", 4120.00, 1_200_000),
    "OZON": ("IT", 2900.00, 1_500_000),
    "VKCO": ("IT", 560.00, 1_100_000),
    "POSI": ("IT", 3200.00, 700_000),
    "HEAD": ("IT", 4300.00, 600_000),
    # Металлургия
    "GMKN": ("Металлургия", 14200.00, 800_000),
    "NLMK": ("Металлургия", 190.00, 5_300_000),
    "CHMF": ("Металлургия", 1700.00, 1_400_000),
    "PLZL": ("Металлургия", 13500.00, 650_000),
    "MAGN": ("Металлургия", 46.00, 8_000_000),
}

# Средняя дневная волатильность по секторам (доля, 0.02 = 2 %)
SECTOR_VOLATILITY: dict[str, float] = {
    "Нефтегаз": 0.021,
    "Финансы": 0.019,
    "IT": 0.032,
    "Металлургия": 0.026,
}

# Заданные в условии строки: (тикер, дата) -> (цена, объём, волатильность)
FIXED_ROWS: dict[tuple[str, str], tuple[float, int, float]] = {
    ("GAZP", "2024-01-31"): (165.40, 12_500_000, 0.023),
    ("SBER", "2024-01-31"): (285.60, 18_000_000, 0.018),
    ("LKOH", "2024-01-31"): (6780.00, 2_500_000, 0.021),
    ("YNDX", "2024-01-31"): (4120.00, 1_200_000, 0.031),
    ("GMKN", "2024-01-31"): (14200.00, 800_000, 0.026),
    ("GAZP", "2024-02-29"): (170.20, 13_200_000, 0.024),
    ("SBER", "2024-02-29"): (295.30, 19_200_000, 0.019),
    ("LKOH", "2024-02-29"): (6890.00, 2_650_000, 0.022),
    ("YNDX", "2024-02-29"): (4250.00, 1_350_000, 0.033),
    ("GMKN", "2024-02-29"): (14500.00, 850_000, 0.027),
}

TRADING_DAYS_PER_MONTH = 21
PRICE_DAMPING = 0.5
SHOCK_PROBABILITY = 0.04
SHOCK_MULTIPLIER = (1.6, 2.3)


def generate_stocks(seed: int = SEED) -> pd.DataFrame:
    """Возвращает длинную таблицу date, ticker, sector, price, volume, volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="ME")
    rows: list[dict] = []

    for ticker, (sector, start_price, base_volume) in TICKERS.items():
        price = start_price
        for date in dates:
            key = (ticker, date.strftime("%Y-%m-%d"))

            if key in FIXED_ROWS:
                price, volume, volatility = FIXED_ROWS[key]
            else:
                volatility = rng.normal(SECTOR_VOLATILITY[sector], 0.0025)
                if rng.random() < SHOCK_PROBABILITY:  # шоковый месяц
                    volatility *= rng.uniform(*SHOCK_MULTIPLIER)
                volatility = float(np.clip(volatility, 0.008, 0.09))

                # месячный разброс цены: дневная волатильность * sqrt(дней), с демпфированием
                # (на практике внутримесячные колебания частично возвращаются к среднему)
                monthly_sigma = PRICE_DAMPING * volatility * np.sqrt(TRADING_DAYS_PER_MONTH)
                price *= float(np.exp(rng.normal(0.004, monthly_sigma) - monthly_sigma**2 / 2))
                # в «нервные» месяцы торгуют активнее: слабая связь внутри бумаги
                nervousness = 8.0 * (volatility - SECTOR_VOLATILITY[sector])
                volume = int(base_volume * rng.lognormal(nervousness, 0.18))

            # у дешёвых бумаг (VTBR) нужна точность до 4 знаков, у остальных хватает 2
            decimals = 4 if start_price < 1 else 2
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sector": sector,
                    "price": round(price, decimals),
                    "volume": int(volume),
                    "volatility": round(volatility, 3),
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "ticker"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация stocks.csv")
    parser.add_argument("--output", type=Path, default=Path("stocks.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = generate_stocks(args.seed)
    df.to_csv(args.output, index=False)
    print(f"Сохранено {len(df)} строк ({df['ticker'].nunique()} тикеров) в {args.output}")


if __name__ == "__main__":
    main()
