"""Практическая работа №2: аналитический дашборд по 20 акциям российского рынка за 2024 год.

Дашборд 2x2:
  1. линейный график: динамика цены одной акции (по умолчанию SBER) по месяцам;
  2. столбчатый (sns.barplot): средний объём торгов по секторам, доверительный интервал 95 %;
  3. boxplot (sns.boxplot): распределение волатильности по секторам;
  4. scatter + регрессия (sns.regplot): зависимость волатильности от объёма торгов.

После графиков печатается текстовый вывод с аналитическими заключениями (он же
сохраняется в report.txt), сам дашборд записывается в dashboard.png (300 dpi).

Запуск:
    python dashboard.py                 # построить, сохранить и показать
    python dashboard.py --no-show       # без открытия окна (скрипты, CI)
    python dashboard.py --ticker YNDX   # динамика другой акции на графике 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats

from generate_data import generate_stocks

# ---------------------------------------------------------------------------
# Оформление
# ---------------------------------------------------------------------------

# Цвет закреплён за сектором и одинаков на всех графиках (цвет следует за сущностью).
# Первые 4 слота категориальной палитры, проверены валидатором (CVD, контраст).
SECTOR_ORDER = ["Нефтегаз", "Финансы", "IT", "Металлургия"]
SECTOR_COLORS = {
    "Нефтегаз": "#2a78d6",
    "Финансы": "#eb6834",
    "IT": "#1baf7a",
    "Металлургия": "#eda100",
}
INK = "#0b0b0b"  # основной текст
INK_SECONDARY = "#52514e"  # подписи, источники
GRID = "#e1e0d9"
NEUTRAL_POINT = "#6b6a66"  # точки без категории (график 4)
ACCENT = "#d03b3b"  # линия тренда и выделение риска

MONTHS_RU = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
SOURCE_NOTE = "Источник: stocks.csv (синтетические данные, сгенерированы np.random, seed=42)"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # корректно рисует кириллицу
            "figure.figsize": (10, 6),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def thousands(value: float, _pos: int | None = None) -> str:
    """12345 -> '12 345' (разделитель тысяч - неразрывный пробел)."""
    return f"{value:,.0f}".replace(",", " ")


def ru_num(value: float, decimals: int = 0) -> str:
    """Число в русском формате: 1234567.8 -> '1 234 567,8'."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def set_title(ax: Axes, text: str) -> None:
    ax.set_title(text, fontsize=13, fontweight="bold", loc="left", pad=12)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["date", "ticker", "sector", "price", "volume", "volatility"]


def load_data(path: Path) -> pd.DataFrame:
    """Читает stocks.csv (если файла нет - создаёт его) и проверяет данные."""
    if not path.exists():
        print(f"Файл {path} не найден - генерирую заново.")
        generate_stocks().to_csv(path, index=False)

    df = pd.read_csv(path, parse_dates=["date"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В {path} нет столбцов: {', '.join(missing)}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    for col in ("price", "volume", "volatility"):
        if (df[col] <= 0).any():
            raise ValueError(f"В столбце {col} есть неположительные значения")
    if df["ticker"].nunique() < 5:
        raise ValueError("По условию нужно минимум 5 тикеров")

    unknown = set(df["sector"]) - set(SECTOR_ORDER)
    if unknown:
        raise ValueError(f"Нет цвета для секторов: {sorted(unknown)}")

    df = df.sort_values(["ticker", "date"], ignore_index=True)
    df["volume_mln"] = df["volume"] / 1e6
    df["volatility_pct"] = df["volatility"] * 100
    df["sector"] = pd.Categorical(df["sector"], categories=SECTOR_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def plot_price_dynamics(ax: Axes, df: pd.DataFrame, ticker: str) -> None:
    """График 1 (линейный): цена одной акции по месяцам."""
    data = df[df["ticker"] == ticker].sort_values("date")
    if data.empty:
        raise ValueError(f"Тикер {ticker} не найден. Доступны: {', '.join(sorted(df['ticker'].unique()))}")

    sector = data["sector"].iloc[0]
    color = SECTOR_COLORS[sector]
    x = data["date"].dt.month.to_numpy()
    y = data["price"].to_numpy()

    ax.plot(x, y, marker="o", markersize=7, linewidth=2, color=color, markeredgecolor="white")
    ax.set_xticks(range(1, 13), MONTHS_RU)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel("Месяц 2024 г. (цена на конец месяца)")
    ax.set_ylabel("Цена акции, руб.")
    set_title(ax, f"Динамика цены {ticker} ({sector}) по месяцам")

    # Выборочные подписи: минимум, максимум и последнее значение (не число на каждой точке).
    marks = {int(np.argmin(y)): ("мин.", "top"), int(np.argmax(y)): ("макс.", "bottom")}
    marks.setdefault(len(y) - 1, ("дек.", "bottom"))
    for i, (tag, va) in marks.items():
        offset = -12 if va == "top" else 12
        ax.annotate(
            f"{tag}: {ru_num(y[i], 2)}",
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            va=va,
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )

    pad = (y.max() - y.min()) * 0.18 or y.max() * 0.05
    ax.set_ylim(y.min() - pad, y.max() + pad)

    change = (y[-1] / y[0] - 1) * 100
    ax.text(
        0.02,
        0.04,
        f"За год: {change:+.1f} %".replace(".", ","),
        transform=ax.transAxes,
        fontsize=10.5,
        va="bottom",
        color=INK_SECONDARY,
    )


def plot_volume_by_sector(ax: Axes, df: pd.DataFrame) -> None:
    """График 2 (barplot): средний месячный объём торгов по секторам, ДИ 95 %."""
    sns.barplot(
        data=df,
        x="sector",
        y="volume_mln",
        hue="sector",  # цвет по секторам
        order=SECTOR_ORDER,
        hue_order=SECTOR_ORDER,
        palette=SECTOR_COLORS,
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"color": INK_SECONDARY, "linewidth": 1.5},
        edgecolor="white",
        linewidth=2,
        dodge=False,
        legend=False,
        ax=ax,
    )
    set_title(ax, "Средний объём торгов по секторам")
    ax.set_xlabel("Сектор экономики")
    ax.set_ylabel("Объём торгов, млн шт. в месяц")

    # Подписи над доверительным интервалом: верхний край усов берём у линий seaborn
    # (центр линии -> номер столбца).
    means = df.groupby("sector", observed=True)["volume_mln"].mean().reindex(SECTOR_ORDER)
    tops = means.to_numpy().copy()
    for line in ax.lines:
        xs, ys = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        ok = np.isfinite(xs) & np.isfinite(ys)  # в путь усов с засечками seaborn вставляет NaN
        if ok.any():
            idx = int(round(xs[ok].mean()))
            if 0 <= idx < len(tops):
                tops[idx] = max(tops[idx], ys[ok].max())
    ymax = tops.max()
    for i, sector in enumerate(SECTOR_ORDER):
        ax.text(i, tops[i] + ymax * 0.02, ru_num(means[sector], 1), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0, ymax * 1.18)
    ax.text(0.98, 0.96, "усы: 95 %-й доверительный интервал", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK_SECONDARY)


def plot_volatility_box(ax: Axes, df: pd.DataFrame) -> str:
    """График 3 (boxplot): распределение волатильности по секторам.

    Возвращает сектор с самым высоким риском (по медиане волатильности).
    """
    sns.boxplot(
        data=df,
        x="sector",
        y="volatility_pct",
        hue="sector",
        order=SECTOR_ORDER,
        hue_order=SECTOR_ORDER,
        palette=SECTOR_COLORS,
        width=0.55,
        linewidth=1.4,
        saturation=1,
        flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "white",
                    "markeredgecolor": INK_SECONDARY},
        medianprops={"color": INK, "linewidth": 2},
        dodge=False,
        legend=False,
        ax=ax,
    )
    set_title(ax, "Распределение волатильности по секторам")
    ax.set_xlabel("Сектор экономики")
    ax.set_ylabel("Дневная волатильность, %")

    # ВЫВОД ПО РИСКУ (по условию - в комментарии к коду):
    # самый высокий риск у сектора IT - у него максимальные медиана и верхний квартиль,
    # а самые резкие выбросы (одиночные «шоковые» месяцы) уходят выше 5-7 %.
    # Нефтегаз и Финансы спокойнее; в них выбросы редки и заметно ниже.
    # Сектор в коде определяется автоматически, чтобы вывод не расходился с данными.
    medians = df.groupby("sector", observed=True)["volatility_pct"].median()
    riskiest = str(medians.idxmax())

    # Подпись самого сильного выброса и стрелка на сектор с наибольшим риском.
    peak = df.loc[df["volatility_pct"].idxmax()]
    ax.set_ylim(1, df["volatility_pct"].max() * 1.12)
    ax.annotate(
        f"{peak['ticker']}, {MONTHS_RU[peak['date'].month - 1].lower()}: {ru_num(peak['volatility_pct'], 1)} %",
        (SECTOR_ORDER.index(peak["sector"]), peak["volatility_pct"]),
        textcoords="offset points", xytext=(18, -2), ha="left", va="center",
        fontsize=9.5, color=INK,
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.8},
    )
    idx = SECTOR_ORDER.index(riskiest)
    ax.text(
        idx, 1.1,
        "▲ наибольший риск", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=ACCENT,
    )
    ax.text(0.02, 0.96, "точки - выбросы (за 1,5 × IQR)", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=INK_SECONDARY)
    return riskiest


@dataclass(frozen=True)
class Relationship:
    r: float
    p_value: float
    rho: float
    slope_per_mln: float
    n: int
    within: dict[str, float]

    @property
    def verdict(self) -> str:
        strength = abs(self.r)
        word = "сильная" if strength >= 0.6 else "умеренная" if strength >= 0.3 else "слабая"
        sign = "отрицательная" if self.r < 0 else "положительная"
        return f"{word} {sign}" if self.p_value < 0.05 else "статистически не значима"


def plot_volume_vs_volatility(ax: Axes, df: pd.DataFrame) -> Relationship:
    """График 4 (regplot): зависимость волатильности от объёма торгов."""
    sns.regplot(
        data=df,
        x="volume_mln",
        y="volatility_pct",
        scatter_kws={"alpha": 0.45, "s": 40, "color": NEUTRAL_POINT, "edgecolor": "white", "linewidths": 0.5},
        line_kws={"color": ACCENT, "linewidth": 2.5},
        ci=95,
        ax=ax,
    )
    set_title(ax, "Влияние объёма торгов на волатильность")
    ax.set_xlabel("Объём торгов, млн шт. в месяц")
    ax.set_ylabel("Дневная волатильность, %")

    r, p_value = stats.pearsonr(df["volume_mln"], df["volatility_pct"])
    rho = stats.spearmanr(df["volume_mln"], df["volatility_pct"]).statistic
    slope = stats.linregress(df["volume_mln"], df["volatility_pct"]).slope
    within = {
        str(s): float(g["volume_mln"].corr(g["volatility_pct"]))
        for s, g in df.groupby("sector", observed=True)
    }
    rel = Relationship(float(r), float(p_value), float(rho), float(slope), len(df), within)

    p_text = "p < 0,001" if p_value < 0.001 else f"p = {ru_num(p_value, 3)}"
    ax.text(
        0.97, 0.96,
        f"r (Пирсон) = {ru_num(r, 2)}\nρ (Спирмен) = {ru_num(rho, 2)}\n{p_text}, n = {rel.n}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID},
    )
    ax.text(0.03, 0.03, "полоса: 95 %-й доверительный интервал регрессии",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK_SECONDARY)
    return rel


# ---------------------------------------------------------------------------
# Сборка дашборда
# ---------------------------------------------------------------------------


def build_dashboard(df: pd.DataFrame, ticker: str) -> tuple[Figure, str, Relationship]:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))

    plot_price_dynamics(axes[0, 0], df, ticker)
    plot_volume_by_sector(axes[0, 1], df)
    riskiest = plot_volatility_box(axes[1, 0], df)
    relationship = plot_volume_vs_volatility(axes[1, 1], df)

    fig.suptitle(
        "Аналитический дашборд: российский рынок акций, 2024 г.",
        fontsize=18, fontweight="bold", y=0.993,
    )
    fig.text(0.5, 0.952, f"{df['ticker'].nunique()} акций, {df['sector'].nunique()} сектора, "
             f"помесячные данные ({len(df)} наблюдений)",
             ha="center", fontsize=11.5, color=INK_SECONDARY)
    fig.text(0.01, 0.006, SOURCE_NOTE, fontsize=9, color=INK_SECONDARY, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.02, 1, 0.945), h_pad=3, w_pad=3)
    return fig, riskiest, relationship


# ---------------------------------------------------------------------------
# Текстовые выводы
# ---------------------------------------------------------------------------


def build_conclusions(df: pd.DataFrame, riskiest: str, rel: Relationship) -> str:
    vol = df.groupby("sector", observed=True)["volatility_pct"].agg(["median", "mean"])
    calmest = str(vol["median"].idxmin())

    avg_price = df.groupby("ticker")["price"].mean()
    priciest = str(avg_price.idxmax())
    cheapest = str(avg_price.idxmin())
    top_price = df.loc[(df["ticker"] == priciest), "price"].mean()

    sector_volume = df.groupby("sector", observed=True)["volume_mln"].mean()
    busiest = str(sector_volume.idxmax())
    quietest = str(sector_volume.idxmin())

    year = df.sort_values("date").groupby("ticker")["price"].agg(["first", "last"])
    change = (year["last"] / year["first"] - 1) * 100
    best, worst = str(change.idxmax()), str(change.idxmin())

    within = ", ".join(f"{s} {ru_num(v, 2)}" for s, v in rel.within.items())
    within_sign = np.mean(list(rel.within.values()))
    link = (
        "Внутри секторов связь слабее и по знаку другая: " + within + "."
        if np.sign(within_sign) != np.sign(rel.r)
        else "Внутри секторов связь того же знака: " + within + "."
    )

    lines = [
        "АНАЛИТИЧЕСКИЕ ВЫВОДЫ",
        "=" * 60,
        f"1. Самый волатильный сектор - {riskiest}: медиана дневной волатильности "
        f"{ru_num(vol.loc[riskiest, 'median'], 2)} % (среднее {ru_num(vol.loc[riskiest, 'mean'], 2)} %). "
        f"Самый спокойный - {calmest} ({ru_num(vol.loc[calmest, 'median'], 2)} %). "
        f"Разница в риске между ними - в {ru_num(vol.loc[riskiest, 'median'] / vol.loc[calmest, 'median'], 1)} раза.",
        f"2. Самая дорогая акция - {priciest}: средняя цена за 2024 г. {ru_num(top_price, 0)} руб. "
        f"(самая дешёвая - {cheapest}, {ru_num(avg_price[cheapest], 4 if avg_price[cheapest] < 1 else 2)} руб.). "
        "Цена одной бумаги не говорит о размере компании: она зависит от числа выпущенных акций.",
        f"3. По объёму торгов лидирует сектор {busiest} ({ru_num(sector_volume[busiest], 1)} млн шт. в месяц), "
        f"меньше всего торгуется сектор {quietest} ({ru_num(sector_volume[quietest], 1)} млн шт.). "
        "Широкие доверительные интервалы на графике 2 показывают, что внутри сектора бумаги сильно различаются по ликвидности.",
        f"4. Связь объёма и волатильности по всем наблюдениям: r = {ru_num(rel.r, 2)}, "
        f"ρ = {ru_num(rel.rho, 2)}, {'p < 0,001' if rel.p_value < 0.001 else 'p = ' + ru_num(rel.p_value, 3)}. "
        f"Это {rel.verdict} корреляция: каждый дополнительный миллион акций в месяц "
        f"сопровождается изменением волатильности на {ru_num(rel.slope_per_mln, 3)} п.п. "
        + link
        + (
            f" Общий тренд во многом объясняется различиями секторов: самый волатильный сектор {riskiest} "
            f"торгуется объёмом {ru_num(sector_volume[riskiest], 1)} млн шт. против "
            f"{ru_num(df['volume_mln'].mean(), 1)} млн в среднем по рынку."
            if np.sign(within_sign) != np.sign(rel.r)
            else ""
        ),
        f"5. По динамике цен за год лучшая бумага - {best} ({ru_num(change[best], 1)} %), "
        f"худшая - {worst} ({ru_num(change[worst], 1)} %). Корреляция не означает причинность, а данные "
        "синтетические: выводы иллюстрируют метод и не являются инвестиционной рекомендацией.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Дашборд по акциям российского рынка за 2024 год")
    parser.add_argument("--data", type=Path, default=Path("stocks.csv"), help="входной CSV")
    parser.add_argument("--output", type=Path, default=Path("dashboard.png"), help="файл дашборда")
    parser.add_argument("--report", type=Path, default=Path("report.txt"), help="файл с выводами")
    parser.add_argument("--ticker", default="SBER", help="акция для графика 1")
    parser.add_argument("--no-show", action="store_true", help="не открывать окно с графиком")
    args = parser.parse_args()

    setup_style()
    df = load_data(args.data)
    fig, riskiest, relationship = build_dashboard(df, args.ticker.upper())

    # Сохраняем ДО plt.show(): после показа фигура может быть очищена.
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Дашборд сохранён: {args.output}\n")

    conclusions = build_conclusions(df, riskiest, relationship)
    print(conclusions)
    args.report.write_text(conclusions + "\n", encoding="utf-8")

    if not args.no_show:
        plt.show()  # один вызов в самом конце
    plt.close(fig)


if __name__ == "__main__":
    main()
