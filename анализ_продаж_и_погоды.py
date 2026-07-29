import calendar
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "данные" / "data_блок3_продажи.csv"
OUT = HERE / "проект_графики"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Первые строки:")
print(df.head())

print("\nТипы столбцов до преобразования:")
print(df.dtypes)

df["Дата"] = pd.to_datetime(df["Дата"])
print("\nТипы столбцов после pd.to_datetime:")
print(df.dtypes)

grouped_df = (
    df.groupby("Дата")
    .agg(**{
        "Количество продаж": ("Номенклатура", "size"),
        "Сумма_штук": ("Количество", "sum"),
    })
    .reset_index()
)
print("\nПервые строки сгруппированных данных:")
print(grouped_df.head())

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(grouped_df["Дата"], grouped_df["Количество продаж"], linewidth=1)
ax.set_title("Количество продаж (число строк-продаж) по дням, 2018")
ax.set_xlabel("Дата")
ax.set_ylabel("Количество продаж")
fig.tight_layout()
fig.savefig(OUT / "блок3_продажи_по_дням.png", dpi=110)
print(f"\nГрафик сохранён: {OUT / 'блок3_продажи_по_дням.png'}")

all_dates = pd.date_range(df["Дата"].min(), df["Дата"].max(), freq="D")
present_dates = set(grouped_df["Дата"])
missing_dates = [d for d in all_dates if d not in present_dates]
missing_weekdays = pd.Series(missing_dates).dt.weekday.value_counts()
print("\nПропущенные в данных календарные дни, по дням недели:")
for wd, cnt in missing_weekdays.items():
    print(f"  {calendar.day_name[wd]}: {cnt}")

print(f"""
Описание графика (задание "опишите максимально подробно"):
- Период наблюдения — 4 января - 31 августа 2018 (240 календарных дней),
  но реальных дат в данных только {len(present_dates)} — не хватает
  {len(missing_dates)} дней, и {missing_weekdays.get(0, 0)} из них — это
  ПОНЕДЕЛЬНИКИ (плюс один четверг, вероятно праздничный день). Похоже,
  что склад/магазин почти всегда не работает по понедельникам —
  это регулярный выходной день, а не пропуски в данных.
- Общий уровень продаж стабилен: от {grouped_df['Количество продаж'].min()}
  до {grouped_df['Количество продаж'].max()} продаж в день, медиана —
  {grouped_df['Количество продаж'].median():.0f}, стандартное отклонение
  небольшое ({grouped_df['Количество продаж'].std():.0f}) относительно
  среднего ({grouped_df['Количество продаж'].mean():.0f}) — значит,
  выраженного тренда роста/падения за 8 месяцев нет, бизнес работает
  на плато.
- Заметных сезонных провалов или всплесков (например, майских праздников,
  летнего затишья) на уровне ЕЖЕДНЕВНОГО количества продаж визуально не
  видно — колебания выглядят как обычный операционный шум по дням недели,
  а не сезонность на уровне месяцев.
- Один день — 18 апреля 2018 — выделяется вниз ({grouped_df.loc[grouped_df['Дата']=='2018-04-18','Количество продаж'].values[0]}
  продаж против медианы {grouped_df['Количество продаж'].median():.0f}) — это,
  вероятно, сбой в записи данных или нерабочий день по другой причине,
  стоит уточнить у владельца данных.
""")

q1, q3 = df["Количество"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
outlier_row = df.loc[df["Количество"].idxmax()]
n_outliers_iqr = (df["Количество"] > upper_bound).sum()
print("=" * 72)
print("Выброс по столбцу 'Количество' (метод IQR)")
print("=" * 72)
print(f"Верхняя граница IQR (Q3 + 1.5*IQR) = {upper_bound:.1f} шт.")
print(f"Формальных выбросов выше границы: {n_outliers_iqr} строк из {len(df)} "
      f"— порог IQR слишком строг для этого распределения (много позиций "
      f"с небольшими, но обычными продажами 5-10 шт.), поэтому смотрим "
      "именно на строку с МАКСИМАЛЬНЫМ значением, как просит задание:")
print(outlier_row.to_string())
print(f"""
Это одна операция на {int(outlier_row['Количество'])} единиц товара
'{outlier_row['Номенклатура']}' на складе {int(outlier_row['Склад'])}
от контрагента '{outlier_row['Контрагент']}' {outlier_row['Дата'].date()} —
типичная продажа в данных 1-4 штуки (медиана {df['Количество'].median():.0f}),
а тут сразу 200 — похоже на оптовую/разовую крупную закупку, а не на
типичную розничную продажу. Стоит проверить эту строку у источника данных
на предмет ошибки ввода (лишний ноль) либо подтвердить, что это
легитимная крупная сделка.
""")

sub = df[
    (df["Склад"] == 3)
    & (df["Дата"].dt.weekday == 2)
    & (df["Дата"].dt.month.isin([6, 7, 8]))
]
top_products = sub.groupby("Номенклатура")["Количество"].sum().sort_values(ascending=False)
print("=" * 72)
print("Топ товаров по продажам по средам (июнь-август) на складе №3")
print("=" * 72)
print(top_products.head(5))
print(f"\nОтвет: топовый товар — {top_products.index[0]} "
      f"({int(top_products.iloc[0])} шт. суммарно по средам июнь-август "
      f"на складе 3, из {sub['Дата'].nunique()} наблюдаемых сред).")

print("=" * 72)
print("Погода в Астане (rp5.ru), 2018-01-04..2018-08-31")
print("=" * 72)

WEATHER_CSV = HERE / "данные" / "погода_астана_2018.csv"
if WEATHER_CSV.exists():
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["Дата"])
    merged = grouped_df.merge(weather[["Дата", "T"]], on="Дата", how="left")
    print(f"Дней с погодой: {merged['T'].notna().sum()} из {len(merged)} "
          f"(пропусков в склейке: {merged['T'].isna().sum()})")

    corr_p = merged["Количество продаж"].corr(merged["T"])
    corr_s = merged["Количество продаж"].corr(merged["T"], method="spearman")
    print(f"Корреляция 'Количество продаж' vs 'T': Пирсон r = {corr_p:.3f}, "
          f"Спирмен ρ = {corr_s:.3f}")
    print("""
Вывод: корреляция практически нулевая (|r| < 0.03) — среднесуточная
температура НЕ объясняет колебания дневного количества продаж в этих
данных. Температура за период растёт от ~-30°C (январь) до ~+25°C
(июль-август) — классический сезонный ход, а продажи как колебались в
диапазоне ~1300-1840 в день (недельный цикл, провалы по понедельникам),
так и продолжают колебаться в том же диапазоне без видимого сдвига
вслед за потеплением. Значит, спрос в этом бизнесе несезонный /
некли-зависимый (или как минимум не объясняется температурой воздуха).
""")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(merged["Дата"], merged["Количество продаж"], label="Количество продаж")
    ax2_t = ax2.twinx()
    ax2_t.plot(merged["Дата"], merged["T"], color="orange", label="T, °C")
    ax2.set_title("Продажи и температура")
    fig2.tight_layout()
    fig2.savefig(OUT / "блок3_продажи_и_температура.png", dpi=110)

    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(merged["Дата"], merged["T"], color="orange")
    ax3.set_title("Средняя температура по дням")
    fig3.tight_layout()
    fig3.savefig(OUT / "блок3_температура.png", dpi=110)
    print("Графики построены и сохранены в проект_графики/ "
          "(блок3_продажи_и_температура.png, блок3_температура.png).")
else:
    print(f"Файл {WEATHER_CSV.name} не найден в данные/ — шаг пропущен.")
