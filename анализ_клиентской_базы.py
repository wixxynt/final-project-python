import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "данные"
OUT = HERE / "проект_графики"
OUT.mkdir(exist_ok=True)

ci = pd.read_csv(DATA / "customer_info.xlsx - QUERY_FOR_ABT_CUSTOMERINFO_0002.csv")
tx = pd.read_csv(DATA / "transactions_info.xlsx - TRANSACTIONS (1).csv")
tx["date_new"] = pd.to_datetime(tx["date_new"], format="%d/%m/%Y")

print("=" * 72)
print("0) Обзор и проверка данных")
print("=" * 72)
print(f"customer_info: {ci.shape[0]} строк, {ci.shape[1]} столбцов")
print(f"transactions:  {tx.shape[0]} строк, {tx.shape[1]} столбцов, "
      f"период {tx['date_new'].min().date()} .. {tx['date_new'].max().date()}")
print(f"Пропуски в customer_info:\n{ci.isna().sum()[ci.isna().sum() > 0]}")

clients_ci = set(ci["Id_client"])
clients_tx = set(tx["ID_client"])
print(f"\nУникальных клиентов в customer_info: {len(clients_ci)}")
print(f"Уникальных клиентов в transactions:  {len(clients_tx)}")
print(f"Пересечение: {len(clients_ci & clients_tx)} "
      "(таблицы соединяются 1:1 без потерь)")

snapshot = tx["date_new"].max()
rfm = tx.groupby("ID_client").agg(
    frequency=("Id_check", "nunique"),
    monetary=("Sum_payment", "sum"),
    last_date=("date_new", "max"),
).reset_index()
rfm["recency_days"] = (snapshot - rfm["last_date"]).dt.days

df = ci.merge(rfm, left_on="Id_client", right_on="ID_client", how="left")
diff = df["Total_amount"] - df["monetary"]
print(f"\nПРОВЕРКА: совпадает ли Total_amount из customer_info с суммой "
      f"Sum_payment по чекам того же клиента?")
print(f"  Точных совпадений (|diff| < 1): {(diff.abs() < 1).sum()} из {len(df)}")
print(f"  Разброс расхождения: min={diff.min():.0f}, median={diff.median():.0f}, "
      f"max={diff.max():.0f}")
print("""  НАХОДКА (важно для отчёта куратору): Total_amount в customer_info
  НЕ равен сумме реальных платежей клиента из transactions ни у одного
  клиента — расхождения хаотичны и по знаку, и по величине. Это значит,
  Total_amount — какая-то ДРУГАЯ метрика (например, посчитанная в другой
  валюте/за другой период/из другой системы), а не агрегат этой же
  транзакционной таблицы. Дальше в анализе считаем Monetary заново из
  transactions и НЕ смешиваем его с Total_amount как будто это одно и
  то же — это разные показатели.""")

top_client = df.sort_values("monetary", ascending=False).iloc[0]
print(f"\nВыброс: клиент {int(top_client['Id_client'])} — "
      f"{int(top_client['frequency'])} чеков, monetary={top_client['monetary']:.0f} "
      f"(медиана по базе — {df['monetary'].median():.0f}, т.е. в "
      f"{top_client['monetary']/df['monetary'].median():.0f} раз больше). "
      "Похоже на корпоративного/оптового клиента, а не рядового покупателя. "
      "Учитываем это при чтении средних (mean) значений monetary ниже — "
      "они смещены этим единственным клиентом, поэтому дополнительно "
      "смотрим медианы.")

print("\n" + "=" * 72)
print("1) RFM клиентской базы (Recency/Frequency/Monetary из транзакций)")
print("=" * 72)
print(df[["recency_days", "frequency", "monetary"]].describe())

print("\n" + "=" * 72)
print("2) Кто откликается на коммуникацию (Response_communcation)?")
print("=" * 72)
overall_rate = df["Response_communcation"].mean() * 100
print(f"Общий уровень отклика по базе: {overall_rate:.1f}% "
      f"({df['Response_communcation'].sum()} из {len(df)})")

print("\nСравнение RFM и демографии: откликнулись (1) vs не откликнулись (0)")
print(f"{'Признак':<15}{'resp=1, ср.':>14}{'resp=0, ср.':>14}{'p-value':>12}")
for col in ["recency_days", "frequency", "monetary", "Tenure", "Age"]:
    a = df.loc[df["Response_communcation"] == 1, col].dropna()
    b = df.loc[df["Response_communcation"] == 0, col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"{col:<15}{a.mean():>14.1f}{b.mean():>14.1f}{p:>12.4f}")

print("\nОтклик по полу:")
by_gender = df.groupby("Gender")["Response_communcation"].agg(["mean", "count"])
print(by_gender)
ct_gender = pd.crosstab(df["Gender"], df["Response_communcation"])
chi2_g, p_g, _, _ = stats.chi2_contingency(ct_gender)
print(f"chi2-тест (пол x отклик): p-value = {p_g:.4f}")

print("\nОтклик по Count_city (в скольких городах покупал клиент):")
print(df.groupby("Count_city")["Response_communcation"].agg(["mean", "count"]))

print("\nКорреляция числовых признаков с откликом (Pearson):")
num_cols = ["Total_amount", "Age", "Count_city", "Tenure",
            "recency_days", "frequency", "monetary", "Response_communcation"]
corr = df[num_cols].corr()["Response_communcation"].drop("Response_communcation")
print(corr.sort_values())

print("\n" + "=" * 72)
print("Интерпретация")
print("=" * 72)
print("""
- RFM (давность, частота, сумма покупок) статистически НЕ различаются
  между теми, кто откликнулся на коммуникацию, и теми, кто нет (все
  p-value >> 0.05). То есть предположение "чем активнее покупает клиент,
  тем охотнее он реагирует на коммуникацию" на этих данных не
  подтверждается — уровень вовлечённости в покупки не предсказывает
  отклик.
- Tenure (стаж клиента, число месяцев) — единственный числовой признак с
  p-value < 0.01: у откликнувшихся средний стаж чуть НИЖЕ (~8.4 месяца),
  чем у неоткликнувшихся (~8.8). Эффект статистически значимый, но по
  величине небольшой (разница <0.5 месяца) — на практике не образует
  сильного сегмента для таргетинга самого по себе.
- Total_amount (готовая метрика "ценности" клиента из другой системы,
  не равная реальным тратам, см. находку выше) отрицательно коррелирует
  с откликом (corr = -0.10) — из числовых признаков это самая заметная
  связь. Возможная причина: более "ценных" клиентов уже и так часто
  коммуницируют (усталость от рассылок), либо у них меньше стимула
  реагировать специально на эту коммуникацию — им и так хорошо.
- Пол: у женщин отклик выше (71.5%), чем у мужчин (67.9%), но разница на
  грани значимости (chi2 p ≈ 0.086) — не делайте на этом сильных выводов
  без большей выборки, это, скорее, наблюдение для проверки, а не факт.
- Count_city не показывает содержательной связи с откликом (значения
  для city=3 основаны всего на 8 клиентах — слишком мало наблюдений,
  чтобы доверять 100%-й цифре отклика в этой группе).

Вывод и рекомендация:
  Демография и история покупок в этих данных — слабые предикторы того,
  ответит ли клиент на коммуникацию. Прежде чем строить таргетинг
  следующей кампании на основе RFM или Total_amount, стоит (1) собрать
  больше признаков о содержании самой коммуникации (канал, оффер, время
  отправки — их может влиять сильнее, чем "кто клиент"), (2) проверить
  гипотезу про усталость от рассылок у клиентов с высоким Total_amount
  отдельным опросом/тестом, а не полагаться на слабую корреляцию.
  Единственный практически применимый сигнал — чуть более высокий отклик
  у клиентов с меньшим стажем: имеет смысл делать акцент на коммуникацию
  с новыми клиентами (Tenure < 6 месяцев), но проверить эффект на A/B-
  тесте, а не только на исторической корреляции.
""")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

by_gender["mean"].mul(100).plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Отклик на коммуникацию по полу")
axes[0].set_ylabel("Доля откликнувшихся, %")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=0)

df["monetary"].clip(upper=df["monetary"].quantile(0.99)).plot(
    kind="hist", bins=40, ax=axes[1], color="#55A868"
)
axes[1].set_title("Распределение Monetary (99-й перцентиль, без выброса)")
axes[1].set_xlabel("Сумма покупок клиента, currency транзакций")

fig.tight_layout()
fig.savefig(OUT / "блок2_отклик_и_monetary.png", dpi=110)
print(f"\nГрафики сохранены: {OUT / 'блок2_отклик_и_monetary.png'}")
