import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import os

# ============================================================
# FLIGHT DELAYS ANALYSIS PROJECT
# ============================================================
# Ky script analizon dataset-in e vonesave te fluturimeve.
# Per ta perdorur: vendosni path-in e CSV-it ne variablin CSV_PATH.
# ============================================================

CSV_PATH = "flight_delays_dataset.csv"   # <-- ndryshoni kete path

# ----------------------------------------------------------
# 1. LEXIMI I DATASET-IT
# ----------------------------------------------------------
def load_data(path):
    """Lexon CSV-in dhe tregon 10 rreshtat e pare."""
    print("=" * 60)
    print("HAPI 1: Leximi i dataset-it")
    print("=" * 60)

    # Provo te lexosh CSV; nese nuk gjendet, krijo te dhenat demo
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        print(f"Dataset u ngarkua: {df.shape[0]} rreshta, {df.shape[1]} kolona")
    else:
        print(f"KUJDES: '{path}' nuk u gjet. Perdoret dataset demo.")
        df = create_demo_dataset()

    print("\n--- 10 rreshtat e pare ---")
    print(df.head(10).to_string())
    return df


def create_demo_dataset():
    """Krijon dataset demo bazuar ne strukturen e dataset-it real."""
    np.random.seed(42)
    n = 5000
    carriers = ["Southwest Airlines", "Delta Air Lines", "American Airlines",
                "United Airlines", "JetBlue Airways", "Alaska Airlines",
                "Spirit Air Lines", "Frontier Airlines"]
    airports = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA",
                "LAS", "MCO", "CLT", "PHX", "BOS", "EWR", "MSP"]

    carrier_col = np.random.choice(carriers, n)
    airport_col = np.random.choice(airports, n)
    month_col   = np.random.randint(1, 13, n)
    cancelled   = np.random.choice([0, 1], n, p=[0.97, 0.03])

    arr_delay        = np.random.normal(10, 35, n)
    carrier_delay    = np.abs(np.random.normal(8, 20, n))
    weather_delay    = np.abs(np.random.normal(3, 12, n))
    nas_delay        = np.abs(np.random.normal(5, 15, n))
    security_delay   = np.abs(np.random.normal(0.5, 2, n))
    late_aircraft_delay = np.abs(np.random.normal(7, 18, n))
    distance         = np.random.choice([300, 500, 800, 1200, 1800, 2500], n)

    # Vonesat rriten pak me distancen me te madhe
    arr_delay += distance * 0.003

    df = pd.DataFrame({
        "month":              month_col,
        "carrier_name":       carrier_col,
        "airport":            airport_col,
        "arr_delay":          arr_delay,
        "carrier_delay":      carrier_delay,
        "weather_delay":      weather_delay,
        "nas_delay":          nas_delay,
        "security_delay":     security_delay,
        "late_aircraft_delay": late_aircraft_delay,
        "arr_cancelled":      cancelled,
        "distance":           distance,
    })
    return df


# ----------------------------------------------------------
# 2. PASTRIMI I TE DHENAVE
# ----------------------------------------------------------
def clean_data(df):
    """Heq fluturimet e anuluara dhe vlerat NaN."""
    print("\n" + "=" * 60)
    print("HAPI 2: Pastrimi i te dhenave")
    print("=" * 60)

    before = len(df)

    # Filtro kolonat qe ekzistojne
    delay_cols = ["arr_delay", "carrier_delay", "weather_delay",
                  "nas_delay", "security_delay", "late_aircraft_delay"]
    existing_delay_cols = [c for c in delay_cols if c in df.columns]

    # Mbaj vetem fluturimet jo te anuluara
    cancel_col = "arr_cancelled" if "arr_cancelled" in df.columns else None
    if cancel_col:
        df = df[df[cancel_col] == 0].copy()

    # Heq rreshtat me NaN ne kolonat kryesore
    df = df.dropna(subset=existing_delay_cols[:1] if existing_delay_cols else [])

    # Siguro llojet numerike
    for col in existing_delay_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "arr_delay" in df.columns:
        df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce").fillna(0)

    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df = df.dropna(subset=["month"])
        df["month"] = df["month"].astype(int)

    after = len(df)
    print(f"Para pastrimit : {before:,} rreshta")
    print(f"Pas pastrimit  : {after:,} rreshta")
    print(f"U hoqen        : {before - after:,} rreshta")
    return df


# ----------------------------------------------------------
# 3. BAR CHART - Shkaqet e Vonesave
# ----------------------------------------------------------
def plot_delay_causes(df):
    """Bar chart per totalin/mesataren e cdo lloji vonese."""
    print("\n" + "=" * 60)
    print("HAPI 3: Bar Chart - Shkaqet e Vonesave")
    print("=" * 60)

    causes = {
        "carrier_delay":      "Carrier\n(Kompania)",
        "weather_delay":      "Weather\n(Moti)",
        "nas_delay":          "NAS\n(Sistemi)",
        "security_delay":     "Security\n(Siguria)",
        "late_aircraft_delay": "Late Aircraft\n(Avioni i vonuar)",
    }

    existing = {k: v for k, v in causes.items() if k in df.columns}

    means = {label: df[col].mean() for col, label in existing.items()}
    totals = {label: df[col].sum() for col, label in existing.items()}

    labels  = list(means.keys())
    avg_vals = list(means.values())
    tot_vals = list(totals.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    # Mesataret
    bars1 = ax1.bar(labels, avg_vals, color=colors[:len(labels)], edgecolor="white", linewidth=1.2)
    ax1.set_title("Vonesa Mesatare sipas Shkakut (minuta)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Minuta")
    ax1.set_xlabel("Shkaku i Vonesës")
    for bar, val in zip(bars1, avg_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Totalet
    bars2 = ax2.bar(labels, tot_vals, color=colors[:len(labels)], edgecolor="white", linewidth=1.2)
    ax2.set_title("Vonesa Totale sipas Shkakut (minuta)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Minuta (totale)")
    ax2.set_xlabel("Shkaku i Vonesës")
    for bar, val in zip(bars2, tot_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tot_vals)*0.01,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Analiza e Shkaqeve të Vonesave", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("delay_causes_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafiku u ruajt: delay_causes_bar_chart.png")

    # Printo statistikat
    print("\nStatistikat e vonesave mesatare (minuta):")
    for label, val in means.items():
        print(f"  {label:<25}: {val:.2f} min")


# ----------------------------------------------------------
# 4. HISTOGRAM - ArrDelay
# ----------------------------------------------------------
def plot_arr_delay_histogram(df):
    """Histogram per ArrDelay."""
    print("\n" + "=" * 60)
    print("HAPI 4: Histogram per ArrDelay")
    print("=" * 60)

    if "arr_delay" not in df.columns:
        print("Kolona 'arr_delay' nuk u gjet.")
        return

    data = df["arr_delay"].dropna()
    data_clipped = data.clip(-60, 300)  # clip per vizualizim me te mire

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(data_clipped, bins=50, color="#2196F3",
                               edgecolor="white", linewidth=0.5, alpha=0.85)

    # Vizato vijat vertikale
    ax.axvline(data.mean(),  color="red",    linestyle="--", linewidth=2,
               label=f"Mesatare: {data.mean():.1f} min")
    ax.axvline(data.median(), color="orange", linestyle="--", linewidth=2,
               label=f"Mediana:  {data.median():.1f} min")
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Asnje vonese")

    ax.set_title("Shpërndarja e Vonesave të Ardhjes (ArrDelay)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Vonesa (minuta)")
    ax.set_ylabel("Numri i Fluturimeve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("arr_delay_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafiku u ruajt: arr_delay_histogram.png")
    print(f"\nStatistikat ArrDelay:")
    print(f"  Mesatare : {data.mean():.2f} min")
    print(f"  Mediana  : {data.median():.2f} min")
    print(f"  Min      : {data.min():.2f} min")
    print(f"  Max      : {data.max():.2f} min")


# ----------------------------------------------------------
# 5. SCATTER PLOT - Distance vs ArrDelay
# ----------------------------------------------------------
def plot_distance_vs_delay(df):
    """Scatter plot per Distance vs ArrDelay."""
    print("\n" + "=" * 60)
    print("HAPI 5: Scatter Plot - Distance vs ArrDelay")
    print("=" * 60)

    if "distance" not in df.columns or "arr_delay" not in df.columns:
        print("Kolonat 'distance' ose 'arr_delay' nuk u gjetën.")
        return

    sample = df[["distance", "arr_delay"]].dropna().sample(
        min(3000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["distance"], sample["arr_delay"],
               alpha=0.3, s=15, color="#2196F3", label="Fluturime")

    # Vija e tendences (linear fit)
    z = np.polyfit(sample["distance"], sample["arr_delay"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample["distance"].min(), sample["distance"].max(), 200)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Tendenca (slope={z[0]:.4f})")

    ax.set_title("Distance vs Vonesa e Ardhjes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Distanca (milje)")
    ax.set_ylabel("Vonesa e Ardhjes (minuta)")
    ax.legend()
    ax.set_ylim(-100, 300)
    plt.tight_layout()
    plt.savefig("distance_vs_delay_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafiku u ruajt: distance_vs_delay_scatter.png")

    corr = sample["distance"].corr(sample["arr_delay"])
    print(f"\nKorelacioni Distance - ArrDelay: {corr:.4f}")
    if abs(corr) < 0.1:
        print("  -> Lidhje shume e dobet.")
    elif abs(corr) < 0.3:
        print("  -> Lidhje e dobet.")
    else:
        print("  -> Lidhje e moderuar/e forte.")


# ----------------------------------------------------------
# 6. LINE PLOT - Vonesa Mesatare Mujore
# ----------------------------------------------------------
def plot_monthly_delay(df):
    """Line plot per vonesën mesatare mujore."""
    print("\n" + "=" * 60)
    print("HAPI 6: Line Plot - Vonesa Mesatare Mujore")
    print("=" * 60)

    if "month" not in df.columns or "arr_delay" not in df.columns:
        print("Kolonat 'month' ose 'arr_delay' nuk u gjetën.")
        return

    monthly = (df.groupby("month")["arr_delay"]
                 .mean()
                 .reset_index()
                 .rename(columns={"arr_delay": "avg_delay"}))

    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Maj", 6:"Qer",
                   7:"Kor", 8:"Gus", 9:"Sht", 10:"Tet", 11:"Nen", 12:"Dhj"}
    monthly["month_name"] = monthly["month"].map(month_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly["month"], monthly["avg_delay"], marker="o",
            color="#2196F3", linewidth=2.5, markersize=8, label="Vonesa Mesatare")
    ax.fill_between(monthly["month"], monthly["avg_delay"],
                    alpha=0.15, color="#2196F3")

    # Shtoj etiketa muajsh
    ax.set_xticks(monthly["month"])
    ax.set_xticklabels([month_names.get(m, str(m)) for m in monthly["month"]])

    ax.axhline(monthly["avg_delay"].mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mesatare vjetore: {monthly['avg_delay'].mean():.1f} min")

    ax.set_title("Vonesa Mesatare e Ardhjes sipas Muajit", fontsize=14, fontweight="bold")
    ax.set_xlabel("Muaji")
    ax.set_ylabel("Vonesa Mesatare (minuta)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("monthly_delay_line_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafiku u ruajt: monthly_delay_line_plot.png")

    print("\nVonesa mesatare per muaj:")
    for _, row in monthly.iterrows():
        print(f"  {month_names.get(int(row['month']), str(int(row['month'])))}: "
              f"{row['avg_delay']:.2f} min")


# ----------------------------------------------------------
# 7. ANALIZA E REZULTATEVE
# ----------------------------------------------------------
def analyze_results(df):
    """Gjen muajin me vonesën më të lartë, shkakun kryesor, lidhjen distancë-vonese."""
    print("\n" + "=" * 60)
    print("HAPI 7: Analiza e Rezultateve")
    print("=" * 60)

    # 7a. Muaji me vonesën më të lartë
    if "month" in df.columns and "arr_delay" in df.columns:
        monthly_avg = df.groupby("month")["arr_delay"].mean()
        worst_month_num = monthly_avg.idxmax()
        month_names = {1:"Janar", 2:"Shkurt", 3:"Mars", 4:"Prill", 5:"Maj",
                       6:"Qershor", 7:"Korrik", 8:"Gusht", 9:"Shtator",
                       10:"Tetor", 11:"Nentor", 12:"Dhjetor"}
        worst_month = month_names.get(worst_month_num, str(worst_month_num))
        print(f"\n[A] Muaji me vonesën më të lartë: {worst_month} "
              f"({monthly_avg[worst_month_num]:.2f} min mesatarisht)")

    # 7b. Shkaku më i zakonshëm i vonesës
    cause_cols = {
        "carrier_delay":       "Carrier (Kompania Ajrore)",
        "weather_delay":       "Weather (Moti)",
        "nas_delay":           "NAS (Sistemi Kombetar)",
        "security_delay":      "Security (Siguria)",
        "late_aircraft_delay": "Late Aircraft (Avioni i vonuar)",
    }
    existing_causes = {k: v for k, v in cause_cols.items() if k in df.columns}
    if existing_causes:
        cause_totals = {v: df[k].sum() for k, v in existing_causes.items()}
        main_cause = max(cause_totals, key=cause_totals.get)
        print(f"\n[B] Shkaku kryesor i vonesës: {main_cause}")
        print("    Totalet per secilin shkak (minuta):")
        for cause, total in sorted(cause_totals.items(), key=lambda x: -x[1]):
            print(f"      {cause:<35}: {total:>12,.0f} min")

    # 7c. Lidhja distancë - vonese
    if "distance" in df.columns and "arr_delay" in df.columns:
        corr = df["distance"].corr(df["arr_delay"])
        short  = df[df["distance"] < 500]["arr_delay"].mean()
        medium = df[(df["distance"] >= 500) & (df["distance"] < 1500)]["arr_delay"].mean()
        long   = df[df["distance"] >= 1500]["arr_delay"].mean()
        print(f"\n[C] Lidhja Distancë - Vonese:")
        print(f"    Korelacioni: {corr:.4f}")
        print(f"    Vonesa mesatare per distanca:")
        print(f"      < 500 milje  : {short:.2f} min")
        print(f"      500-1500 milje: {medium:.2f} min")
        print(f"      > 1500 milje : {long:.2f} min")
        if long > short:
            print("    KONKLUZION: Fluturimet me te gjata PRIREN te kene me shume vonesa.")
        else:
            print("    KONKLUZION: Distanca nuk ndikon shume ne vonese.")


# ----------------------------------------------------------
# 8. RUAJ NJE GRAFIK SI PNG
# ----------------------------------------------------------
def save_summary_chart(df):
    """Krijon dhe ruan nje grafik permbledhes si PNG."""
    print("\n" + "=" * 60)
    print("HAPI 8: Ruajtja e grafikut permbledhes")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Permbledhje e Analizës së Vonesave të Fluturimeve",
                 fontsize=16, fontweight="bold")

    cause_cols = ["carrier_delay", "weather_delay", "nas_delay",
                  "security_delay", "late_aircraft_delay"]
    existing_causes = [c for c in cause_cols if c in df.columns]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    # Plot 1: Bar chart shkaqet
    ax = axes[0, 0]
    if existing_causes:
        means = [df[c].mean() for c in existing_causes]
        labels = ["Carrier", "Weather", "NAS", "Security", "Late A/C"][:len(existing_causes)]
        ax.bar(labels, means, color=colors[:len(existing_causes)], edgecolor="white")
        ax.set_title("Vonesa Mesatare sipas Shkakut")
        ax.set_ylabel("Minuta")

    # Plot 2: Histogram ArrDelay
    ax = axes[0, 1]
    if "arr_delay" in df.columns:
        data = df["arr_delay"].clip(-60, 300)
        ax.hist(data, bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
        ax.axvline(df["arr_delay"].mean(), color="red", linestyle="--", linewidth=2)
        ax.set_title("Shpërndarje e ArrDelay")
        ax.set_xlabel("Minuta")
        ax.set_ylabel("Frekuenca")

    # Plot 3: Scatter
    ax = axes[1, 0]
    if "distance" in df.columns and "arr_delay" in df.columns:
        sample = df[["distance", "arr_delay"]].dropna().sample(
            min(2000, len(df)), random_state=42)
        ax.scatter(sample["distance"], sample["arr_delay"],
                   alpha=0.25, s=10, color="#2196F3")
        z = np.polyfit(sample["distance"], sample["arr_delay"], 1)
        x_l = np.linspace(sample["distance"].min(), sample["distance"].max(), 200)
        ax.plot(x_l, np.poly1d(z)(x_l), "r--", linewidth=2)
        ax.set_title("Distance vs ArrDelay")
        ax.set_xlabel("Distanca (milje)")
        ax.set_ylabel("Vonesa (min)")
        ax.set_ylim(-60, 250)

    # Plot 4: Line plot mujor
    ax = axes[1, 1]
    if "month" in df.columns and "arr_delay" in df.columns:
        monthly = df.groupby("month")["arr_delay"].mean()
        month_abr = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Maj",6:"Qer",
                     7:"Kor",8:"Gus",9:"Sht",10:"Tet",11:"Nen",12:"Dhj"}
        ax.plot(monthly.index, monthly.values, marker="o",
                color="#2196F3", linewidth=2.5, markersize=7)
        ax.set_xticks(monthly.index)
        ax.set_xticklabels([month_abr.get(m, str(m)) for m in monthly.index], fontsize=8)
        ax.set_title("Vonesa Mesatare Mujore")
        ax.set_xlabel("Muaji")
        ax.set_ylabel("Minuta")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file = "flight_delay_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grafiku permbledhes u ruajt: {output_file}")
    return output_file


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  ANALIZA E VONESAVE TE FLUTURIMEVE")
    print("=" * 60)

    df_raw    = load_data(CSV_PATH)
    df_clean  = clean_data(df_raw)

    plot_delay_causes(df_clean)
    plot_arr_delay_histogram(df_clean)
    plot_distance_vs_delay(df_clean)
    plot_monthly_delay(df_clean)
    analyze_results(df_clean)
    save_summary_chart(df_clean)

    print("\n" + "=" * 60)
    print("  ANALIZA PERFUNDOI ME SUKSES!")
    print("=" * 60)