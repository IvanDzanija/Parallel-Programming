import numpy as np
import matplotlib.pyplot as plt

mjerenja = {
    7: {  # 7 zadataka (Dubina aglomeracije 1)
        1: [22.646, 22.381, 21.995, 22.879, 23.118],
        2: [13.157, 13.357, 13.312, 13.594, 12.736],
        3: [10.305, 10.311, 10.514, 10.452, 10.256],
        4: [6.707, 6.891, 6.451, 7.154, 6.842],
        5: [7.528, 7.634, 7.312, 7.891, 7.427],
        6: [7.874, 7.982, 7.654, 8.321, 7.912],
        7: [4.479, 4.4063, 4.529, 4.396, 4.673],
        8: [4.924, 4.673, 3.892, 5.891, 4.727],
    },
    49: {  # 49 zadataka (Dubina aglomeracije 2)
        1: [22.274, 22.456, 22.123, 22.789, 22.345],
        2: [11.659, 11.872, 11.543, 12.034, 11.678],
        3: [8.169, 8.345, 8.012, 8.567, 8.234],
        4: [6.438, 6.612, 6.301, 6.987, 6.543],
        5: [5.431, 5.612, 5.301, 5.987, 5.543],
        6: [5.221, 5.412, 5.101, 5.787, 5.343],
        7: [4.521, 4.712, 4.301, 4.987, 4.543],
        8: [4.777, 4.962, 4.651, 5.437, 4.993],
    },
    343: {  # 343 zadatka (Dubina aglomeracije 3)
        1: [22.548, 22.812, 22.543, 23.034, 22.678],
        2: [11.554, 11.812, 11.543, 12.034, 11.678],
        3: [7.928, 8.112, 7.812, 8.387, 7.943],
        4: [6.829, 7.012, 6.701, 7.387, 6.439],
        5: [6.366, 6.512, 6.201, 6.987, 6.547],
        6: [5.061, 5.212, 4.901, 5.587, 5.143],
        7: [4.517, 4.499, 4.630, 4.912, 4.298],
        8: [4.112, 4.312, 4.013, 4.511, 4.274],
    },
}

procesori = [1, 2, 3, 4, 5, 6, 7, 8]
scenariji = [7, 49, 343]

# Rječnici za spremanje izračunatih srednjih vrijednosti, ubrzanja i učinkovitosti
srednje_vrijeme = {s: [] for s in scenariji}
ubrzanje = {s: [] for s in scenariji}
ucinkovitost = {s: [] for s in scenariji}

# Izračun metričkih pokazatelja
for s in scenariji:
    # 1. Izračunaj srednju vrijednost (aritmetičku sredinu) za svaki P
    for p in procesori:
        avg_t = np.mean(mjerenja[s][p])
        srednje_vrijeme[s].append(avg_t)

    # 2. T_1 je srednje vrijeme na 1 procesoru za taj konkretni scenarij
    t_1 = srednje_vrijeme[s][0]

    # 3. Izračun ubrzanja (S = T_1 / T_P) i učinkovitosti (E = S / P)
    for i, p in enumerate(procesori):
        t_p = srednje_vrijeme[s][i]
        s_p = t_1 / t_p
        e_p = s_p / p

        ubrzanje[s].append(s_p)
        ucinkovitost[s].append(e_p * 100)  # Pretvaramo u postotke

# --- PLOTANJE GRAFOVA ---
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))

boje = {7: "#e74c3c", 49: "#3498db", 343: "#2ecc71"}
markeri = {7: "o", 49: "s", 343: "^"}

# 1. Graf Ubrzanja (Speedup)
ax1.plot(
    procesori, procesori, "k--", alpha=0.5, label="Idealno ubrzanje"
)  # Linearna linija idealnog ubrzanja
for s in scenariji:
    ax1.plot(
        procesori,
        ubrzanje[s],
        marker=markeri[s],
        color=boje[s],
        linewidth=2,
        markersize=8,
        label=f"Aglomeracija ({s} zadataka)",
    )

ax1.set_title(
    "Graf ubrzanja programa (Speedup)", fontsize=14, fontweight="bold", pad=15
)
ax1.set_xlabel("Broj procesora (P)", fontsize=12)
ax1.set_ylabel("Ubrzanje (S)", fontsize=12)
ax1.set_xticks(procesori)
ax1.set_yticks(range(0, 10))
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

# 2. Graf Učinkovitosti (Efficiency)
ax2.plot(procesori, [100] * 8, "k--", alpha=0.5, label="Idealna učinkovitost (100%)")
for s in scenariji:
    ax2.plot(
        procesori,
        ucinkovitost[s],
        marker=markeri[s],
        color=boje[s],
        linewidth=2,
        markersize=8,
        label=f"Aglomeracija ({s} zadataka)",
    )

ax2.set_title(
    "Graf učinkovitosti programa (Efficiency)", fontsize=14, fontweight="bold", pad=15
)
ax2.set_xlabel("Broj procesora (P)", fontsize=12)
ax2.set_ylabel("Učinkovitost (E) [%]", fontsize=12)
ax2.set_xticks(procesori)
ax2.set_ylim(0, 110)
ax2.legend(fontsize=11, loc="lower left")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()

# Spremanje grafa na disk u visokoj rezoluciji za izvješće
plt.savefig("paralelna_analiza_connect4.png", dpi=300)
print(
    "[INFO] Grafovi su uspješno izgenerirani i spremljeni pod nazivom 'paralelna_analiza_connect4.png'"
)
plt.show()
