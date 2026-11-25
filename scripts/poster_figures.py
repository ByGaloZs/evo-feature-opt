"""Figuras del póster A0 (PDF vectorial, a tamaño final y con la paleta común).

No forma parte del sistema: solo lee resultados ya generados en ``results/`` y no toca
validación ni ningún parámetro. Produce, en ``poster/figuras/``:

- ``resultado_cajas.pdf``: e′ por configuración en California y Diabetes, con la línea de e (panel 1).
- ``diabetes_particiones.pdf``: cambio del MAE de C por partición, D23 (panel 4).
- ``fiabilidad_mutacion.pdf``: % de ejecuciones con MAE en CV < 0,5, exploratorio (panel 5).
- ``originales_conservados.pdf``: fracción de ejecuciones de C que conserva cada original (panel 2).
- ``transformaciones.pdf``: forma de las dos transformaciones más frecuentes (panel 3).

Uso::

    python scripts/poster_figures.py --results results --out poster/figuras
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Paleta común (la misma que en poster.tex)
COLORES = {
    "baseline": "#4D4D4D",
    "A": "#CC79A7",
    "B": "#56B4E9",
    "C_tree": "#D55E00",
    "C_gep": "#009E73",
}
CARBON = "#2F3542"
GRIS = "#8A8F99"
ETIQUETAS = {"A": "A", "B": "B", "C_tree": "C", "C_gep": "C-GEP"}
# Límite superior del eje en California: por encima, los valores atípicos se dibujan
# como triángulos en el borde con su valor, para no aplastar la escala.
TOPE = {"california": 0.565, "diabetes": None}
MM = 1 / 25.4  # milímetros a pulgadas

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 27,
    "axes.titlesize": 33,
    "axes.titleweight": "bold",
    "axes.labelsize": 28,
    "xtick.labelsize": 27,
    "ytick.labelsize": 27,
    "axes.edgecolor": CARBON,
    "axes.labelcolor": CARBON,
    "xtick.color": CARBON,
    "ytick.color": CARBON,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,
    "axes.formatter.use_locale": False,
})


def coma(x: float, dec: int = 1) -> str:
    """Número con coma decimal (estilo español)."""
    return f"{x:.{dec}f}".replace(".", ",")


def cargar_e_prime(results: Path, config: str, dataset: str) -> tuple[np.ndarray, float]:
    """Devuelve los e′ de las 30 semillas y el e (línea base) de una configuración."""
    archivos = sorted((results / "runs" / f"final_{config}" / dataset).glob("seed_*.json"))
    datos = [json.loads(a.read_text()) for a in archivos]
    e_prime = np.array([d["mae"] for d in datos])
    e = datos[0]["baseline_mae"]
    return e_prime, e


def figura_cajas(results: Path, out: Path) -> None:
    """Panel 1: e′ por configuración, con la línea de e, en los dos datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(300 * MM, 205 * MM))
    rng = np.random.default_rng(0)  # solo para separar los puntos en horizontal
    configs = ["A", "B", "C_tree", "C_gep"]
    for ax, (dataset, titulo, dec) in zip(axes, [("california", "California", 2), ("diabetes", "Diabetes", 0)]):
        valores, e = [], None
        for c in configs:
            v, e = cargar_e_prime(results, c, dataset)
            valores.append(v)
        tope = TOPE[dataset]
        pos = np.arange(1, len(configs) + 1)
        cajas = ax.boxplot(valores, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                           whis=(0, 100) if tope is None else 1.5,
                           medianprops={"color": CARBON, "linewidth": 3},
                           whiskerprops={"color": CARBON, "linewidth": 1.5},
                           capprops={"color": CARBON, "linewidth": 1.5},
                           boxprops={"linewidth": 1.5, "edgecolor": CARBON})
        for caja, c in zip(cajas["boxes"], configs):
            caja.set_facecolor(COLORES[c])
            caja.set_alpha(0.45)
        for p, v, c in zip(pos, valores, configs):
            x = p + rng.uniform(-0.18, 0.18, size=len(v))
            dentro = v <= tope if tope is not None else np.ones_like(v, dtype=bool)
            ax.scatter(x[dentro], v[dentro], s=60, color=COLORES[c], edgecolor="white", linewidth=0.6, zorder=3)
            fuera = np.sort(v[~dentro])[::-1]
            if len(fuera):
                ax.scatter(p, tope - 0.0022, marker="^", s=190, color=COLORES[c], edgecolor=CARBON, zorder=4)
                ax.text(p + 0.2, tope - 0.0022, "\n".join(coma(f, 2) for f in fuera), fontsize=23,
                        va="top", ha="left", color=CARBON, linespacing=1.0)
        ax.axhline(e, color=COLORES["baseline"], linestyle="--", linewidth=2.5, zorder=1)
        ax.set_xticks(pos, [ETIQUETAS[c] for c in configs])
        ax.set_xlim(0.45, len(configs) + 0.55)
        if tope is not None:
            ax.set_ylim(0.455, tope)
        ax.set_title(titulo, loc="left", color=CARBON)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _, d=dec: coma(y, d)))
        ax.grid(axis="y", color="#E3E4E8", linewidth=1)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE en validación (e′)")
    linea = matplotlib.lines.Line2D([], [], color=COLORES["baseline"], linestyle="--", linewidth=2.5)
    fig.legend([linea], ["línea base (e): regresión lineal con todos los originales"], loc="upper right",
               frameon=False, fontsize=24, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.94))
    fig.savefig(out / "resultado_cajas.pdf")
    plt.close(fig)


def figura_particiones(results: Path, out: Path) -> None:
    """Panel 4: cambio del MAE de C (mediana de las semillas) en cada partición de Diabetes."""
    filas = json.loads((results / "d23" / "resumen.json").read_text())["rows"]
    nombres, cambios = [], []
    for k, v in filas.items():
        nombres.append("2026\n(original)" if k.startswith("2026") else k)
        cambios.append(-100 * v["rel_improvement_median"])  # negativo = menos error
    fig, ax = plt.subplots(figsize=(236 * MM, 112 * MM))
    x = np.arange(len(nombres))
    for xi, c in zip(x, cambios):
        color = COLORES["C_tree"] if c < 0 else GRIS
        ax.vlines(xi, 0, c, color=color, linewidth=5)
        ax.scatter(xi, c, s=380, color=color, zorder=3)
        ax.text(xi, c + (0.45 if c > 0 else -0.45), f"{'+' if c > 0 else '–'}{coma(abs(c))} %",
                ha="center", va="bottom" if c > 0 else "top", fontsize=23, color=CARBON)
    ax.axhline(0, color=COLORES["baseline"], linewidth=2.5, linestyle="--")
    ax.set_xticks(x, nombres)
    ax.set_xlim(-0.5, len(nombres) - 0.5)
    ax.set_ylabel("Cambio MAE (%)")
    ax.set_ylim(-5.2, 4.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{int(y):+d}".replace("+0", "0")))
    ax.text(-0.45, 3.9, "abajo = menos error", ha="left", va="top", fontsize=23, color=COLORES["C_tree"])
    ax.grid(axis="y", color="#E3E4E8", linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "diabetes_particiones.pdf")
    plt.close(fig)


def figura_mutacion(results: Path, out: Path) -> None:
    """Panel 5: % de ejecuciones con MAE en CV < 0,5 según la representación y la mutación."""
    brazos = json.loads((results / "explore_mutation" / "resumen.json").read_text())["arms"]
    filas = [
        ("Árboles, mut. 0,2*", brazos["pm0.2 (control)"], COLORES["C_tree"], 1.0),
        ("Árboles, mut. 0,5", brazos["pm0.5"], COLORES["C_tree"], 0.55),
        ("Árboles, mut. 1,0", brazos["pm1"], COLORES["C_tree"], 0.55),
        ("GEP", brazos["GEP (referencia)"], COLORES["C_gep"], 1.0),
    ]
    fig, ax = plt.subplots(figsize=(236 * MM, 100 * MM))
    y = np.arange(len(filas))[::-1]
    for yi, (nombre, b, color, alfa) in zip(y, filas):
        ax.barh(yi, b["pct_cv_below_threshold"], color=color, alpha=alfa, height=0.62)
        ax.text(b["pct_cv_below_threshold"] + 1.5, yi,
                f"{b['n_cv_below_threshold']}/{b['n']}", va="center", fontsize=25, color=CARBON)
    ax.set_yticks(y, [f[0] for f in filas])
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% de ejecuciones con CV < 0,5")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "fiabilidad_mutacion.pdf")
    plt.close(fig)


ORIGINALES_CALIFORNIA = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
                         "Latitude", "Longitude"]


def cargar_runs(results: Path, config: str, dataset: str) -> list[dict]:
    """Carga los JSON de las 30 ejecuciones finales de una configuración."""
    archivos = sorted((results / "runs" / f"final_{config}" / dataset).glob("seed_*.json"))
    return [json.loads(a.read_text()) for a in archivos]


def figura_originales(results: Path, out: Path) -> None:
    """Panel 2: fracción de ejecuciones de C (árboles) que conserva cada original en California."""
    runs = cargar_runs(results, "C_tree", "california")
    frac = {n: sum(n in r["kept_originals"] for r in runs) / len(runs) for n in ORIGINALES_CALIFORNIA}
    orden = sorted(frac, key=frac.get)
    fig, ax = plt.subplots(figsize=(172 * MM, 118 * MM))
    y = np.arange(len(orden))
    valores = [100 * frac[n] for n in orden]
    colores = [CARBON if v >= 50 else "#C9CCD3" for v in valores]
    ax.barh(y, valores, color=colores, height=0.68)
    for yi, v in zip(y, valores):
        ax.text(v + 2, yi, f"{v:.0f} %", va="center", fontsize=22, color=CARBON)
    ax.set_yticks(y, orden)
    ax.set_xlim(0, 125)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=24)
    ax.set_title("Originales que conserva C", loc="left", fontsize=26, color=CARBON)
    fig.tight_layout()
    fig.savefig(out / "originales_conservados.pdf")
    plt.close(fig)


def figura_transformaciones(results: Path, out: Path) -> None:
    """Panel 3: forma de log(1+|z|) y de z/√|z|, con la frecuencia de su variable en las fórmulas."""
    runs = cargar_runs(results, "C_tree", "california")
    usa = {n: sum(any(n in f for f in r["formulas"]) for r in runs) / len(runs) for n in ("AveRooms", "AveOccup")}
    z = np.linspace(-3, 3, 400)
    paneles = [
        (np.log1p(np.abs(z)), "log(1 + |AveRooms|)", None),
        (np.sign(z) * np.sqrt(np.abs(z)), "AveOccup / √|AveOccup|", None),
    ]
    print(f"AveRooms en las fórmulas del {100 * usa['AveRooms']:.0f} % de las ejecuciones; "
          f"AveOccup en el {100 * usa['AveOccup']:.0f} %")
    fig, axes = plt.subplots(1, 2, figsize=(236 * MM, 76 * MM))
    for ax, (f, titulo, nota) in zip(axes, paneles):
        ax.plot(z, z, color="#C9CCD3", linewidth=2.5, linestyle="--")
        ax.plot(z, f, color=COLORES["C_tree"], linewidth=4.5)
        ax.axhline(0, color="#E3E4E8", linewidth=1.2, zorder=0)
        ax.axvline(0, color="#E3E4E8", linewidth=1.2, zorder=0)
        ax.set_title(titulo, loc="left", fontsize=22, color=CARBON)
        ax.text(-2.9, 2.9, "- - sin transformar", fontsize=17, color="#9A9EA6", va="top")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-2, 0, 2])
        ax.set_ylim(-3.1, 3.1)
        ax.tick_params(labelsize=20)
        ax.set_xlabel("valor estandarizado (z)", fontsize=20)
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "transformaciones.pdf")
    plt.close(fig)


def main() -> None:
    """Genera todas las figuras del póster."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("poster/figuras"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figura_cajas(args.results, args.out)
    figura_particiones(args.results, args.out)
    figura_mutacion(args.results, args.out)
    figura_originales(args.results, args.out)
    figura_transformaciones(args.results, args.out)
    print(f"Figuras escritas en {args.out}")


if __name__ == "__main__":
    main()
