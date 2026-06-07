# Adaptacyjne wykrywanie fraudów oparte na grafach
## dla transakcji bankowych i e-commerce

**Franciszek Job · Jakub Ciszewski**
AGH · semestr 8

*Wykrywanie ewoluujących wzorców fraudu w strumieniach transakcji, których klasyczne detektory nie widzą.*

---

## Problem i rozwiązanie

**Problem**
- Banki i payment processory tracą **miliardy dolarów rocznie** na fraudach transakcyjnych.
- Fraudsterzy nieustannie zmieniają taktyki - statyczne modele ML **po cichu degradują się** w ciągu tygodni i miesięcy (*concept drift*).
- Klasyczne detektory tabelaryczne **nie widzą fraudów relacyjnych**: siatek słupów (mule rings), wspólnych urządzeń, klonowania kart, przejmowania kont.

**Nasze rozwiązanie - adaptacyjny system wykrywania fraudów, który:**
1. Wykorzystuje **graf transakcji** (karty ↔ urządzenia ↔ IP ↔ sklepy), żeby wychwytywać fraudy relacyjne.
2. **Uczy się w trybie ciągłym** w miarę jak wzorce fraudu się zmieniają - bez zapominania starych schematów.

**Dla kogo:** banki, sieci kartowe, payment processory, platformy e-commerce.

---

## Mechanizmy AI

**Stosowane w detektorze**
<!-- - **Detektory anomalii oparte na sieciach grafowych** (Graph Neural Networks z PyGOD: DOMINANT, AnomalyDAE, CoLA, GAAN, …) - oceniają każdą transakcję w kontekście jej sąsiedztwa (powiązane konta, urządzenia, sklepy). -->
- **Detektory anomalii oparte na sieciach grafowych** (Graph Neural Networks z PyGOD): oceniają każdą transakcję w kontekście jej sąsiedztwa (powiązane konta, urządzenia, sklepy).
- **Klasyczne baseline jednoklasowe** (np. Isolation Forest z PyOD) - kontrola sanity, że sygnał z grafu faktycznie daje wartość.
- **Strategie continual learning** z pyCLAD - utrzymują detektor sprawnym, gdy fraud ewoluuje.

**Nieużywane - i dlaczego**
- *Systemy regułowe / reprezentacja wiedzy* - schematy fraudu mutują zbyt szybko; ręcznie pisane reguły dezaktualizują się w ciągu miesięcy i nie wychwytują ukrytych wzorców relacyjnych.
- *Uczenie ze wzmocnieniem (RL)* - wykrywanie fraudu to ocenianie pojedynczej transakcji, a nie sekwencyjna decyzja-akcja; nie ma polityki do optymalizacji.

---

## Jak trenujemy - i realistyczne scenariusze banku

**Kategorie (etykiety):** binarne - `fraud` / `legitimate`.

**Atrybuty**
- *Cechy tabelaryczne transakcji:* kwota, czas, waluta, urządzenie, IP, sklep, karta, konto, geo, … (z wysoko ocenianych pipeline'ów preprocessingowych z Kaggle).
- *Krawędzie grafu:* wspólna karta / urządzenie / IP / konto / sklep - ta sama encja dotykająca wielu transakcji.

<!-- **Trzy reżimy nadzoru = trzy realistyczne scenariusze wdrożeniowe**

| Rzeczywistość banku | Dostępne etykiety | Nasz reżim |
|---|---|---|
| Tylko potwierdzona historia "czystych" (bez chargebacków) | Brak dla fraudów | *Semi-supervised / jednoklasowe* |
| Backlog zespołu śledczego | 1 % → 5 % → 10 % → 20 % | *Few-shot* |
| Pełna historia z etykietami (rzadkie, drogie) | Wszystkie | *W pełni nadzorowane* | -->

**Główny dataset - IEEE-CIS Fraud Detection** (Kaggle).
Dlaczego ten: ~590k transakcji; jawne kolumny encji (`card1–6`, `DeviceInfo`, `addr1/2`, `id_30–34`, domena email) tworzą naturalne krawędzie grafu; timestampy transakcji wspierają drift czasowy; duża pula publicznych pipeline'ów preprocessingowych z Kaggle do ponownego użycia.
*Stretch:* walidacja na drugim datasecie z FDB (np. Fraud Ecommerce) dla robustności - tylko jeśli Faza 1 + Faza 2 na IEEE-CIS skończą się na czas.

---

## Architektura systemu

```mermaid
flowchart TD
    S["Strumień transakcji"]
    A["Preprocessing i budowa grafu"]
    D["Adapter PyGOD ↔ pyCLAD"]
    E["Modele GNN z PyGOD"]
    F["Klasyczne baseline (PyOD)<br/>np. Isolation Forest"]
    G["Oflagowane transakcje<br/>→ kolejka śledcza"]
    H["Ewaluacja continual<br/>ROC-AUC · BWT · FWT · Forgetting"]

    S --> A
    A --> D
    D --> E
    D --> F
    E --> G
    F --> G
    E --> H
    F --> H

    classDef adapter fill:#ffe4b5,stroke:#cc7700,stroke-width:2px;
    classDef io fill:#e8f0fe,stroke:#1f3a5f;
    classDef eval fill:#e6f4ea,stroke:#1e8e3e;
    class D adapter;
    class S,G io;
    class H eval;
```

<!-- **Adapter PyGOD ↔ pyCLAD** to nasz kluczowy wkład techniczny - jedno API `fit / predict / score` re-używane w eksperymentach statycznych *i* continual. -->

---

## Jak utrzymujemy detektor sprawnym, gdy fraud ewoluuje

**Dlaczego to ważne.** Detektor wytrenowany na styczniowych fraudach po cichu degraduje się do czerwca. Modelujemy to wprost.


*Podział czasowy* - trening na minionych miesiącach, test na przyszłych. Naśladuje rzeczywiste wdrożenie: wczorajszy model wobec dzisiejszych transakcji.
<!-- - *Podział klastrowy (concept splits)* - KMeans / HDBSCAN w przestrzeni cech definiuje ukryte "schematy" fraudu; zadania przychodzą jedno po drugim. -->

**Metryki**
- *Jakość detekcji:* **ROC-AUC**.
- *Zachowanie continual:* **Backward Transfer (BWT)**, **Forward Transfer (FWT)**, **Forgetting** - czy detektor pamięta stare schematy, przenosi doświadczenie do przodu, czy katastroficznie zapomina?

---

## Uruchomienie

### 1. Setup (jednorazowo)

```bash
# Wymaga uv: https://docs.astral.sh/uv/getting-started/installation/
./setup.sh
```

Skrypt automatycznie:
- instaluje Python 3.12 i zależności w `.venv`
- tworzy katalogi `data/raw/` i `results/`
- pobiera dataset IEEE-CIS jeśli `~/.kaggle/kaggle.json` istnieje

Jeśli nie masz Kaggle CLI, pobierz dataset ręcznie z `kaggle.com/competitions/ieee-fraud-detection/data` i wrzuć pliki do `data/raw/`.

### 2. Trening i ewaluacja

```bash
uv run python experiments/run_experiment.py
```

Skrypt uruchamia pełny pipeline:
1. Ładuje dane i dzieli na okna czasowe
2. Trenuje **baseline** (GNN bez continual learning)
3. Trenuje **Replay** (GNN z buforem historycznych próbek)
4. Dla każdego okna treningowego ewaluuje oba modele na wszystkich oknach
5. Drukuje tabelę metryk i zapisuje wykresy do `results/auc_matrices.png`

### 3. Tryb uruchomienia (CL vs bez CL)

Domyślnie skrypt uruchamia oba warianty. Możesz wybrać tylko jeden:

```bash
# tylko baseline (bez continual learning)
uv run python experiments/run_experiment.py --mode baseline

# tylko Replay (z continual learning)
uv run python experiments/run_experiment.py --mode replay

# oba — porównanie (domyślne)
uv run python experiments/run_experiment.py --mode both
```

Tabela metryk i wykres `auc_matrices.png` generują się tylko gdy uruchomione są oba tryby.

### 4. Wybór detektora

```bash
# OCGNN — domyślny, pamięć O(n), działa przy dużych oknach
uv run python experiments/run_experiment.py --detector ocgnn

# DOMINANT — O(n²) pamięci, wymaga mniejszych okien (max ~5000 wierszy przy 12 GB RAM)
uv run python experiments/run_experiment.py --detector dominant --max-rows-per-window 5000

# AnomalyDAE
uv run python experiments/run_experiment.py --detector anomalydae --max-rows-per-window 5000

# CONAD
uv run python experiments/run_experiment.py --detector conad --max-rows-per-window 5000
```

Dostępne detektory:

| Detektor | Pamięć | Uwagi |
|---|---|---|
| `ocgnn` | O(n) | domyślny, działa przy 50k węzłów |
| `dominant` | O(n²) | rekonstruuje macierz sąsiedztwa, wymaga małych okien |
| `anomalydae` | O(n²) | jak DOMINANT |
| `conad` | O(n²) | jak DOMINANT |

### 5. Wszystkie parametry

```bash
uv run python experiments/run_experiment.py \
  --mode both \               # baseline / replay / both
  --detector ocgnn \          # ocgnn / dominant / anomalydae / conad
  --n-windows 6 \             # liczba okien czasowych
  --max-rows-per-window 50000 \  # rozmiar okna (mniej = szybciej, mniej RAM)
  --v-cols-max 200 \          # zakres cech V1..Vn (max 339)
  --epochs 50 \               # liczba epok treningu
  --hidden-channels 128 \     # rozmiar warstwy ukrytej GNN
  --buffer-size 5000 \        # rozmiar bufora replay
  --replay-ratio 0.3          # udział próbek replay w treningu
```

Najlepsze znane parametry (Replay AUC=0.6566, FWT=0.1611):

```bash
uv run python experiments/run_experiment.py \
  --v-cols-max 200 \
  --buffer-size 5000 \
  --replay-ratio 0.3 \
  --epochs 50 \
  --hidden-channels 128
```

---

## Biblioteki i tech stack

- PyGOD
- pyCLAD
- pyOD
- pandas


<!-- | Warstwa | Biblioteka |
|-------|---------|
| Modele anomalii grafowych | **PyGOD** |
| Backend grafowy | **PyTorch Geometric**, PyTorch |
| Protokół continual learning | **pyCLAD** |
| Klasyczne baseline | **PyOD**, scikit-learn |
| Dane / cechy | pandas, numpy, publiczne notebooki Kaggle |
| Dataset | **IEEE-CIS Fraud Detection** (główny) - Kaggle |
| Śledzenie eksperymentów | Weights & Biases *(do ustalenia)* | -->

<!-- --- -->

<!-- ## Produkt

Działający detektor fraudów oparty na grafie · Adapter PyGOD ↔ pyCLAD · Scenariusze continual na publicznych danych bankowych/e-commerce · Dowód, że adaptuje się do driftu. -->
