# Outline prezentacji - Canva / Google Slides
## Adaptacyjne wykrywanie fraudów oparte na grafach

> Format: kopiuj każdy slajd do narzędzia graficznego.
> Każdy blok zawiera: **TYTUŁ**, punkty do wklejenia, sugestię wizualną i notatki prezentera.

---

## SLAJD 1 - Tytuł

**TYTUŁ SLAJDU:**
Adaptacyjne wykrywanie fraudów oparte na grafach

**PODTYTUŁ:**
dla transakcji bankowych i e-commerce

**TREŚĆ:**
Franciszek Job · Jakub Ciszewski
AGH · Semestr 8 · 2026-05-17

**TAGLINE (małym tekstem pod spodem):**
*Wykrywamy ewoluujące wzorce fraudu, których klasyczne detektory nie widzą.*

> 🎨 **Sugestia wizualna:** Ciemne granatowe tło. Na środku lub w tle: animowana sieć grafowa (połączone węzły) - np. z unsplash.com wyszukaj "network graph dark". Logo AGH w rogu.

> 🎤 **Notatki prezentera:** Przywitaj się, podaj imiona, powiedz że zaraz opowiesz o tym co robicie i dlaczego. (~20 sek)

---

## SLAJD 2 - Problem i rozwiązanie

**TYTUŁ SLAJDU:**
Problem i rozwiązanie

**TREŚĆ - lewa kolumna (Problem):**
- Banki i e-commerce tracą **miliardy dolarów rocznie** na fraudach
- Fraudsterzy ciągle zmieniają taktyki → modele **degradują się w ciszy**
- Klasyczne systemy **nie widzą relacji**: siatki słupów, klonowanie kart, wspólne urządzenia

**TREŚĆ - prawa kolumna (Nasze rozwiązanie):**
- Budujemy **graf transakcji** - węzły połączone wspólną kartą, urządzeniem, IP, sklepem
- System **uczy się ciągle** - adaptuje do nowych schematów bez zapominania starych
- Dla kogo: banki, payment processory, platformy e-commerce

> 🎨 **Sugestia wizualna:** Podział 50/50 - lewa strona: ikona karty kredytowej z czerwonym X lub wykres spadający. Prawa strona: schemat małego grafu (3-4 połączone kółka). Kolor akcentu: czerwony (problem) vs zielony (rozwiązanie).

> 🎤 **Notatki prezentera:** Opisz problem - klasyczne modele są ślepe na relacje między transakcjami. Powiedz że to właśnie rozwiązujecie: graf + uczenie ciągłe. (~40 sek)

---

## SLAJD 3 - Mechanizmy AI

**TYTUŁ SLAJDU:**
Mechanizmy sztucznej inteligencji

**TREŚĆ - Stosowane metody:**
- 🔵 **Grafowe sieci neuronowe (GNN)** z biblioteki PyGOD - oceniają transakcję w kontekście sąsiedztwa w grafie
- 🔵 **Klasyczne metody jednoklasowe** (Isolation Forest, LOF, OCSVM) - punkt porównawczy
- 🔵 **Strategie continual learning** z pyCLAD - replay, regularyzacja, naive sequential

**TREŚĆ - Nieużywane i dlaczego:**
- ❌ **Systemy regułowe** - schematy fraudu zmieniają się zbyt szybko; ręczne reguły stają się przestarzałe w ciągu tygodni
- ❌ **Uczenie ze wzmocnieniem (RL)** - wykrywanie fraudu to ocena pojedynczej transakcji, nie sekwencyjna decyzja; nie ma polityki do optymalizacji

> 🎨 **Sugestia wizualna:** Trzy ikony w rzędzie dla stosowanych metod (np. sieć neuronowa / drzewo / zegar). Pod spodem dwa rzędy z czerwonym X dla nieużywanych. Czyste tło, dużo powietrza.

> 🎤 **Notatki prezentera:** Powiedz które metody stosujesz i po co. Sekcja "nieużywane" jest wymagana przez prowadzącego - wyjaśnij krótko dlaczego RL i reguły nie mają tu sensu. (~50 sek)

---

## SLAJD 4 - Uczenie nadzorowane

**TYTUŁ SLAJDU:**
Dane i uczenie nadzorowane

**TREŚĆ - Kategorie i atrybuty:**
- **Etykiety (kategorie):** fraud / legitimate - 2 klasy
- **Cechy węzłów:** kwota, czas, waluta, urządzenie, IP, karta, sklep, adres, geo
- **Krawędzie grafu:** dwie transakcje dzielące kartę / urządzenie / IP / sklep → krawędź
- **Preprocessing:** gotowe pipeline'y z Kaggle (wysoko oceniane notebooki)

**TREŚĆ - 3 reżimy nadzoru (tabela):**

| Scenariusz bankowy | Dostępne etykiety | Reżim |
|---|---|---|
| Tylko potwierdzona historia "czysta" | Brak fraudów | Semi-supervised |
| Backlog zespołu śledczego | 1–20% fraudów | Few-shot |
| Pełna historia z etykietami | Wszystkie | Fully supervised |

**DATASET:** IEEE-CIS Fraud Detection (Kaggle) - ~590 000 transakcji

> 🎨 **Sugestia wizualna:** Tabela 3-rzędowa z kolorowanymi wierszami (szary / żółty / zielony) + mała ikona Kaggle lub bazy danych w rogu.

> 🎤 **Notatki prezentera:** Powiedz że masz dwie klasy: fraud i legit. Wyjaśnij atrybuty i jak budujesz graf (krawędzie ze wspólnych encji). Potem 3 scenariusze - odpowiadają rzeczywistości bankowej. (~60 sek)

---

## SLAJD 5 - Architektura systemu

**TYTUŁ SLAJDU:**
Architektura systemu

**TREŚĆ - Opis diagramu (wkleić jako bloki flowchart w Canva):**

```
[Strumień transakcji bankowych / e-commerce]
                    ↓
      [Preprocessing + budowa grafu]
         (wspólne karty / urządzenia / IP)
                    ↓
    ┌───────────────────────────────────┐
    │      Adapter PyGOD ↔ pyCLAD      │  ← kluczowy wkład techniczny
    │   fit / predict / score          │
    └──────────┬────────────┬──────────┘
               ↓            ↓
        [Modele GNN]   [Baseline PyOD]
        z PyGOD        Isolation Forest
               ↓            ↓
    [Oflagowane transakcje → kolejka śledcza]
               ↓
    [Ewaluacja: ROC-AUC · BWT · FWT · Forgetting]
```

**OPIS POD DIAGRAMEM:**
Adapter PyGOD ↔ pyCLAD to nasz kluczowy wkład - jedno API reużywane w eksperymentach statycznych i continual.

> 🎨 **Sugestia wizualna:** Narysuj flowchart w Canva używając bloków. Adapter zaznacz pomarańczowym kolorem. Strumień transakcji i wyniki - niebieski. Ewaluacja - zielony. Strzałki łączące bloki.

> 🎤 **Notatki prezentera:** Przejdź przez diagram od góry do dołu. Zatrzymaj się na adapterze - to Wasza główna praca techniczna. (~40 sek)

---

## SLAJD 6 - Continual learning i metryki

**TYTUŁ SLAJDU:**
Jak detektor nadąża za ewoluującym fraudem

**TREŚĆ - Dwa scenariusze driftu:**
- 📅 **Podział czasowy** - trening na przeszłych miesiącach, test na przyszłych. Naśladuje realne wdrożenie: wczorajszy model wobec dzisiejszych transakcji.
- 🔵 **Podział klastrowy** - KMeans / HDBSCAN grupuje transakcje w "schematy fraudu". Każdy klaster = nowe zadanie w sekwencji.

**TREŚĆ - Metryki:**
- **ROC-AUC** - jakość detekcji (odporna na niezbalansowane klasy)
- **Forgetting** - ile model stracił na starych zadaniach?
- **BWT (Backward Transfer)** - czy nowe zadania szkodzą czy pomagają starym?
- **FWT (Forward Transfer)** - czy doświadczenie z przeszłości przyspiesza naukę nowych?

> 🎨 **Sugestia wizualna:** Lewa strona - oś czasu z podziałem na miesiące (styczeń → luty → marzec → …). Prawa strona - 4 metryki jako ikony / bullet points z krótkimi definicjami.

> 🎤 **Notatki prezentera:** Powiedz że detektor wytrenowany w styczniu degraduje się do czerwca - i że to modelujecie wprost. Wyjaśnij dwa scenariusze i co mierzą metryki. (~50 sek)

---

## SLAJD 7 - Biblioteki

**TYTUŁ SLAJDU:**
Biblioteki i tech stack

**TREŚĆ - Tabela:**

| Warstwa | Biblioteka |
|---|---|
| Modele GNN (anomalie w grafach) | **PyGOD** |
| Backend grafowy | **PyTorch Geometric** + PyTorch |
| Continual learning | **pyCLAD** |
| Klasyczne baseline | **PyOD** + scikit-learn |
| Preprocessing danych | pandas + numpy + Kaggle |
| Dataset | **IEEE-CIS** (Kaggle) |
| Eksperymenty | Weights & Biases |

> 🎨 **Sugestia wizualna:** Tabela z logotypami bibliotek w lewej kolumnie (dostępne na GitHub każdej biblioteki). Alternatywnie: hex-grid z nazwami bibliotek, każdy w swoim kolorze.

> 🎤 **Notatki prezentera:** Wymień szybko - nie musisz tłumaczyć każdej. Podkreśl PyGOD i pyCLAD jako kluczowe. (~20 sek)

---

## SLAJD 8 - Harmonogram i deliverables

**TYTUŁ SLAJDU:**
Plan pracy i deliverables

**TREŚĆ - Harmonogram:**
- **Faza 1** *(pierwsza połowa semestru)*
  - Detektor oceniony w 3 realistycznych scenariuszach bankowych
  - Szkielet adaptera PyGOD ↔ pyCLAD
- **Faza 2** *(druga połowa semestru)*
  - Scenariusze continual (czasowe + klastrowe)
  - Pełna integracja PyGOD ↔ pyCLAD
  - Badanie empiryczne: jak detektor radzi sobie z driftem?

**TREŚĆ - Finalne deliverables:**
- ✅ Działający detektor fraudów oparty na grafie
- ✅ Adapter PyGOD ↔ pyCLAD (wkład open-source)
- ✅ Scenariusze continual learning na danych bankowych
- ✅ Empiryczny dowód adaptacji do driftu

> 🎨 **Sugestia wizualna:** Oś czasu pozioma z dwoma blokami (Faza 1 / Faza 2). Pod spodem 4 checkboxy z deliverables. Akcent kolorystyczny na checkboxach (zielony).

> 🎤 **Notatki prezentera:** Powiedz krótko co będzie w każdej fazie. Zamknij listą deliverables. Podziękuj i zaproś do pytań. (~30 sek)

---

## Checklist przed Moodle

- [ ] Wszystkie 8 wymaganych punktów briefu pokryte (patrz mapowanie powyżej)
- [ ] Dry-run z timerem - cel: 4:30–5:00
- [ ] Eksport do PDF z Canva / Google Slides
- [ ] Upload na Moodle
