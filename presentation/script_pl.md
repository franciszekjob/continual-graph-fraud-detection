# Skrypt prezentacji — tekst mówiony

Tekst do wygłoszenia slajd po slajdzie (`presentation/slides.html`). Czas docelowy ~8–10 min.
Można czytać niemal dosłownie albo traktować jako notatki. **Pogrubione** = warto zaakcentować głosem.

---

## Slajd 1 — Tytuł
Dzień dobry. Nasz projekt to **adaptacyjny system wykrywania fraudów w transakcjach bankowych**.
Robiliśmy go we dwóch — Franciszek Job i Jakub Ciszewski. Pokażemy najpierw o co chodzi i jak
to zbudowaliśmy, a potem przejdziemy do wyników.

## Slajd 2 — Wprowadzenie
Fraud w transakcjach to **powszechny i kosztowny problem** — dotyka banków, systemów płatności,
e-commerce. Co ważne, fraudsterzy działają **w sposób zorganizowany i ewoluujący**: na jeden fraud
składa się zwykle cała siatka powiązanych transakcji, a metody zmieniają się w czasie. Naszym
celem był system, który radzi sobie i z **relacyjną strukturą** danych, i ze **zmiennością wzorców**.

## Slajd 3 — Problemy obecnych rozwiązań
Dwa główne problemy klasycznych podejść. Po pierwsze, **modele się degradują** — wytrenowany model
jest użyteczny krótko, bo oszuści wymyślają nowe metody. Po drugie, **klasyczne detektory ignorują
relacje** między transakcjami — patrzą na pojedynczą transakcję, a fraud to często sieć transakcji.

## Slajd 4 — Nasze rozwiązanie
Stąd nasz pomysł: reprezentujemy dane jako **graf transakcji** — wierzchołek to transakcja, a
krawędź to **wspólna encja** między dwiema transakcjami, np. ta sama karta, adres czy domena e-mail.
Do tego system działa w trybie **continual learning** — doucza się nowych wzorców na bieżąco,
zamiast trenować raz i zamarzać.

## Slajd 5 — Problemy naszego rozwiązania
Takie podejście ma dwa wyzwania. **Rozmiar grafu** — dane transakcyjne są ogromne, więc naiwny graf
potrafi nie zmieścić się w RAM-ie. I **catastrophic forgetting** — ucząc się nowych wzorców, model
może zapominać te stare.

## Slajd 6 — Rozwiązania, które testujemy
Na te problemy mieliśmy plan: **mądry dobór encji** tworzących krawędzie, żeby ograniczyć rozmiar
grafu; **Replay** — dorzucanie starych przykładów przy uczeniu na nowych; oraz **EWC**, czyli karanie
modelu za zmianę wag istotnych dla wcześniejszych danych. Od razu zaznaczę — EWC nie zdążyliśmy
uruchomić, wrócę do tego na końcu.

## Slajd 7 — Architektura systemu
Pipeline jest prosty i modułowy: wczytanie danych, preprocessing z podziałem na okresy czasowe,
budowa grafu, model grafowy z **PyGOD**, opakowany naszym adapterem w framework continual learningu
**pyCLAD**, i na końcu ewaluacja. Dwa pomarańczowe klocki — PyGOD i pyCLAD — to serce systemu i nasz
własny adapter, który je spina.

## Slajd 8 — Metryki ewaluacji
Mierzymy dwie rzeczy. **Czy w ogóle wykrywamy fraudy** — tu używamy **ROC-AUC**. I **czy system
uczy się w czasie** — tu patrzymy na Forgetting, **Backward Transfer** (czy nowa nauka pomaga czy
szkodzi staremu) i **Forward Transfer** (czy stara wiedza pomaga uczyć się nowego). Te trzy metryki
pochodzą wprost z pyCLAD-a.

---

## Slajd 9 — Część 2: Wyniki
I tutaj zaczyna się druga część — to, co faktycznie udało się zbudować i policzyć.

## Slajd 10 — Co przetestowaliśmy
Króciutko o setupie. Dane to **IEEE-CIS, około 590 tysięcy transakcji**, z czego tylko ~3,5% to
fraudy — czyli mocno niezbalansowane. Podzieliliśmy je na **6 miesięcznych konceptów**, żeby
odwzorować dryf wzorców w czasie. Graf jest **homogeniczny**: węzeł to transakcja, krawędź to wspólna
encja — i kluczowe, że liczbę krawędzi trzymamy na poziomie **O(N·k)**, a nie O(N²), więc graf się
mieści w pamięci. Przetestowaliśmy trzy detektory — **OCGNN, CoLA i DOMINANT** — w trzech strategiach
continual learningu: **Naive, Replay i Cumulative**. Jako punkt odniesienia mamy też model
**statyczny**: trenowany raz na pierwszym miesiącu i oceniany na wszystkich sześciu.

## Slajd 11 — Jak czytać wyniki
Zanim pokażę liczby, jak je czytać. **ROC-AUC**: 1,0 to ideał, **0,5 to rzut monetą** — tę granicę
zaznaczam wszędzie czerwoną linią. Na **heatmapie** wiersz mówi, czego model się nauczył, a kolumna
na czym go testujemy — przekątna to świeża wiedza, a dolny wiersz to skuteczność po przejściu całego
strumienia. **ContinualAverage** to uśredniona skuteczność przez cały strumień. A **BWT i FWT** mówią,
czy wiedza przenosi się wstecz i w przód — ujemny BWT oznacza zapominanie.

## Slajd 12 — Przegląd: model × strategia
To nasz obrazek "na jeden rzut oka" — średnia skuteczność dla każdej kombinacji modelu i strategii.
Widać uczciwie: **większość pól jest blisko losowego**. Wyraźnie odstają dwa: **OCGNN z Replayem
około 0,69** oraz **DOMINANT z Replayem i V-features około 0,73** — i to jest nasz najlepszy wynik.

## Slajd 13 — Skuteczność: continual vs static
Tu zestawiamy uczenie ciągłe z modelem statycznym, słupek w słupek. Najważniejsza obserwacja:
tam gdzie Replay działa — jak w OCGNN — **uczenie ciągłe wyraźnie bije zamrożony model statyczny**.
Czyli docztanie się na bieżąco faktycznie ma sens. Przy słabszych konfiguracjach różnice są
niewielkie i oba podejścia kręcą się wokół losowego.

## Slajd 14 — Transfer wiedzy
Ten wykres to Forward i Backward Transfer. Najciekawsze jest to, że **Backward Transfer jest
praktycznie zerowy** w każdej konfiguracji. To dobra wiadomość: w naszym reżimie **prawie nie ma
catastrophic forgetting** — uczenie nowych miesięcy nie psuje wiedzy o starych. Forward Transfer
mniej więcej podąża za ogólnym poziomem skuteczności modelu.

## Slajd 15 — Koszt obliczeniowy
Teraz uczciwie o kosztach — uwaga, skala jest logarytmiczna. **DOMINANT to kobyła obliczeniowa**:
wariant z V-features trenował się ponad **7,6 godziny**, podczas gdy OCGNN czy CoLA liczyły się
w sekundy lub minuty. To wynika z tego, że DOMINANT rekonstruuje **gęstą macierz sąsiedztwa** — koszt
rośnie kwadratowo z liczbą węzłów. I dlatego jego warianty Naive oraz Cumulative na pełnym grafie
**padały na OOM** — po prostu nie mieliśmy tyle pamięci. To jest istotne ograniczenie, nie wstydliwy
szczegół.

## Slajd 16 — Pełne macierze
Dla kompletności — wszystkie macierze ROC-AUC naraz, learned w wierszach, evaluated w kolumnach.
Nie trzeba ich czytać co do liczby; chodzi o obraz: ciemniejsze, cieplejsze pola to lepsza
skuteczność. Widać, że dwa panele po prawej — DOMINANT z V-features i OCGNN/Replay — są wyraźnie
cieplejsze od reszty.

## Slajd 17 — Wnioski
Podsumowując. **Replay pomaga** — OCGNN z Replayem 0,69 wobec 0,55 dla Naive. **V-features mocno
podbijają DOMINANT** — z 0,51 na 0,73, i to nasz najlepszy wynik. **Mało zapominania** — zerowy BWT
mówi, że continual learning nie kasuje starej wiedzy. **Skala boli** — najcięższy detektor nie mieści
się w pamięci w niektórych strategiach. I uczciwie: większość konfiguracji jest blisko losowego —
zadanie jest naprawdę trudne, reżim one-class plus dryf — więc traktujemy to jako **prototyp, który
pokazuje, że cały pipeline działa od końca do końca**.

## Slajd 18 — Ograniczenia i dalsze prace
Co dalej. Po pierwsze, **zasoby** — DOMINANT w Naive i Cumulative trzeba dokończyć na większej
maszynie. Po drugie, **EWC** — to był plan, którego nie zdążyliśmy zrealizować, i to naturalny
następny krok. Po trzecie, **tuning** — więcej epok, większe bufory replay, V-features dla wszystkich
modeli. I wreszcie więcej detektorów, jak AnomalyDAE, oraz bogatsze cechy i definicje krawędzi.

## Slajd 19 — Dziękujemy
To wszystko z naszej strony — dziękujemy za uwagę i chętnie odpowiemy na pytania.

---

### Możliwe pytania i krótkie odpowiedzi
- **Czemu wyniki są niskie?** Bardzo niezbalansowane dane (3,5% fraudów), reżim one-class
  (trenujemy głównie na legalnych transakcjach) i silny dryf między miesiącami. To trudny benchmark;
  pokazujemy działający pipeline i kierunki, które realnie poprawiają wynik (Replay, V-features).
- **Co to V-features?** Dodatkowe kolumny V1–V339 z datasetu (391 cech zamiast 52). Dużo bogatszy
  sygnał wejściowy — stąd skok DOMINANT z 0,51 na 0,73.
- **Czemu DOMINANT pada na OOM, a OCGNN nie?** DOMINANT rekonstruuje gęstą macierz N×N sąsiedztwa;
  OCGNN i CoLA nie mają tego elementu, więc skalują się znacznie lepiej.
- **Czemu GPU i CPU mieszane na wykresie kosztów?** Część runów liczyliśmy na CPU, część na GPU —
  kolor słupka to pokazuje. Dla jakości (ROC-AUC) urządzenie nie ma znaczenia, dla czasu ma.
