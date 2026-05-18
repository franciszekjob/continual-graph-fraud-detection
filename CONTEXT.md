Problem - Fraud Detection

Franciszek Job, Jakub Ciszewski

Chcemy zrobić adaptacyjny system wykrywania fraudów

Problemy obecnych rozwiązań:
- modele się degradują, czyli modele wytrenowane na danych są użyteczne przez krótki czas, ponieważ scamerzy wymyślają nowe metody
- klasyczne detektory nie uwzględniają relacji między danymi, czyli skupiają się tylko na jednej transakcji, podczas gdy na jeden fraud składa się często siatka wielu transakcji

Nasze rozwiązanie:
- Wykorzystujemy graf reprezentujący transakcje, w którym wierzchołkami są transakcje a krawędź to wspólna encja między dwoma krawędziami, czyli np. wspólna karta kredytowa, wspólny adres ip itp.
- System pracuje w trybie continual learning, czyli doucza się nowych wzorców na bieżąco

Problemy naszego rozwiązania:
- Zbiory danych dla transakcji bankowych są bardzo obszerne co podczas podejścia grafowego może być bardzo kosztowne pamięciowo - możemy otrzymać mega-graf, który nie zmieści się w pamięci RAM.
- Catastrophic forgetting - model ucząc się nowych wzorców przez to, że próbuje dopasować swoją strategię do nowych danych może zapominać o poprzednich wzorcach

Rozwiązania, które będziemy testować:
- Optymalne wybranie encji, które będą tworzyć krawędzie
- Replay - podczas uczenia modelu na nowych danych dorzucamy do nich stare wzorce
- EWC - Elastic weight consolidation, czyli karanie modelu za zmienianie istotnych wag dla nowych przykładów

Architektura systemu:
Data ingestion -> Preprocessing danych (podział na okresy) -> Stworzenie grafu(NetworkX) -> Stworzenie modelu z pyGOD -> opakowanie modelu w pyCLAD -> ewaluacja wyników

Zbiór danych:
IEEE-CIS Fraud Detection - duży zbiór danych; etykietowany; z podziałem czasowym

Metryki użyte do ewaluacji:
	Grupa 1. (czy detektor wykrywa fraudy?)
		- ROC-AUC
	Grupa 2. (czy uczy się w czasie?)
		- Forgetting (czy model zapomina?)
		- Backward transfer (czy nauka nowych rzeczy pomaga czy szkodzi starym?)
		- Forward transfer (czy nauczone stare rzeczy pomagają w nauce nowych?)
Biblioteki:
- numpy
- pandas
- networkx
- pygod
- pyclad