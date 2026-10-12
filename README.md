
# Predviđanje cena polovnih automobila

## Opis projekta

Cilj ovog projekta je razvoj modela mašinskog učenja za predviđanje cena polovnih automobila na osnovu njihovih karakteristika.

Model predviđa cenu automobila u američkim dolarima (USD), koristeći podatke kao što su proizvođač, model, godina proizvodnje, stanje, kilometraža, tip goriva, zapremina motora, boja, menjač, pogon i segment vozila.

## Skup podataka

Za izradu projekta koristi se skup podataka `cars.csv`.

Skup sadrži 56.244 oglasa automobila i 12 originalnih kolona.

Ciljna promenljiva je `priceUSD`, odnosno cena automobila u američkim dolarima.

## Eksplorativna analiza podataka (EDA)

Eksplorativna analiza podataka urađena je u Jupyter Notebook-u `notebooks/01_eda.ipynb`.

Analiza obuhvata:
- pregled strukture i tipova podataka;
- proveru nedostajućih vrednosti;
- osnovnu statističku analizu;
- analizu raspodele cena;
- analizu godina proizvodnje;
- pregled najčešćih proizvođača;
- analizu odnosa između numeričkih karakteristika.

## Čišćenje i priprema podataka

Podaci se obrađuju pomoću sledećih postupaka:
- uklanjanje dupliranih redova;
- uklanjanje redova sa nevažećim cenama;
- obrada nedostajućih vrednosti;
- kreiranje novih karakteristika;
- kodiranje kategorijskih promenljivih;
- standardizacija numeričkih promenljivih;
- podela podataka na skup za treniranje i testiranje.

## Kreiranje novih karakteristika

U projektu su kreirane sledeće karakteristike:
- `car_age` — starost automobila;
- `mileage_per_year` — prosečna kilometraža po godini starosti;
- `engine_volume_liters` — zapremina motora izražena u litrima;
- `is_newer_car` — pokazatelj da li je automobil proizveden 2015. godine ili kasnije;
- `is_high_mileage` — pokazatelj da li automobil ima najmanje 200.000 kilometara;
- `brand_model` — kombinacija proizvođača i modela automobila.

## Korišćeni modeli mašinskog učenja

U projektu su trenirana i upoređena četiri regresiona modela:

1. Linearna regresija (Linear Regression)
2. Stablo odlučivanja (Decision Tree Regressor)
3. Slučajna šuma (Random Forest Regressor)
4. Gradijentno pojačavanje (Gradient Boosting Regressor)

## Evaluacija modela

Performanse modela procenjuju se pomoću sledećih metrika:

- **MAE** — srednja apsolutna greška;
- **MSE** — srednja kvadratna greška;
- **RMSE** — koren srednje kvadratne greške;
- **R²** — koeficijent determinacije.

Niže vrednosti MAE, MSE i RMSE uglavnom ukazuju na manje greške u predviđanju. Vrednost R² bliža jedinici uglavnom ukazuje na bolje objašnjavanje varijacija cena u test skupu.

## Struktura projekta

- `data/` — skup podataka;
- `notebooks/` — Jupyter Notebook za analizu;
- `src/data_cleaning.py` — čišćenje podataka;
- `src/feature_engineering.py` — kreiranje novih karakteristika;
- `src/data_preprocessing.py` — priprema i podela podataka;
- `src/model_training.py` — treniranje modela;
- `src/model_evaluation.py` — evaluacija modela;
- `src/model_comparison.py` — poređenje modela;
- `models/` — sačuvani istrenirani modeli;
- `results/` — rezultati evaluacije i grafikoni poređenja;
- `requirements.txt` — potrebne Python biblioteke.

## Instalacija

Potrebno je imati instaliran Python.

Instalacija potrebnih biblioteka vrši se komandom:

```bash
python -m pip install -r requirements.txt
```

## Pokretanje projekta

Komande se izvršavaju iz glavnog foldera projekta:

```bash
python -m src.model_training
python -m src.model_evaluation
python -m src.model_comparison
```

## Rezultati

Istrenirani modeli čuvaju se u folderu `models/` u `.joblib` formatu.

Rezultati evaluacije čuvaju se u fajlu `results/model_metrics.csv`, dok se tabela poređenja čuva u fajlu `results/model_comparison.csv`.

Grafikoni poređenja modela čuvaju se u folderu `results/`.

## Zaključak

Projekat obuhvata celokupan proces razvoja modela za predviđanje cena polovnih automobila: od istraživanja i čišćenja podataka, preko kreiranja novih karakteristika i treniranja više modela, do evaluacije i poređenja njihovih performansi.

Najbolji model bira se na osnovu ostvarenih rezultata na test skupu.
