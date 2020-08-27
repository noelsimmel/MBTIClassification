# MBTIClassification
Erstellt Persönlichkeitsprofile für Twitter-Nutzende anhand des Myers-Briggs Type Indicators (MBTI).

## Installation
Entwickelt mit Python 3.8.3 unter Windows 10 Home.

```main.py```, ```mbti_classifier.py``` und ```twitter_classes.py``` müssen heruntergeladen werden. Es wird ein Standard-Zugang zur Twitter-API benötigt; die Zugangsdaten sollen in einer .env-Datei abgespeichert werden (Vorlage im Repo).

Der Klassifikator kann z.B. auf dem (deutschen) [TwiSty-Korpus](https://www.uantwerpen.be/en/research-groups/clips/research/datasets/) trainiert werden. Alternativ kann jede .json-Datei mit den Schlüsseln "user_id" und "mbti" verwendet werden. Die User-ID ist für jeden Twitter-Account einzigartig und kann nicht geändert werden, anders als der Nutzer-/Anzeigename (@). Nutzernamen können z.B. [hier](https://tweeterid.com/) in User-IDs umgewandelt werden.

## Nutzung
### Training
```cmd
python main.py data model -t [gold]
```

#### Argumente
* ```data```: Dateipfad zu den Trainingdaten. Format: .json mit Schlüsseln "user_id" und "mbti"
* ```model```: Dateipfad, in den das trainierte Modell geschrieben werden soll. Format: .tsv
* ```-t```: Flagge, die den Trainingsmodus startet
* ```gold```: Dateipfad zu den Testaten für die Evaluation, wenn vorhanden. Wenn das Datenset noch nicht in Trainings-/Testdaten aufgesplittet ist, wird ```data``` intern gesplittet. Format: .json, s.o.

#### Beispiel (ungesplittetes Korpus)
```cmd
python main.py data/TwiSty-DE.json model.tsv -t
```

#### Beispiel (gesplittetes Korpus)
```cmd
python main.py data/dataset_training.json model.tsv -t data/dataset_test.json
```

### Inferenz
```cmd
python main.py data model output
```

#### Argumente
* ```data```: Dateipfad zu den Eingabedaten. Format: .json mit Schlüssel "user_id"
* ```model```: Dateipfad aus dem das trainierte Modell eingelesen werden soll. Format: .tsv
* ```output```: Dateipfad, in den die Vorhersagen geschrieben werden sollen. Format: .tsv

#### Beispiel
```cmd
python main.py data/my_data.json model.tsv predictions.tsv
```

## Unit-Tests
Einsehbar in ```test_module.py```. Aus dem deutschen TwiSty-Korpus wurde ein kleines Subset ```dataset_unittest.json``` mit 20 User-IDs entnommen, auf dem getestet wird.

```cmd
python -m unittest test_module
```
