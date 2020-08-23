# MBTIClassification
Erstellt Persönlichkeitsprofile für Twitter-Nutzende anhand des Myers-Briggs Type Indicators (MBTI).

## Installation
Entwickelt mit Python 3.8.3 unter Windows 10 Home.

```main.py```, ```MBTIClassifier.py``` und ```TwitterClasses.py``` müssen heruntergeladen werden. Es wird ein Standard-Zugang zur Twitter-API benötigt; die Zugangsdaten sollen in einer .env-Datei abgespeichert werden (Vorlage im Repo).

Der Klassifikator kann z.B. auf dem (deutschen) [TwiSty-Korpus](https://www.uantwerpen.be/en/research-groups/clips/research/datasets/) trainiert werden. Alternativ kann jede .json-Datei mit den Schlüsseln 'user_id' und 'mbti' verwendet werden.

## Nutzung
### Training
```cmd
python main.py data model -t [gold]
```

### Inferenz
```cmd
python main.py data model
```

### Argumente
* ```data```: Dateipfad zu den Trainingdaten (Trainingsmodus) bzw. Eingabedaten (Inferenzmodus). Format: .json mit Schlüsseln 'user_id' und 'mbti' (für Training)
* ```model```: Dateipfad, in den das trainierte Modell geschrieben werden soll (T) bzw. aus dem es eingelesen werden soll (I). Format: .tsv
* ```-t```: Flagge, die den Trainingsmodus startet
* ```gold```: Dateipfad zu den Gold-Daten für die Evaluation im Trainingsmodus. Wenn das Datenset noch nicht in Trainings-/Testdaten aufgesplittet ist, wird ```data``` intern gesplittet. Format: .json, s.o.

