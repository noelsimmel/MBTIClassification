# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Datei: main.py
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

import sys
from MBTIClassifier import MBTIClassifier

def main(args):
    '''
    Main-Funktion. 
    Trainiert den Klassifikator oder benutzt ihn für die Klassifikation.

    **Parameter**:
    - args (list): Argumentliste aus der Kommandozeile. 
        1. Name der json-Datei mit den Twitter-Daten. Muss die Eigenschaften 
        mbti und user_id haben.
        2. Name der tsv-Datei, in die die Features gespeichert werden sollen 
        (im Training) bzw. aus der sie ausgelesen werden sollen (bei der Inferenz).
        3. (optional) Die Flag -t startet den Trainingsmodus. Ohne sie wird der 
        Inferenzmodus gestartet.
        4. (optional) Name der json-Datei mit den Testdaten.
    '''

    data_filename = args[1]
    feature_filename = args[2]
    assert data_filename[-5:] == '.json'
    assert feature_filename[-4:] == '.tsv'
    clf = MBTIClassifier()

    # Training
    if len(args) > 3 and args[3] == '-t':
        print("TRAINING")
        if len(args) == 4:
            # Datenset splitten
            clf.split_dataset(data_filename)
            clf.train('dataset_training.json', feature_filename)
            clf.evaluate('dataset_test.json', feature_filename)
        
        else:
            gold_filename = args[4]
            assert gold_filename[-5:] == '.json'
            # Klassifikator mit Trainingsdaten instantiieren
            # Die Features werden in feature_filename gespeichert und können 
            # im Inferenzschritt daraus eingelesen werden
            clf.train(data_filename, feature_filename)
            clf.evaluate(gold_filename, feature_filename)

    # Inferenz
    else:
        print("INFERENZ")
        # Klassifikator mit echten Daten und Features-Datei instantiieren
        clf.predict(data_filename, feature_filename)


if __name__ == '__main__':
    # Eingabe der Kommandozeile überprüfen
    argc = len(sys.argv)
    if argc < 3 or (argc > 5 or sys.argv[3] != '-t'):
        print(argc, sys.argv)
        print("TRAININGSMODUS: python main.py data model -t [gold]")
        print("INFERENZMODUS: python main.py data model")
        sys.exit()
    # Main-Funktion ausführen
    main(sys.argv)
