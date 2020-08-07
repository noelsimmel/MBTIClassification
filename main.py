# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

import sys
from MBTIClassifier import MBTIClassifier

def main(args):
    '''
    Main-Funktion. 
    Trainiert den Klassifikator oder benutzt ihn für die Klassifikation. \n

    **Parameter**: \n
    args (list): Argumentliste aus der Kommandozeile. 
        1. Name der json-Datei mit den Twitter-Daten im Format des TwiSty-Korpus'.
        2. Name der tsv-Datei, in die die Features gespeichert werden sollen 
        (im Training) bzw. aus der sie ausgelesen werden sollen (bei der Inferenz).
        3. (optional) Die Flag -t startet den Trainingsmodus. Ohne sie wird der 
        Inferenzmodus gestartet.
    '''

    data_filename = args[1]
    feature_filename = args[2]
    print(data_filename, feature_filename)
    
    # Training
    if len(args) == 4 and args[3] == '-t':
        print("TRAINING")
        # Klassifikator mit Trainingsdaten instantiieren
        # Die Features werden in feature_filename gespeichert und können im 
        # Inferenzschritt daraus eingelesen werden
        clf = MBTIClassifier(data_filename, feature_filename)
        clf.train()
        clf.evaluate()

    # Inferenz
    else:
        print("INFERENZ")
        # Klassifikator mit echten Daten und Features-Datei instantiieren
        clf = MBTIClassifier(data_filename, feature_filename)
        clf.predict()


if __name__ == '__main__':
    # Eingabe der Kommandozeile überprüfen
    argc = len(sys.argv)
    if argc < 3 or argc > 4:
        print("TRAINING: python main.py data features -t")
        print("PREDICTING: python main.py data features")
        sys.exit()
    # Main-Funktion ausführen
    main(sys.argv)
