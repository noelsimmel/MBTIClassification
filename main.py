# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

import sys
from preprocessing import preprocess
from MBTIClassifier import MBTIClassifier
from MBTIClassifierTrain import MBTIClassifierTrain 

def main(args):
    data_filename = args[1]
    feature_filename = args[2]
    print(data_filename, feature_filename)
    
    # Training
    if argc == 4 and args[3] == '-t':
        print("TRAINING")
        pass

    # Inferenz
    else:
        print("INFERENZ")
        pass


if __name__ == '__main__':
    # Eingabe der Kommandozeile überprüfen
    argc = len(sys.argv)
    if argc < 3 or argc > 4:
        print("TRAINING: python main.py data features -t")
        print("PREDICTING: python main.py data features")
        sys.exit()
    # Main-Funktion ausführen
    main(sys.argv)
