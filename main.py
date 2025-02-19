# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Datei: main.py
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

import sys
from mbti_classifier import MBTIClassifier, BadInputError

def start_split_train_mode(data_filename, model_filename):
    '''
    Splittet die Eingabedaten in Trainings-, Validierungs- und Testdaten auf  
    und schreibt diese in den data-Ordner. Im Anschluss wird der Klassifikator 
    direkt mit diesen Dateien trainiert und getestet.

    **Parameter:**
    - data_filename (str): Dateiname/Pfad der Eingabedaten. Muss eine 
                           json-Datei mit Schlüsseln user_id und mbti sein.
    - model_filename (str): Name der Datei, in die das Modell im tsv-Format 
                            geschrieben werden soll.
    '''

    clf = MBTIClassifier()
    clf.split_dataset(data_filename)
    clf.train('data/dataset_training.json', model_filename)
    clf.evaluate('data/dataset_test.json', model_filename)

def start_train_mode(data_filename, model_filename, gold_filename):
    '''
    Trainiert den Klassifikator auf den Trainingsdaten und evaluiert ihn auf 
    den Gold-Daten.

    **Parameter:**
    - data_filename (str): Dateiname/Pfad der Trainingsdaten. Muss eine 
                           json-Datei mit Schlüsseln user_id und mbti sein.
    - model_filename (str): Name der Datei, in die das Modell im tsv-Format 
                            geschrieben werden soll.
    - gold_filename (str): Dateiname/Pfad der Testdaten. Muss eine 
                           json-Datei mit Schlüsseln user_id und mbti sein.
    '''

    clf = MBTIClassifier()
    clf.train(data_filename, model_filename)
    clf.evaluate(gold_filename, model_filename)

def start_predict_mode(data_filename, model_filename, output_filename):
    '''
    Startet den Inferenzmodus für die Eingabedaten mit dem trainierten Klassifikator.

    **Parameter:**
    - data_filename (str): Dateiname/Pfad der Eingabedaten. Muss eine 
                           json-Datei mit Schlüssel user_id sein.
    - model_filename (str): Name/Pfad der tsv-Datei mit dem trainierten Modell.
    - output_filename (str): Name/Pfad für die tsv-Datei, in die die Vorhersagen 
                             gespeichert werden sollen.
    '''

    clf = MBTIClassifier()
    clf.predict(data_filename, model_filename, output_filename)

def main(args):
    '''
    Main-Funktion. 
    Überprüft den User-Input und ruft den entsprechenden Programmmodus auf.

    **Parameter**:
    - args (list): Argumentliste aus der Kommandozeile. 
    '''

    argc = len(args)
    if argc < 3:
        print("TRAININGSMODUS: python main.py data model -t [gold]")
        print("INFERENZMODUS: python main.py data model output")
        sys.exit()
    data_filename = args[1]
    if data_filename[-5:] != '.json': 
        raise BadInputError("Eingabedatei muss im json-Format sein")
    model_filename = args[2]
    if model_filename[-4:] != '.tsv':
        raise BadInputError("Modell muss im tsv-Format sein")

    if argc == 4 and args[3] != '-t':
        # Inferenzmodus
        output_filename = args[3]
        if output_filename[-4:] != '.tsv':
            raise BadInputError("Ausgabedatei muss im tsv-Format sein")
        start_predict_mode(data_filename, model_filename, output_filename)
    elif argc == 4 and args[3] == '-t':
        # Trainingsmodus mit Datenset-Split
        start_split_train_mode(data_filename, model_filename)
    elif argc == 5 and args[3] == '-t':
        # Trainingsmodus ohne Split
        gold_filename = args[4]
        if gold_filename[-5:] != '.json':
            raise BadInputError("Eingabedatei muss im json-Format sein")
        start_train_mode(data_filename, model_filename, gold_filename)
    else:
        print("TRAININGSMODUS: python main.py data model -t [gold]")
        print("INFERENZMODUS: python main.py data model output")
        sys.exit()


if __name__ == '__main__':
    main(sys.argv)
