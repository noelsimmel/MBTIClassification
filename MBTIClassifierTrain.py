# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

# JSON: verschachteltes dict: außen: id
# innen: other_tweet_ids (list), mbti, user_id, gender, confirmed_tweet_ids (list)

class MBTIClassifierTrain:
    '''
    Trainingsklasse für den Klassifikator. 
    Aus den Trainingsdaten werden relevante Features extrahiert, aggregiert und 
    für die Inferenz in eine separate Datei gespeichert.
    '''

    def __init__(self, corpus_filename, output_filename):
        '''
        Konstruktor. \n

        **Parameter**: \n
        corpus_filname (str): Dateiname des Korpus' im json-Format
        output_filename (str): Name der Datei, in welche die Features gespeichert werden sollen
        '''

        self.train_data, self.val_data, self.test_data = self.split_dataset(corpus_filename)

    def split_dataset(self):
        pass

    def extract_features(self):
        pass

    def save_features(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass
