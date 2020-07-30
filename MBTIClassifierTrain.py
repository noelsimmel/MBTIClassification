# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

# JSON: verschachteltes dict: außen: id
# innen: other_tweet_ids (list), mbti, user_id, gender, confirmed_tweet_ids (list)
# Deutsch: {'ISTJ': 12, 'INFP': 95, 'ENTP': 26, 'ENFJ': 18, 'INTJ': 38, 'ISTP': 14, 
# 'ENTJ': 14, 'INFJ': 48, 'ENFP': 40, 'INTP': 60, 'ISFP': 16, 'ESTP': 5, 'ISFJ': 10, 
# 'ESFJ': 8, 'ESTJ': 4, 'ESFP': 3}

import pandas as pd
from sklearn.model_selection import train_test_split

class MBTIClassifierTrain:
    '''
    Trainingsklasse für den Klassifikator. 
    Aus den Trainingsdaten werden relevante Features extrahiert, aggregiert und 
    für die Inferenz in eine separate Datei gespeichert.
    '''

    def __init__(self, input_filename, output_filename):
        '''
        Konstruktor. \n

        **Parameter**: \n
        input_filename (str): Dateiname/Pfad der Eingabedaten im json-Format.
        output_filename (str): Name der Datei, in welche die Features gespeichert werden sollen.
        '''

        self.train, self.val, self.test = self.split_dataset(input_filename)

    def _preprocess(self, fn):
        '''
        Hilfsfunktion für split_dataset. Verarbeitet das TwiSty-Korpus vor und 
        speichert die Daten in einem Pandas Dataframe.
        '''

        # Dataframe mit n Zeilen, 6 Spalten
        df = pd.read_json(fn).transpose()
        # Fortlaufender Index statt User ID als Index
        df.reset_index(inplace=True)
        # Irrelevante Spalten löschen
        df.drop(columns=['index', 'other_tweet_ids', 'gender'], inplace=True)
        # Spalte mit Tweet IDs umbenennen
        df.rename(columns={'confirmed_tweet_ids': 'tweet_ids'}, inplace=True)
        # Pro User nur 100 Tweets betrachten, um Twitter-API-Limit nicht zu überschreiten
        for i in range(len(df)):
            df['tweet_ids'][i] = df['tweet_ids'][i][:100]
        return df

    def split_dataset(self, fn):
        '''
        Liest die Daten ein und teilt sie in Trainings-, Validierungs- und 
        Testdaten im Verhältnis ca. 70:10:20 auf. \n

        **Parameter:** \n
        fn (str): Dateiname/Pfad der Eingabedaten.
        '''

        data = self._preprocess(fn)
        # Zunächst in Trainings- und Testdaten aufsplitten
        # Stratifizieren, d.h. beim Splitten soll das Klassenverhältnis 
        # erhalten bleiben (da die Klassen sehr ungleich verteilt sind)
        temp, test = train_test_split(data, test_size=0.2, stratify=data['mbti'])
        # Aus den Trainingsdaten 10% für Validierungsdaten abzweigen
        train, val = train_test_split(temp, test_size=0.1, stratify=temp['mbti'])
        


    def extract_features(self):
        pass

    def save_features(self):
        pass

    def validate(self):
        '''
        Validiert den Klassifikator auf dem Datenset für die Validierung.
        '''

        pass

    def evaluate(self):
        '''
        Testet den Klassifikator auf den Testdaten.
        '''

        pass
