# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Datei: test_module.py – Unit-Tests
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

import logging
import os
import pandas as pd
import unittest
from mbti_classifier import MBTIClassifier

logging.disable(logging.CRITICAL)

class TestClassifier(unittest.TestCase):
    '''
    Test-Klasse für den MBTI-Klassifikator. 
    Aufruf mit python -m unittest test_module.
    '''
    def setUp(self):
        self.clf = MBTIClassifier()
        self.data = 'data/dataset_unittest.json'

    def test_1_api_connection_established(self):
        # Einen Tweet von @Twitter downloaden
        tweet = self.clf.api.get_status(507816092)
        self.assertEqual(tweet.text, "Twitter is about to go offline.  See you soon!")

    def test_2_data_load_from_json(self):
        # Daten aus einer json-Datei in einen DataFrame laden
        df = self.clf._preprocess(self.data)
        self.assertIs(type(df), pd.DataFrame)

    def test_3_dataset_split(self):
        # Datenset aufsplitten
        train_data, val_data, test_data = self.clf.split_dataset('data/TwiSty-DE.json')
        os.remove('data/dataset_training.json')
        os.remove('data/dataset_validation.json')
        os.remove('data/dataset_test.json')
        self.assertTrue(len(train_data) > len(test_data) > len(val_data))

    def test_4_feature_extraction(self):
        # Daten einlesen und Features extrahieren
        df = self.clf._preprocess(self.data)
        features = self.clf._extract_features(df)
        self.assertEqual(features.shape, (14, 24)) # 14 Instanzen

    def test_5_training(self):
        # Modell trainieren
        aggregated_features = self.clf.train(self.data, 'unittest.tsv')
        # Datenset enthält 8 verschiedene Klassen
        self.assertEqual(len(aggregated_features), 8) # 8 Klassen

    def test_6_predicting(self):
        # Modell benutzen um Vorhersagen zu treffen
        # Aus Einfachheitsgründen Trainingsset = Eingabedaten
        predictions = self.clf.predict(self.data, 'unittest.tsv', 'unittest_predictions.tsv')
        self.assertEqual(predictions.shape, (14, 4)) # 14 Instanzen

    def test_7_evaluating(self):
        # Accuracy evaluieren
        # Aus Einfachheitsgründen Trainingsset = Testset
        accuracy = self.clf.evaluate(self.data, 'unittest.tsv')
        os.remove('unittest.tsv')
        os.remove('unittest_predictions.tsv')
        os.remove('gold_predictions.tsv')
        self.assertTrue(accuracy > 0.0)


if __name__ == '__main__':
    unittest.main(buffer=True)
