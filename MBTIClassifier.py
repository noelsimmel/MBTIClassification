# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Datei: MBTIClassifierTrain.py – Trainiert einen Klassifikator basierend auf einem Twitter-Korpus
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

# JSON: verschachteltes dict: außen: id
# innen: other_tweet_ids (list), mbti, user_id, gender, confirmed_tweet_ids (list)
# Deutsch: {'ISTJ': 12, 'INFP': 95, 'ENTP': 26, 'ENFJ': 18, 'INTJ': 38, 'ISTP': 14, 
# 'ENTJ': 14, 'INFJ': 48, 'ENFP': 40, 'INTP': 60, 'ISFP': 16, 'ESTP': 5, 'ISFJ': 10, 
# 'ESFJ': 8, 'ESTJ': 4, 'ESFP': 3}

# Standardmodule
from collections import namedtuple
import logging
import os
# Externe Module
import dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import tweepy
# Eigene Klassen
from TwitterClasses import Tweet, User

# Logging-Details
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
file_handler = logging.FileHandler('classifier.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class MBTIClassifier:
    '''
    Trainingsklasse für den Klassifikator. 
    Aus den Trainingsdaten werden relevante Features extrahiert, aggregiert und 
    für die Inferenz in eine separate Datei gespeichert.
    '''

    def __init__(self, input_filename, features_filename, train=False):
        '''
        Konstruktor. \n

        **Parameter**: \n
        input_filename (str): Dateiname/Pfad der Eingabedaten im json-Format.
        features_filename (str): Name der tsv-Datei, in welche die Features gespeichert werden 
        sollen (Training) bzw. aus welcher sie gelesen werden sollen (Inferenz).
        '''

        # Credentials aus .env-Datei laden. Mehr Info: https://bit.ly/3glK6fd 
        dotenv.load_dotenv('.env')
        auth = tweepy.OAuthHandler(os.environ.get('CONSUMER_KEY'), os.environ.get('CONSUMER_SECRET'))
        auth.set_access_token(os.environ.get('ACCESS_KEY'), os.environ.get('ACCESS_SECRET'))
        # Verbindung zur Twitter-API herstellen
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        logger.info("Verbindung zur Twitter-API hergestellt")

        if train:
            self.train_data, self.val_data, self.test_data = self.split_dataset(input_filename)
            self.model = self.train(self.train_data, features_filename)
            # self.model = pd.read_csv("features_smalldataset.tsv", sep='\t', index_col=0) # Achtung: Data leak
            # print(self.model)
            self.evaluate(self.test_data)

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
        df.drop(columns=['index', 'other_tweet_ids', 'confirmed_tweet_ids', 'gender'], inplace=True)
        df = df[int(len(df)/3):]    # TESTEN
        logger.info(f"Daten von {fn} eingelesen ({len(df)} Zeilen)")
        return df

    def split_dataset(self, fn):
        '''
        Liest die Daten ein und teilt sie in Trainings-, Validierungs- und 
        Testdaten auf. \n

        **Parameter:** \n
        fn (str): Dateiname/Pfad der Eingabedaten.
        '''

        # Daten einlesen und vorverarbeiten
        data = self._preprocess(fn)

        # Zunächst in Trainings- und Testdaten aufsplitten
        # Stratifizieren, d.h. beim Splitten soll das Klassenverhältnis 
        # erhalten bleiben (da die Klassen sehr ungleich verteilt sind).
        # Das Verhältnis 70:10:20 ließ sich aufgrund des kleinen Datensatzes
        # nicht einhalten, da sonst nicht alle Klassen überall enthalten wären.
        # Bei 60:19:21 enthält jedes Datenset alle 16 Klassen.
        temp, test = train_test_split(data, test_size=0.21, stratify=data['mbti'])
        # Aus den Trainingsdaten die Validierungsdaten abzweigen
        train, val = train_test_split(temp, test_size=0.24, stratify=temp['mbti'])
        logger.info(f"Datensatz in Train/Val/Test gesplittet, Verhältnis \
             {(len(train)/len(data)):.2f}/{(len(val)/len(data)):.2f}/{(len(test)/len(data)):.2f}")

        # TODO: Trainingsdaten oversamplen, da sonst 50% der Klassen <10 mal vertreten sind
        return train, val, test

    def _download_tweets(self, row):
        '''
        '''

        try:
            # Wenn Account privat ist, leere Liste zurückgeben
            if self.api.get_user(user_id=row.user_id).protected:
                logger.warning(f"User {row.user_id}: Privater Account")
                return []

            # Pro Account 120 Tweets downloaden 
            # tweet_objects ist eine Liste von status-Objekten der Twitter-API
            # max_id bestimmt das minimale Alter der Tweets,
            # d.h. alle Tweets wurden am/vor dem 11.08.20 verfasst (für Reproduzierbarkeit)
            tweet_objects = self.api.user_timeline(user_id=row.user_id, 
                                                   max_id=1293223325322412032, count=120)
            # Tweets in interne Tweet-Objekte umwandeln, Retweets ignorieren
            tweets = [Tweet(t) for t in tweet_objects if 'retweeted_status' not in t._json]     
            if len(tweets) == 0:
                # Wenn die Tweets aus dem Korpus inzwischen gelöscht wurden
                logger.warning(f"User {row.user_id}: Keine Tweets mehr verfügbar")
            else:
                logger.debug(f"User {row.user_id}: {len(tweets)} Tweets heruntergeladen")
            return tweets
        # Nicht existierende Accounts abfangen
        except Exception as e:
            logger.warning(f"User {row.user_id}: Tweepy-Error: {str(e)}")
            return []

    def _get_twitter_statistics(self, row):
        '''
        '''

        logger.debug(f"Twitter-Features für {row.user_id} extrahieren")
        # 1. User-Statistiken (aus dem Profil extrahieren)
        user = User(self.api.get_user(user_id=row.user_id))

        # for t in row.tweets:
        #     print(t)

        # 2. Tweet-Statistiken (aus den heruntergeladenen Tweets extrahieren)
        tweet_number = len(row.tweets)
        # rate = Absolute Häufigkeit des Attributes / Anzahl an Tweets für diese*n User
        hashtags_rate = sum(t.hashtags_count for t in row.tweets) / tweet_number
        mentions_rate = sum(t.mentions_count for t in row.tweets) / tweet_number
        favs_rate = sum(t.fav_count for t in row.tweets) / tweet_number
        rts_rate = sum(t.rt_count for t in row.tweets) / tweet_number
        # ll = likelihood = Wie viel % der Tweets dieses Attribut hatten
        # Da diese Attribute binär sind
        # d.h. rts_rate ist die durchschnittliche Anzahl an RTs, 
        # die ein Tweet dieser/s Users bekommt
        media_ll = sum(t.has_media for t in row.tweets) / tweet_number
        url_ll = sum(t.has_url for t in row.tweets) / tweet_number
        reply_ll = sum(t.is_reply for t in row.tweets) / tweet_number

        # Alles als namedtuple speichern und zurückgeben
        field_names = ['description', 'followers_friends_ratio', 'is_verified', 
                       'has_profile_url', 'hashtags', 'mentions', 'favs', 'rts', 
                       'media_ll', 'url_ll', 'reply_ll']
        TwitterStatistics = namedtuple('TwitterStatistics', field_names)
        stats = TwitterStatistics(user.description, user.followers_friends_ratio,
                                  user.is_verified, user.has_profile_url, 
                                  hashtags_rate, mentions_rate, favs_rate, rts_rate, 
                                  media_ll, url_ll, reply_ll)
        return stats
    
    def _get_linguistic_features(self, row):
        '''
        ((Zuerst als Test ein paar einfache Features extrahieren))
        '''

        logger.debug(f"Linguistische Features für {row.user_id} extrahieren")
        tweet_number = len(row.tweets)
        # for t in row.tweets:
            # sents = self.__sentence_split(t)
            # tokens = self.__tokenize(t)
            # pos = self.__pos_tag(t)
            # lemma = self.__lemmatize(t)
            # named_entities = self.__named_entity_recognition(t)

        length = sum(len(t.text) for t in row.tweets) / tweet_number
        tokens = sum(len(t.text.split()) for t in row.tweets) / tweet_number
        questions = sum(t.text.count('?')/len(t.text) for t in row.tweets) / tweet_number
        exclamations = sum(t.text.count('!')/len(t.text) for t in row.tweets) / tweet_number

        # Alles als namedtuple speichern und zurückgeben
        # field_names = ['chars', 'letters', 'capitals', 'numbers', 'special_chars', 
        #                'punctuation', 'questions', 'exclamations', 'words', 'word_length',
        #                'long_words', 'emoticons', 'emoji', 'typos', 'type_token_ratio',
        #                'hapax_legomena', 'sentences', 'sentence_length', 'pos', 
        #                'sentiment', 'named_entities']
        field_names = ['length', 'tokens', 'questions', 'exclamations']
        LingStatistics = namedtuple('LingStatistics', field_names)
        stats = LingStatistics(length, tokens, questions, exclamations)
        return stats
    
    def extract_features(self, df):
        '''
        '''

        logger.info(f"Beginn Feature-Extraktion: DF hat {len(df.columns)} Spalten")
        # Dataframe kopieren, fortlaufenden Index erstellen
        features = df.reset_index()
        # zum testen immer die selben ids nehmen
        # features = features.iloc[0:2]
        # features.at[0, 'mbti'] = 'ESTP'
        # features.at[0, 'user_id'] = 23361113
        # features.at[1, 'mbti'] = 'ISFJ'
        # features.at[1, 'user_id'] = 202324814

        # Tweets pro Zeile herunterladen und als Liste in neuer Spalte speichern
        full_tweets = features.apply(self._download_tweets, axis=1)
        features = features.assign(tweets=full_tweets.values)

        # Alle Zeilen mit leeren Tweet-Listen löschen (z.B. weil Profil gelöscht wurde)
        old_len = len(features)
        features = features[features['tweets'].map(lambda d: len(d)) > 0]
        features.reset_index(drop=True, inplace=True)
        if len(features) < old_len:
            logger.warning(f"{(old_len-len(features))} nicht verfügbare User gelöscht \
                            (jetzt noch {len(features)} Zeilen)")
        if len(features) == 0:
            raise ValueError("Keine validen User vorhanden")

        # Metadaten aus Userprofil und Tweets ziehen, 11 neue Spalten erstellen
        twitter_stats = features.apply(self._get_twitter_statistics, axis=1)
        twitter_stats_df = pd.DataFrame(list(twitter_stats), columns=twitter_stats[0]._fields)
        features = pd.concat([features, twitter_stats_df], axis=1)

        # Linguistische Features bestimmen
        ling_features = features.apply(self._get_linguistic_features, axis=1)
        ling_stats_df = pd.DataFrame(list(ling_features), columns=ling_features[0]._fields)
        features = pd.concat([features, ling_stats_df], axis=1)

        # TODO: private accs handlen!
        logger.info(f"Ende Feature-Extraktion: DF hat {len(features.columns)} Spalten")
        return features.round(10)

    def _aggregate_features(self, df):
        '''
        Aggregiert die Features in einem DataFrame für alle Klassen. Da alle Features 
        numerisch sind, wird jeweils der Durchschnitt über alle Instanzen einer Klasse gebildet.

        **Parameter:** \n
        df (DataFrame): Enthält für jede Instanz (User) alle berechneten Features (Ergebnis von 
        extract_features).

        **Returns:** \n
        DataFrame mit |Klassen| Zeilen und |Features| Spalten. Hat weniger Spalten als Input-DF, 
        da z.B. User-ID und Tweets gedroppt werden.
        '''

        logger.info("Features pro Klasse aggregieren")
        # Überflüssige Spalten löschen
        features = df.drop(columns=['index', 'user_id', 'tweets', 'description'])
        # agg_features soll die aggregierten Features für jede Klasse enthalten
        agg_features = pd.DataFrame(columns=features.columns)

        # Für jede Klasse aggregieren
        for t in features.mbti.unique():
            mbti = features[features.mbti == t]
            # pd.agg erstellt einen neuen DF mit den Durchschnittswerten aller Spalten
            mbti_aggregated = mbti.agg(['mean'])
            # Am Anfang des DF eine Spalte mit dem MBTI-Typ einfügen
            mbti_aggregated.insert(loc=0, column='mbti', value=t)
            # An agg_features anhängen
            agg_features = agg_features.append(mbti_aggregated, ignore_index=True)

        # TODO: Glätten mit *100 ?
        return agg_features.round(5)

    def train(self, df, output_filename):
        '''
        ((download, nlp, features, aggregieren))
        '''

        logger.info(f"Beginn Training ({len(df)} Zeilen)")
        # Features aus Trainingsdaten extrahieren
        features = self.extract_features(df)
        # Features aggregieren, d.h. für jede Klasse über alle Instanzen mitteln
        agg_features = self._aggregate_features(features)
        # Aggregierte Features in tsv-Datei schreiben
        agg_features.to_csv(output_filename, sep='\t')
        logger.info(f"Aggregierte Features in {output_filename} geschrieben")
        logger.info(f"Ende Training ({len(df)} Zeilen, {len(agg_features.columns)} Spalten)")
        return agg_features

    def predict(self, data):
        '''
        '''

        logger.info(f"Beginn Vorhersage ({len(data)} Instanzen)")
        # DF für die Differenzen erstellen, zunächst ohne Spalten
        # Finale Dimensionen: |Instanzen| * |Klassen|
        # Eine Spalte enthält jeweils die Differenzen der Instanz zu dieser Gold-Klasse
        differences = pd.DataFrame(0, index=range(len(data)), columns=[])

        # Für Testfall: MBTI-Spalte entfernen
        if 'mbti' in data: data = data.drop(columns=['mbti'])
        # data soll so viele Spalten haben wie model ohne MBTI
        assert data.shape[1] == self.model.shape[1]-1

        # Verschachtelte Liste mit allen Features pro Klasse erstellen
        gold = self.model.values.tolist()
        # Über alle Gold-Klassen iterieren
        for g in gold:
            logger.debug(f"Differenzen zu Klasse {g} berechnen")
            # Absolute Differenz zwischen dieser Gold-Klasse und den Daten berechnen
            # *1, um Rechnung mit Bools möglich zu machen
            diff = abs(data*1 - g[1:]*1)
            # Differenzen über alle Spalten aufsummieren und in jeweiliger Spalte speichern
            differences[g[0]] = diff.sum(axis=1)

        # Für jede Spalte Klasse mit geringster Differenz/Fehler finden = Vorhersage
        preds = pd.DataFrame(differences.idxmin(axis=1), columns=['prediction'])
        # Zusätzlich den Fehler (= Differenz) speichern
        preds['error'] = differences.min(axis=1)
        logger.info(f"Ende Vorhersage ({len(data)} Instanzen)")
        return preds.round(5)

    def validate(self):
        '''
        Validiert den Klassifikator auf dem Datenset für die Validierung.
        '''

        pass

    def evaluate(self, gold):
        '''
        Testet den Klassifikator auf den Testdaten.
        '''

        logger.info(f"Beginn Evaluierung ({len(gold)} Test-Instanzen, {len(self.model)} Klassen)")
        # Testdaten in Feature-Repräsentation umwandeln
        gold_features = self.extract_features(gold)
        # Alles droppen außer die MBTI- und Features-Spalten
        gold_features_only = gold_features.drop(columns=['index', 'user_id',
                                                         'tweets', 'description'])
        # Vorhersagen für Gold-Daten erhalten
        logger.info(f"Vorhersage erhalten: \
            Eingabe-Dimension {gold_features_only.shape}, Modell-Dimension {self.model.shape}")
        preds = self.predict(gold_features_only)
        preds['gold'] = gold_features_only.mbti
        # TODO: Dateiname aus Shell übernehmen
        preds.to_csv('predictions.tsv', sep='\t')
        
        # Accuracy berechnen
        assert len(preds) == len(gold_features_only)
        accuracy = sum(preds.prediction == gold_features_only.mbti)/len(preds)
        logger.info(f"Ende Evaluierung (Accuracy: {accuracy}, Fehler-Schnitt: {preds.error.mean()})")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    f = 'C:/Users/Natze/Documents/Uni/Computerlinguistik/6.Semester/MBTIClassification/data/TwiSty-DE.json'
    # Leerzeilen in logfile einfügen
    logger.info('\n\n')
    clf = MBTIClassifier(f, 'test.tsv', train=True)
