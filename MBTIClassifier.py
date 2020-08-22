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
import concurrent.futures
import logging
import os
import spacy
# from spacymoji import Emoji
# from spacy_langdetect import LanguageDetector
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

        self.nlp = spacy.load('de_core_news_sm')

        self.model = None

        if train:
            # self.train_data, self.val_data, self.test_data = self.split_dataset(input_filename)
            self.train_data = pd.read_json('dataset_training.json').transpose()
            self.val_data = pd.read_json('dataset_validation.json').transpose()
            self.model = self.train(self.train_data, features_filename)
            # self.model = pd.read_csv("features.tsv", sep='\t', index_col=0) # Achtung: Data leak
            self.evaluate(self.val_data)

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
        # User-ID von String zu Int casten
        # TODO: Oder User-ID immer als String?
        df['user_id'] = df['user_id'].astype('int64')
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

        train.to_json('dataset_training.json', orient='index')
        val.to_json('dataset_validation.json', orient='index')
        test.to_json('dataset_test.json', orient='index')
        logger.info("Datensätze als json-Dateien abgespeichert")
        print("Neue Dateien erstellt: \
                dataset_training.json, dataset_validation.json, dataset_test.json")

        # TODO: Trainingsdaten oversamplen, da sonst 50% der Klassen <10 mal vertreten sind
        return train, val, test

    def _thread_function(self, func, args, workers=5):
        '''
        '''

        # TODO: THREAD SAFE DATENSTRUKTUR!
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map() liefert einen Generator -> in Liste konvertieren
            return list(executor.map(func, args))
    
    def __download_tweets(self, user_id):
        '''
        '''

        try:
            # Wenn Account privat ist, leere Liste zurückgeben
            if self.api.get_user(user_id=user_id).protected:
                logger.warning(f"User {user_id}: Privater Account")
                return []

            # Pro Account 120 Tweets downloaden 
            # tweet_objects ist eine Liste von status-Objekten der Twitter-API
            # max_id bestimmt das minimale Alter der Tweets,
            # d.h. alle Tweets wurden am/vor dem 11.08.20 verfasst (für Reproduzierbarkeit)
            tweet_objects = self.api.user_timeline(user_id=user_id, 
                                                   max_id=1293223325322412032, count=120)
            # Tweets in interne Tweet-Objekte umwandeln, Retweets ignorieren
            tweets = [Tweet(t) for t in tweet_objects if 'retweeted_status' not in t._json]     
            if len(tweets) == 0:
                # Wenn die Tweets aus dem Korpus inzwischen gelöscht wurden
                logger.warning(f"User {user_id}: Keine Tweets mehr verfügbar")
            else:
                logger.debug(f"User {user_id}: {len(tweets)} Tweets heruntergeladen")
            return tweets
        # Nicht existierende Accounts abfangen
        except Exception as e:
            logger.warning(f"User {user_id}: Tweepy-Error: {str(e)}")
            return []

    def _get_valid_twitter_data(self, df):
        '''
        '''

        # Liste mit User-IDs aus DataFrame ziehen
        users = df.user_id.values
        # user_ids = [23361113, 202324814] # ESTP, IFSJ -- zum Testen
        # Alle Tweets für alle Accounts runterladen (threaded) = verschachtelte Liste
        tweet_list = self._thread_function(self.__download_tweets, users, workers=10)

        # Alle Accounts ignorieren, von denen keine Tweets runtergeladen werden konnten
        # z.B. weil der Account privat oder gelöscht ist
        # Nur die User und Tweets zurückgeben, für die Tweets runtergeladen werden konnten
        valid_users = [users[i] for i in range(len(users)) if len(tweet_list[i]) > 0]
        valid_tweets = [tweet_list[i] for i in range(len(tweet_list)) if len(tweet_list[i]) > 0]
        if len(valid_users) < len(users):
            logger.warning(f"{len(users)-len(valid_users)} nicht verfügbare User gelöscht \
                            (jetzt noch {len(valid_users)} Zeilen)")
        if len(valid_users) == 0:
            raise ValueError("Keine validen User vorhanden")
        return valid_users, valid_tweets
    
    def _get_user_features(self, user_id):
        '''
        '''

        # User-Objekt der Twitter-API downloaden und in eigene User-Klasse überführen
        user = User(self.api.get_user(user_id=user_id))
        logger.debug(f"User-Features für {user_id} extrahieren")
        # Alles als namedtuple speichern und zurückgeben
        field_names = ['user_id', 'description', 'followers_friends_ratio', 
                       'is_verified', 'has_profile_url']
        UserFeatures = namedtuple('UserFeatures', field_names)
        features = UserFeatures(user.id, user.description, user.followers_friends_ratio,
                                user.is_verified, user.has_profile_url)
        return features

    def _get_twitter_features(self, user_tweets):
        '''
        '''

        field_names = ['user_id', 'hashtags', 'mentions', 'favs', 'rts', 
                       'media_ll', 'url_ll', 'reply_ll']
        TwitterFeatures = namedtuple('TwitterFeatures', field_names)
        for user_id, tweets in user_tweets:
            # logger.debug(f"Twitter-Features für {user_id} extrahieren")
            # Tweet-Statistiken (aus den heruntergeladenen Tweets extrahieren)
            # rate = Absolute Häufigkeit des Attributes / Anzahl an Tweets für diese*n User
            hashtags_rate = sum(t.hashtags_count for t in tweets)
            mentions_rate = sum(t.mentions_count for t in tweets)
            favs_rate = sum(t.fav_count for t in tweets)
            rts_rate = sum(t.rt_count for t in tweets)
            # ll = likelihood = Wie viel % der Tweets dieses Attribut hatten
            # Da diese Attribute binär sind
            # d.h. rts_rate ist die durchschnittliche Anzahl an RTs, 
            # die ein Tweet dieser/s Users bekommt
            media_ll = sum(t.has_media for t in tweets)
            url_ll = sum(t.has_url for t in tweets)
            reply_ll = sum(t.is_reply for t in tweets)

            features = [hashtags_rate, mentions_rate, favs_rate, 
                        rts_rate, media_ll, url_ll, reply_ll]
            tweet_number = len(tweets)
            features_normalized = [user_id] + [f/tweet_number for f in features]

            # Alles als namedtuple speichern und zurückgeben
            yield TwitterFeatures(*features_normalized)
    
    def __get_spacy_features(self, texts):
        '''
        spaCy ignoriert automatisch Mentions, Hashtags und Links.
        '''

        logger.debug(f"Extrahiere spaCy-Features für {len(texts)} Tweets")
        tokens_count = special_chars = emoticons = nents = 0 
        tweet_length = word_length = sent_length = 0
        question_marks = exclamation_marks = numbers = adjectives = 0
        vocab = set()
        # nlp.pipe gibt einen Generator für doc-Objekte zurück
        # laut Docs effizienter als for t in tweets: doc = selp.nlp(t)
        for doc in self.nlp.pipe(texts, n_process=8):
            ld = len(doc)
            tokens_count += ld
            tweet_length += len(doc.text)
            sent_length += sum(len(s) for s in doc.sents)/len(list(doc.sents))
            nents += len(doc.ents) # named entities
            # @Hannah: Könnte man die Schleife vereinfachen?
            for token in doc:
                if token.is_alpha:
                    word_length += len(token)
                    vocab.add(token.lemma_)
                    if token.pos_ == 'ADJ': adjectives += 1
                elif token.is_punct:
                    if token.text == '?': question_marks += 1
                    if token.text == '!': exclamation_marks += 1
                    # Annäherung an Emoticons
                    if len(token) > 1 and ':' in token.text: emoticons += 1
                elif token.like_num: numbers += 1
                elif not token.is_ascii: special_chars += 1

                # Features hier NICHT normalisieren, da die Zahlen sonst sehr klein werden
                # Denke es ist vernachlässigbar, weil Tweets ja ungefähr gleich lang sind

                # Über Anzahl Token im Tweet normalisieren
                # to_normalize = [nents, question_marks, exclamation_marks, numbers, 
                #                 adjectives, emoji, emoticons]
                # nents, question_marks, exclamation_marks, numbers, adjectives, emoji, \
                #     emoticons = [f/ld for f in to_normalize]
        return [tokens_count, word_length, sent_length, len(vocab), 
                tweet_length, nents, question_marks, exclamation_marks,  
                numbers, adjectives, emoticons, special_chars]
    
    def _get_linguistic_features(self, user_tweets):
        '''
        '''

        field_names = ['user_id', 'tokens_count', 'word_length', 'sent_length', 
                       'vocab_size', 'tweet_length', 'named_entities',
                       'question_marks', 'exclamation_marks', 'numbers', 
                       'adjectives', 'emoticons', 'special_characters']
        LingFeatures = namedtuple('LingFeatures', field_names)
        for user_id, tweets in user_tweets:
            logger.debug(f"Linguistische Features für {user_id} extrahieren")
            # tweets = tweets[:5] # Test
            # ttttthrreeeeeeaaaaddd saafeeee .......

            # Features mit spaCy extrahieren
            texts = [t.text for t in tweets]
            spacy_features = self.__get_spacy_features(texts)
            # Über Anzahl Tweets für diese*n User*in normalisieren
            tweet_number = len(tweets)
            features_normalized = [user_id] + [f/tweet_number for f in spacy_features]

            # Alles als namedtuple speichern und zurückgeben
            yield LingFeatures(*features_normalized)
    
    def _extract_features(self, df):
        '''
        '''
        
        logger.info(f"Beginn Feature-Extraktion: {len(df)} Instanzen")
        # Listen von validen User-IDs und zugehören Tweets bekommen
        # d.h. die User, für die Tweets heruntergeladen werden konnten
        logger.info("Beginn Tweet-Download")
        users, tweets = self._get_valid_twitter_data(df)
        # Zippen, damit User-IDs und Features gematcht werden können (thread-safe-ish)
        user_tweets_zipped = list(zip(users, tweets))
        # Features extrahieren: Account-basiert, Twitter-Metadaten-basiert, textbasiert
        logger.info("Beginn Extraktion User-Features")
        user_features = self._thread_function(self._get_user_features, users, workers=10)
        # print(user_features)
        logger.info("Beginn Extraktion Tweet-Features")
        twitter_features = list(self._get_twitter_features(user_tweets_zipped))
        # print(twitter_features)
        logger.info("Beginn Extraktion linguistische Features")
        ling_features = list(self._get_linguistic_features(user_tweets_zipped))
        # print(ling_features)

        # Alles in DataFrame packen
        # @Hannah: Könnte man das vereinfachen..? 
        # zB über alle 3 Listen gleichzeitig iterieren statt zip, findest du das lesbarer?
        features_list = list(zip(user_features, twitter_features, ling_features))
        # Leeren DF mit Spaltennamen erstellen
        all_features = pd.DataFrame(columns=list(user_features[0]._fields
                                    + twitter_features[0]._fields + ling_features[0]._fields))
        # Über alle Instanzen iterieren und DF populieren, wenn IDs gleich sind
        for f in features_list:
            try:
                assert f[0].user_id == f[1].user_id == f[2].user_id
                all_features.loc[len(all_features)] = f[0] + f[1] + f[2]
            except AssertionError:
                # Wenn beim Threading etwas schief gegangen ist: loggen und ignorieren
                logger.debug(f"Assertion failed {f[0].user_id}/{f[1].user_id}/{f[2].user_id}")
                continue

        # Doppelte Spalten (= user_id) droppen
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        # Alles auf 5 Stellen hinterm Komma runden
        all_features = all_features.round(5)
        # User-ID muss int64 sein wie im Eingabe-DF
        all_features['user_id'] = all_features['user_id'].astype('int64')
        logger.info(f"Ende Feature-Extraktion: \
                    {all_features.shape[0]} Instanzen, {all_features.shape[1]-1} Features")
        return all_features
    
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
        features = df.drop(columns=['user_id', 'description'])
        # agg_features soll die aggregierten Features für jede Klasse enthalten
        agg_features = pd.DataFrame(columns=features.columns)

        # Für jede Klasse aggregieren
        for t in features.mbti.unique():
            mbti = features[features.mbti == t]
            # pd.agg erstellt einen neuen DF mit den Durchschnittswerten aller Spalten
            mbti_aggregated = mbti.agg(['mean'])
            # Am Anfang des DF eine Spalte mit dem MBTI-Typ einfügen
            if 'mbti' in mbti_aggregated: mbti_aggregated.drop(columns=['mbti'], inplace=True)
            mbti_aggregated.insert(loc=0, column='mbti', value=t)
            # An agg_features anhängen
            agg_features = agg_features.append(mbti_aggregated, ignore_index=True)

        # Alphabetisch nach MBTI sortieren
        agg_features.sort_values(by=['mbti'], inplace=True, ignore_index=True)
        # TODO: Glätten mit *100 ?
        return agg_features.round(5)

    def train(self, input_df, output_filename):
        '''
        ((download, nlp, features, aggregieren))
        '''

        logger.info(f"Beginn Training ({input_df.shape[0]} Zeilen)")
        # input_df = input_df[:2] # Test
        # Features extrahieren und an den DF anhängen
        features_only = self._extract_features(input_df)
        features = pd.merge(input_df, features_only, on=['user_id'])
        logger.info(f"Ende Training: {features.shape[0]} Zeilen, {features.shape[1]} Spalten")

        # Features aggregieren, d.h. für jede Klasse über alle Instanzen mitteln
        agg_features = self._aggregate_features(features)
        # Aggregierte Features in tsv-Datei schreiben
        agg_features.to_csv(output_filename, sep='\t')
        logger.info(f"Aggregierte Features in {output_filename} geschrieben")
        print(f"Neue Datei erstellt: {output_filename}")
        self.model = agg_features
        return agg_features

    def predict(self, df):
        '''
        '''

        logger.info(f"Beginn Vorhersage ({len(df)} Instanzen)")
        logger.info(f"Eingabe-Dimension {df.shape}, Modell-Dimension {self.model.shape}")
        # Testdaten in Feature-Repräsentation umwandeln
        features = self._extract_features(df)
        # Mergen, um invalide Daten aus df zu entfernen
        data = pd.merge(df, features, on=['user_id'])
        features_only = data.drop(columns=['user_id', 'description'])
        # Bei Validierung/Evaluierung: MBTI-Spalte entfernen
        if 'mbti' in features_only: features_only.drop(columns=['mbti'], inplace=True)

        # DF für die Differenzen erstellen, zunächst ohne Spalten
        # Finale Dimensionen: |Instanzen| * |Klassen|
        # Eine Spalte enthält jeweils die Differenzen der Instanz zu dieser Gold-Klasse
        differences = pd.DataFrame(0, index=range(len(features_only)), columns=[])

        # data soll so viele Spalten haben wie model ohne MBTI
        assert features_only.shape[1] == self.model.shape[1]-1

        # Verschachtelte Liste mit allen Features pro Klasse erstellen
        model = self.model.values.tolist()
        # Über alle Gold-Klassen iterieren
        for m in model:
            logger.debug(f"Differenzen zu Klasse {m} berechnen")
            # Absolute Differenz zwischen dieser Gold-Klasse und den Daten berechnen
            # *1, um Rechnung mit Bools möglich zu machen
            diff = abs(features_only*1 - m[1:]*1)
            # Differenzen über alle Spalten aufsummieren und in jeweiliger Spalte speichern
            differences[m[0]] = diff.sum(axis=1)

        # Für jede Spalte Klasse mit geringster Differenz/Fehler finden = Vorhersage
        preds = pd.DataFrame(differences.idxmin(axis=1), columns=['prediction'])
        assert len(preds) == len(data) == len(features_only)
        preds.insert(0, 'user_id', data.user_id)
        preds['gold'] = data.mbti
        preds['error'] = differences.min(axis=1)
        logger.info(f"Ende Vorhersage ({len(preds)} Instanzen)")
        return preds.round(5)

    def evaluate(self, gold):
        '''
        Testet den Klassifikator auf den Testdaten.
        '''

        logger.info(f"Beginn Evaluierung ({len(gold)} Test-Instanzen, {len(self.model)} Klassen)")
        # Vorhersagen für Gold-Daten erhalten
        # gold = gold[:2]
        preds = self.predict(gold)
        # TODO: Dateiname aus Shell übernehmen
        preds.to_csv('predictions.tsv', sep='\t')
        logger.info("Vorhersage in predictions.tsv geschrieben")
        print("Neue Datei erstellt: predictions.tsv")

        # Accuracy berechnen
        accuracy = sum(preds.prediction == preds.gold)/len(preds)
        logger.info(f"Ende Evaluierung (Accuracy: {accuracy}, Fehler-Schnitt: {preds.error.mean()})")
        print(f"Accuracy: {accuracy}")
        return accuracy


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    f = 'C:/Users/Natze/Documents/Uni/Computerlinguistik/6.Semester/MBTIClassification/data/TwiSty-DE.json'
    # Leerzeilen in logfile einfügen
    # test
    logger.info('\n\n')
    clf = MBTIClassifier(f, 'test.tsv', train=True)
