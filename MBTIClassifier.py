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
            features = self.train(self.train_data, features_filename)
            # self.evaluate(self.test_data, features)

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
        # Einige User haben <100 Tweets, s. Report
        for i in range(len(df)):
            df['tweet_ids'][i] = df['tweet_ids'][i][:50] # TESTEN
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
        logger.info(f"Datensatz in Train/Val/Test gesplittet, \
                      Verhältnis {(len(train)/len(data)):.2f}/{(len(val)/len(data)):.2f}/{(len(test)/len(data)):.2f}")

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

            # Tweets downloaden
            # tweet_objects ist eine Liste von status-Objekten der Twitter-API
            tweet_objects = self.api.statuses_lookup(row.tweet_ids, include_entities=True, trim_user=True)
            tweets = [Tweet(t) for t in tweet_objects]     
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

        # 2. Tweet-Statistiken (aus den heruntergeladenen Tweets extrahieren)
        tweet_number = len(row.tweets)
        # rate = Absolute Häufigkeit des Attributes / Anzahl an Tweets für diese*n User
        hashtags_rate = sum(t.hashtags_count for t in row.tweets) / tweet_number
        mentions_rate = sum(t.mentions_count for t in row.tweets) / tweet_number
        favs_rate = sum(t.fav_count for t in row.tweets) / tweet_number
        rts_rate = sum(t.rt_count for t in row.tweets) / tweet_number
        # ll = likelihood = Wie viel % der Tweets dieses Attribut hatten
        # Da diese Attribute binär sind
        # d.h. rts_rate ist die durchschnittliche Anzahl an RTs, die ein Tweet
        # dieser/s Users bekommt, retweeting_rate ist die Likelihood, dass ein
        # Tweet auf dem Profil ein RT von einer/m anderen User ist
        media_ll = sum(t.has_media for t in row.tweets) / tweet_number
        url_ll = sum(t.has_url for t in row.tweets) / tweet_number
        replying_rate = sum(t.is_reply for t in row.tweets) / tweet_number
        retweeting_rate = sum(t.is_retweet for t in row.tweets) / tweet_number

        # Alles als namedtuple speichern und zurückgeben
        field_names = ['description', 'followers_friends_ratio', 'is_verified', 
                       'has_profile_url', 'hashtags', 'mentions', 'favs', 'rts', 
                       'media_l', 'url_l', 'reply_l', 'rt_l']
        TwitterStatistics = namedtuple('TwitterStatistics', field_names)
        stats = TwitterStatistics(user.description, user.followers_friends_ratio,
                                  user.is_verified, user.has_profile_url, 
                                  hashtags_rate, mentions_rate, favs_rate, rts_rate, 
                                  media_ll, url_ll, replying_rate, retweeting_rate)
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
        # features.at[0, 'tweet_ids'] = ['101404677203169280', '101404935366778880', '10214018918', '103040428324040704', '103081555932626945', '10319886150', '103446102119948289', '103742221991411712', '10431637192', '10445252623', '10568717686', '106310163694239744', '106415999863099392', '106476646546157568', '10665293525291009', '106701306441375745', '10718067246', '10718573608', '108163427507253248', '108163584521019392', '108171924709974017', '108174978452701184', '108191804846899200', '10922710203', '10975244870', '110762382476849153', '11297191061', '11305958043', '11322352961', '11357030863', '113695451907235840', '113701358519070722', '114343972444450816', '115386636136755200', '115387392533331968', '115387925935566848', '117361531141890048', '117363353491476480', '11747351979', '11814407094', '11815771168', '120932563475898368', '12273183256', '126764155335221248', '12702049402', '12792014493', '12834100641538049', '12871420430', '129468032593563648', '129468311120527360', '1297813941', '1299366057', '1299594489', '1300916626', '1301049112', '1301267869', '1302610919', '1304358753', '1304362603', '130685425403695105', '1307180045', '1309811830', '1310116711', '1310778088', '1313296532', '131496253589696512', '131496591440871425', '1315290433', '131731864204480512', '131733103487090688', '131856638092128256', '1318566440', '1320797384', '132082400850223104', '132086501679964160', '1320933809', '1321131469', '1321134663', '1321626331', '132454086598594561', '132537895977361408', '1326871050', '1327489704', '1328326334', '1328414894', '1328561530', '13301676819', '1330984257', '133497260368605184', '1335693226', '1336844713', '1337851877', '133919075410776065', '133919634352119808', '133922691588698112', '1340937630', '1342037121', '134319242022625281', '134320412870975488', '134368615758700545']
        # features.at[1, 'mbti'] = 'ISFJ'
        # features.at[1, 'user_id'] = 202324814
        # features.at[1, 'tweet_ids'] = ['145982462084907008', '145990139867439105', '145991339677454337', '145993360832864256', '145997613878083585', '145999198079299584', '146628056151371776', '146749783393042432', '146751041998827520', '146752027337293824', '146887484435996672', '147021766575927297', '148522442917294081', '148528848886169601', '148739591002800129', '149561269085683712', '150351650165493761', '150356522998829060', '150357323142017024', '150374042719887360', '151016312846565376', '151017315268427776', '151350399905959936', '151374062382358529', '152151739146059776', '152152242286366721', '152154276616085504', '152190349354352641', '152428479185555457', '152527777634058240', '152717365010890752', '152893034776891392', '153981575967674369', '153982287707521024', '153983328935096322', '154206380725764097', '154678318757707776', '154683503403995136', '155026941597065217', '155027788091494401', '155028586720534529', '155028822876618753', '155029724920422400', '155031342487306240', '155048115316072448', '155048877609857024', '155049511289487360', '155050982148014080', '155280139620585473', '155283876414099457', '155287039712034819', '155287813355601920', '155289921245028352', '155432187792076800', '155461102866661376', '155462652137713664', '155463585454231552', '155464253728489473', '155465562603012096', '155475553724534785', '155489145282760705', '155489960835813379', '155798049019527168', '155825260065865728', '155828311736590337', '155829329211817984', '155830443080548352', '155831692572114944', '155833002344189952', '155833934641496065', '155836275801337858', '155837957096480768', '155839209888944128', '155839524663074817', '155840415965261824', '156137443672854528', '156137583292841984', '156138995070402560', '156526770840027136', '156767294901583872', '156772414380974081', '156801089465884672', '156819203796647936', '156852872221442048', '156858662638469120', '156886835124117504', '157086215273844739', '157171587018272768', '157584455735844865', '157595875219226624', '157596567757524992', '157877288594182144', '157893631573893122', '157896126568206336', '157990660237033472', '158590630942089217', '158597079294423040', '158600049067163649', '158608245836091392', '158609387332710400']

        # Tweets pro Zeile herunterladen und als Liste in neuer Spalte speichern
        full_tweets = features.apply(self._download_tweets, axis=1)
        features = features.assign(tweets=full_tweets.values)

        # Alle Zeilen mit leeren Tweet-Listen löschen (z.B. weil Profil gelöscht wurde)
        old_len = len(features)
        # features.to_csv("oldfeatures.tsv", sep='\t')
        features = features[features['tweets'].map(lambda d: len(d)) > 0]
        features.reset_index(drop=True, inplace=True)
        # features.to_csv("newfeatures.tsv", sep='\t')
        if len(features) < old_len:
            logger.warning(f"{(old_len-len(features))} nicht verfügbare User gelöscht (jetzt noch {len(features)} Zeilen)")

        print(features.head())

        # Metadaten aus Userprofil und Tweets ziehen, 12 neue Spalten erstellen
        twitter_stats = features.apply(self._get_twitter_statistics, axis=1)
        twitter_stats_df = pd.DataFrame(list(twitter_stats), columns=twitter_stats[0]._fields)
        features = pd.concat([features, twitter_stats_df], axis=1)

        print(features.head())

        # Linguistische Features bestimmen
        ling_features = features.apply(self._get_linguistic_features, axis=1)
        ling_stats_df = pd.DataFrame(list(ling_features), columns=ling_features[0]._fields)
        features = pd.concat([features, ling_stats_df], axis=1)

        # TODO: private accs handlen!
        logger.info(f"Ende Feature-Extraktion: DF hat {len(features.columns)} Spalten")
        return features

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
        features = df.drop(columns=['index', 'user_id', 'tweet_ids', 'tweets', 'description'])
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

        return agg_features

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

    def predict(self, data, features):
        '''
        '''

        logger.info(f"Beginn Vorhersage ({len(data)} Instanzen)")
        # DF für die Differenzen erstellen, zunächst ohne Spalten
        # Finale Dimensionen: |Instanzen| * |Klassen|
        # Eine Spalte enthält jeweils die Differenzen der Instanzen zu dieser Gold-Klasse
        differences = pd.DataFrame(0, index=range(len(data)), columns=[])

        # Verschachtelte Liste mit allen Features pro Klasse erstellen
        gold = features.values.tolist()
        # Über alle Gold-Klassen iterieren
        for g in gold:
            logger.debug(f"Differenz zu Klasse {g} berechnen")
            # Absolute Differenz zwischen dieser Gold-Klasse und den Daten berechnen
            # Erste Spalte ignorieren, da sie den String mit dem Typ enthält
            # *1, um Rechnung mit Bools möglich zu machen
            # TODO: richtig slicen
            print(data, g)
            diff = abs(data.iloc[:,1:]*1 - g[1:]*1)
            # Differenzen über alle Spalten aufsummieren und in jeweiliger Spalte speichern
            differences[g[0]] = diff.sum(axis=1)

        # Für jede Spalte Klasse mit geringster Differenz/Fehler finden = Vorhersage
        preds = pd.DataFrame(differences.idxmin(axis=1), columns=['prediction'])
        # Zusätzlich den Fehler speichern
        preds['error'] = differences.min(axis=1)
        logger.info(f"Ende Vorhersage ({len(data)} Instanzen)")
        return preds

    def output_prediction(self):
        pass

    def validate(self):
        '''
        Validiert den Klassifikator auf dem Datenset für die Validierung.
        '''

        pass

    def evaluate(self, gold, features):
        '''
        Testet den Klassifikator auf den Testdaten.
        '''

        logger.info(f"Beginn Evaluierung ({len(gold)} Test-Instanzen, {len(features)} Klassen)")
        # Testdaten in Feature-Repräsentation umwandeln
        gold_features = self.extract_features(gold)
        print(gold_features)
        # Vorhersagen für Gold-Daten erhalten
        # preds = self.predict(gold, features)
        # print(preds)
        # Accuracy berechnen
        # ...
        accuracy = 0.0
        logger.info(f"Ende Evaluierung (Accuracy: {accuracy}")

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    f = 'C:/Users/Natze/Documents/Uni/Computerlinguistik/6.Semester/MBTIClassification/data/TwiSty-DE.json'
    # Leerzeilen in logfile einfügen
    logger.info('\n\n')
    clf = MBTIClassifier(f, 'test.tsv', train=True)
