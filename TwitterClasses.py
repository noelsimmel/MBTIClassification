# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality 
# Datei: TwitterClasses.py – Enthält 2 Klassen zur Darstellungen von Tweets und Usern
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

import logging

# Logging-Details
twitter_logger = logging.getLogger(__name__)
twitter_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
file_handler = logging.FileHandler('classifier.log')
file_handler.setFormatter(formatter)
twitter_logger.addHandler(file_handler)

class Tweet:
    '''
    Tweet-Klasse. 
    Extrahiert die für die Klassifikation notwendigen Merkmale aus dem status-Objekt 
    der Twitter API. 
    '''

    def __init__(self, status):
        self.id = None
        self.text = None
        self.hashtags_count = self.mentions_count = 0
        self.fav_count = self.rt_count = 0
        self.has_media = self.has_url = self.is_reply = False
        
        try:
            self._instantiate(status)
        except Exception as e:
            twitter_logger.error(f"TWITTER TWEET: {str(e)}")

    def __str__(self):
        return self.text

    def __len__(self):
        try:
            return len(self.text)
        except TypeError:
            return 0

    def __contains__(self, query):
        try: 
            return query in self.text
        except TypeError:
            return False

    def _instantiate(self, status):
        '''
        Befüllt die Instanzvariablen.
        '''

        self.id = status.id
        self.text = status.text
        self.hashtags_count = len(status.entities['hashtags'])
        self.mentions_count = len(status.entities['user_mentions'])
        self.fav_count = status.favorite_count
        self.rt_count = status.retweet_count
        if 'media' in status.entities:
            self.has_media = True
        # URL ist binäres Feature, da die wenigstens Tweets >1 URL enthalten
        if len(status.entities['urls']) > 0:
            self.has_url = True
        # is_reply gibt an, ob der Tweet eine Antwort auf einen anderen Tweet ist
        # Die Anzahl der Antworten auf einen eigenen Tweet kann mit dem Standard-
        # Zugang zur Twitter-API leider nicht abgefragt werden
        if status.in_reply_to_status_id:
            self.is_reply = True


class User:
    '''
    User-Klasse. 
    Extrahiert die für die Klassifikation notwendigen Merkmale aus dem user-Objekt 
    der Twitter API. 
    '''

    def __init__(self, user):
        self.id = None
        self.description = ""
        # followers_friends_ratio = |Followers|/|Friends|
        # friends sind Accounts, denen ein User folgt. Wenn 0, setze ratio auf |Followers|
        self.followers_friends_ratio = 0.0
        # has_profile_url = Ob im Profil eine URL angegeben wurde
        self.is_verified = self.has_profile_url = False
        
        try:
            self._instantiate(user)
        except Exception as e:
            twitter_logger.error(f"TWITTER USER: {str(e)}")

    def __str__(self):
        return str(self.id)

    def _instantiate(self, user):
        '''
        Befüllt die Instanzvariablen.
        '''

        self.id = user.id
        if user.description:
            self.description = user.description
        if user.friends_count == 0:
            self.followers_friends_ratio = user.followers_count
        else:
            self.followers_friends_ratio = user.followers_count/user.friends_count
        self.is_verified = user.verified
        if user.url:
            self.has_profile_url = True
