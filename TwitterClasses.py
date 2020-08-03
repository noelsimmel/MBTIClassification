# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality 
# Datei: TwitterClasses.py – Enthält 2 Klassen zur Darstellungen von Tweets und Usern
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

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
        self.has_media = self.has_url = self.is_reply = self.is_retweet = False
        
        self._instantiate(status)

    def __str__(self):
        return self.text

    def __len__(self):
        try:
            return len(self.text)
        except TypeError:
            return 0

    def __contains__(self, query):
        return query in self.text

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
        # Speziell für Retweets
        if 'retweeted_status' in status._json:
            self.is_retweet = True
            # Text wird durch Voranstellen des Usernamen evtl. abgeschnitten
            # Deshalb auf Attribut full_text zurückgreifen
            self.text = status._json['retweeted_status']['full_text']
            # Likes und RTs eines Retweets ignorieren
            self.fav_count = 0
            self.rt_count = 0


class User:
    '''
    User-Klasse. 
    Extrahiert die für die Klassifikation notwendigen Merkmale aus dem user-Objekt 
    der Twitter API. 
    '''

    def __init__(self, user):
        self.id = None
        self.description = ""
        # friends_count = Anzahl an Accounts, denen die Person folgt
        # statuses_count = Anzahl an Tweets (ein Tweet wird intern "status" genannt)
        self.followers_count = self.friends_count = self.fav_count = self.statuses_count = 0
        # has_profile_url = Ob im Profil eine URL angegeben wurde
        self.is_verified = self.has_profile_url = False
        
        self._instantiate(user)

    def __str__(self):
        return str(self.id)

    def _instantiate(self, user):
        '''
        Befüllt die Instanzvariablen.
        '''

        self.id = user.id
        if user.description:
            self.description = user.description
        self.followers_count = user.followers_count
        self.friends_count = user.friends_count
        self.fav_count = user.favourites_count
        self.statuses_count = user.statuses_count
        self.is_verified = user.verified
        if user.url:
            self.has_profile_url = True
