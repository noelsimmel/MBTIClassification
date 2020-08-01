# Modul: PRO2-A, SS 2020
# Projekt: Author profiling – Twitter & MBTI personality types
# Autor*in: Noel Simmel (791794)
# Abgabe: 31.08.20

class Tweet:
    '''
    Tweet-Klasse. 
    Extrahiert die für die Klassifikation notwendigen Merkmale aus dem status-Objekt 
    der Twitter API. 
    '''

    def __init__(self, status):
        '''
        '''

        self.id = None
        self.text = None
        self.hashtags = self.urls = self.mentions = 0
        self.fav_count = self.rt_count = 0
        self.has_media = self.is_reply = self.is_retweet = False
        
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
        self.hashtags = len(status.entities['hashtags'])
        self.urls = len(status.entities['urls'])
        self.mentions = len(status.entities['user_mentions'])
        self.fav_count = status.favorite_count
        self.rt_count = status.retweet_count
        if 'media' in status.entities:
            self.has_media = True
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
