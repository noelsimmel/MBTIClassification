# Projektanforderungen To-Do
Thema: Author profiling 

## Aufbau
- [ ] MEW mit allen Funktionen erstellen
- [ ] Docstrings
- [ ] Logging
- [ ] Optimierung
- [ ] Unit-Tests

## Daten
- [ ] Daten downloaden und vorverarbeiten
- [ ] Split in Training-/Validierung-/Test 70:10:20
- [ ] Daten in passende Datenstruktur einlesen
- [ ] Natural language processing (NLTK/spaCy)
  - [ ] Satzgrenzen finden
  - [ ] Tokenisierung
  - [ ] POS-Tagging
  - [ ] Lemmatisierung
  - [ ] (NER)
  - [ ] (Sentimentanalyse)

## Statistiken
- [ ] Statistiken berechnen und normalisieren
  - [ ] Zeichen
    - [ ] Anzahl Zeichen
    - [ ] Anzahl Buchstaben
    - [ ] Anzahl Groß- und Kleinbuchstaben
    - [ ] Anzahl Zahlen
    - [ ] Anzahl Whitespace-Zeichen
    - [ ] Anzahl Sonderzeichen
  - [ ] Interpunktion
    - [ ] Anzahl Kommata, Punkte, Semikolons
    - [ ] Anzahl Frage- und Ausrufezeichen
  - [ ] Wörter
    - [ ] Anzahl Wörter
    - [ ] Durchschnittliche Wortlänge in Zeichen
    - [ ] Anzahl lange und kurze Wörter
    - [ ] Anzahl Emoticons
    - [ ] Anzahl Emoji
    - [ ] Anzahl Rechtschreibfehler
    - [ ] Type-token-Verhältnis
    - [ ] Hapax-legomena-token-Verhältnis
    - [ ] Häufigkeit der häufigsten Wörter (?)
  - [ ] Sätze
    - [ ] Anzahl Sätze
    - [ ] Durchschnittliche Satzlänge in Zeichen und Wörtern
  - [ ] Syntax
    - [ ] Anzahl POS-Tags
  - [ ] Semantik
    - [ ] Sentiment-Score
    - [ ] Anzahl positive und negative Wörter
    - [ ] Anzahl named entities
    - [ ] Anzahl @-Erwähnungen
    - [ ] Anzahl Hashtags
  - [ ] Twitter-spezifisch
    - [ ] Durchschnittliche Anzahl Favs und Retweets
    - [ ] Anzahl Follower
    - [ ] Anzahl Gefolgte
- [ ] Statistiken als .csv speichern
  - [ ] Für einzelne Instanzen inkl. Gold-Klasse
  - [ ] Aggregiert für einzelne Klassen/MBTI-Typen
- [ ] Statistiken visualisieren

## Klassifikation
- [ ] Modell trainieren (s. Projekt-PDF) und testen
- [ ] Vorhersagen und Gold-Klasse in separate Datei speichern
- [ ] Modell evaluieren (accuracy)

## Finishing
- [ ] ReadMe mit Quellen zu den Daten
- [ ] Repository aufräumen, gitignore updaten 
- [ ] Code reviewen
- [ ] Code reviewen lassen

# Bepunktung
100 Punkte insgesamt. Bestanden ab 50 Punkten, 80 Punkte = 2,3
- 40 Punkte für Korrektheit & Struktur
- 40 Punkte für Lesbarkeit & Anwendbarkeit
- 10 Punkte für Unit-Tests
- 5 Punkte für Reviewen eines anderen Projekts
- 5 Punkte für Überarbeitung nach Feedback