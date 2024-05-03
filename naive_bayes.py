import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

# Kelimelerin köklerini belirlemek için gereken fonksiyon
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

# İşlenmiş verileri içeren CSV dosyasını oku
data = pd.read_csv(r"C:\Users\merye\OneDrive\Masaüstü\derin öğrenme\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Duygu değerlerini hesaplamak için gereken fonksiyon
def sentiment_score(rating):
    if rating > 3:
        return 1  # Olumlu
    elif rating < 3:
        return -1  # Olumsuz
    else:
        return 0  # Tarafsız

# İşlenmiş yorumları saklamak için bir liste oluştur
processed_reviews = []

# Stopword'lerin listesini oluştur
stop_words = set(stopwords.words('english'))

# Lemmatizer oluştur
lemmatizer = WordNetLemmatizer()

# Yorumları önişleme
for index, row in data.iterrows():
    if isinstance(row['reviews.text'], str):  # Metin verisi olduğundan emin ol
        review = row['reviews.text']
        # Belirteçleme
        tokens = nltk.word_tokenize(review)
        # Kök belirleme ve etiketleme (lemmatization ve pos tagging)
        tagged_tokens = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens if get_wordnet_pos(tag)]
        # Stopword'leri çıkar
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
        # Tüm harfleri küçük harfe dönüştür
        lowercased_tokens = [word.lower() for word in filtered_tokens]
        processed_reviews.append(" ".join(lowercased_tokens))

# İşlenmiş metinleri 'processed_text' adında bir sütun olarak DataFrame'e ekle
data['processed_text'] = processed_reviews

# Duygu sütununu oluştur
data['sentiment'] = data['reviews.rating'].apply(sentiment_score)

# İşlenmiş verileri yeni bir CSV dosyasına kaydet
data.to_csv('processed_data_with_sentiment.csv', index=False)

# Özellik ve hedef sütunları seç
X = data['processed_text']  # İşlenmiş metin sütunu
y = data['sentiment']        # Duygu sütunu

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF dönüşümü
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Naive Bayes modeli oluşturma ve eğitme
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Test seti üzerinde tahmin yapma
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Doğruluk ve sınıflandırma raporu - Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Doğruluk:", accuracy_nb)
# print("Naive Bayes Sınıflandırma Raporu:")
# print(classification_report(y_test, y_pred_nb))

# Destek Vektör Makineleri modeli oluşturma ve eğitme
svm_classifier = SVC()
svm_classifier.fit(X_train_tfidf, y_train)

# Test seti üzerinde tahmin yapma - Destek Vektör Makineleri
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Doğruluk ve sınıflandırma raporu - Destek Vektör Makineleri
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Destek Vektör Makineleri Doğruluk:", accuracy_svm)
# print("Destek Vektör Makineleri Sınıflandırma Raporu:")
# print(classification_report(y_test, y_pred_svm))

#Karar Ağaçları modeli oluşturma ve eğitme
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_tfidf, y_train)

#Test seti üzerinde tahmin yapma - Karar Ağaçları
y_pred_dt = dt_classifier.predict(X_test_tfidf)

# Doğruluk ve sınıflandırma raporu - Karar Ağaçları
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Karar Ağaçları Doğruluk:", accuracy_dt)
# print("Karar Ağaçları Sınıflandırma Raporu:")
# print(classification_report(y_test, y_pred_dt))

#Lojistik Regresyon modeli oluşturma ve eğitme
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)

# Test seti üzerinde tahmin yapma - Lojistik Regresyon
y_pred_lr = lr_classifier.predict(X_test_tfidf)

# Doğruluk ve sınıflandırma raporu - Lojistik Regresyon
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Lojistik Regresyon Doğruluk:", accuracy_lr)
# print("Lojistik Regresyon Sınıflandırma Raporu:")
# print(classification_report(y_test, y_pred_lr))
