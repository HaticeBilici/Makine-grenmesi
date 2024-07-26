import pandas as pd## veri setini import etme
df=pd.read_csv("/content/drive/MyDrive/btkakademi/Parkison_Dataset.csv")
df

print("\nSütun isimleri (liste olarak):")
print(df.columns.tolist())#sutunları listele

print(df.isnull().sum())#eksik deger var mı kontorlü
print(df.info())
print(df.describe())#açıklama


# Aykırı değerleri veri setinden çıkar
X_clean = X[~outliers]
y_clean = y[~outliers]

#SWM SINIFLANDIRMA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Verileri yükle
df = pd.read_csv("/content/drive/MyDrive/btkakademi/Parkison_Dataset.csv")

X = df.drop(columns=['class', 'id'])### class ve id sutununu  çıkar kalan öznitelikler
y = df['class'].values#target


# Verileri sıralıyoruz
indices = np.arange(y.shape[0])#Veri setindeki örnek sayısı kadar sıralı bir dizi oluşturur
rnd = np.random.RandomState(123)#
shuffled_indices = rnd.permutation(indices)#dizisini rastgele karıştırır
X_shuffled, y_shuffled = X.iloc[shuffled_indices], y[shuffled_indices]

# Veriyi ölçekleme
scaler = StandardScaler()#### normalizasyonn minmax daha kötü sonuç verdi
X_scaled = scaler.fit_transform(X_shuffled)#veri ile besle

# SMOTE ile veri dengesi sağlama
smote = SMOTE(random_state=42)#eğitim verilerinde denge sağlama
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_shuffled, test_size=0.2, random_state=142, shuffle=True)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)#SMOTE, azınlık sınıf örneklerinin sayısını artırarak veri dengesizliğini gideri

# Geniş hiperparametre aralığı ile Grid Search cross validation
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)# -1,paralel hesaplama yapar,GridSearchCV, hiperparametre optimizasyonu için kullanılan bir yöntemdi
grid_search.fit(X_train_smote, y_train_smote)## dengelenmiş verileri kullanarak modeli fit eder

print("En iyi hiperparametreler:", grid_search.best_params_)#en iyi hiperparaemtreler
print("En iyi doğruluk:", grid_search.best_score_)# 'grid_search' nesnesinin 'best_score_' özelliğini kullanır ve en iyi doğruluk skorunu yazdırır

# En iyi model ile tahmin yapma
best_model = grid_search.best_estimator_##en iyi modeli döndürür
y_pred = best_model.predict(X_test)#tahmini yazar

# Modelin performansını değerlendirme
print("\nEn İyi Model Performansı:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

#KNN SINIFLANDIRMA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Verileri yükle
df = pd.read_csv("/content/drive/MyDrive/btkakademi/Parkison_Dataset.csv")
df.head()  # Verileri göster

X = df.drop(columns=['class', 'id'])
y = df['class'].values


# Verileri sıralıyoruz
indices = np.arange(y.shape[0])
rnd = np.random.RandomState(123)
shuffled_indices = rnd.permutation(indices)
X_shuffled, y_shuffled = X.iloc[shuffled_indices], y[shuffled_indices]

# Özellikleri ölçeklendir normalize et
scaler = MinMaxScaler()# standart scaler da oran %91 verdi o yüzden bu daha iyi
X_scaled = scaler.fit_transform(X_shuffled)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_shuffled, test_size=0.2, random_state=142, shuffle=True)

# GridSearchCV ile en iyi K değerini bulma cross validaiton işlemi yapıldı
param_grid = {'n_neighbors': list(range(1, 31))}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi K değerini al
best_k = grid_search.best_params_['n_neighbors']
print(f'En iyi K değeri: {best_k}')

# En iyi hiperparametrelerle modeli oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=best_k)#
knn.fit(X_train, y_train)

# Test setinde tahmin yap
y_pred = knn.predict(X_test)

# Performansı değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f'Doğruluk: {accuracy:.2f}')

# Karışıklık matrisini oluşturma ve görselleştirme
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Detaylı performans raporu
print(classification_report(y_test, y_pred))
#NAİVE BAYES SINIFLANDIRMA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Veriyi yükleme ve ilk 5 satırı gösterme
df = pd.read_csv("/content/drive/MyDrive/btkakademi/Parkison_Dataset.csv")
print("Sütun adları:", df.columns)
print(df.head())  # Verileri göster

# Veriyi ve etiketleri ayırma
X = df.drop(columns=['class', 'id'])
y = df['class'].values

# Verileri sıralıyoruz
indices = np.arange(y.shape[0])
rnd = np.random.RandomState(123)
shuffled_indices = rnd.permutation(indices)
X_shuffled, y_shuffled = X.iloc[shuffled_indices], y[shuffled_indices]

# Veriyi ölçekleme
scaler = MinMaxScaler()  # 0 ile 1 arasında ölçeklendirerek negatif değerlerin oluşmasını engeller ,çünkü MultinomialNB negatif değerlerle çalışmaz
X_scaled = scaler.fit_transform(X_shuffled)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_shuffled, test_size=0.2, random_state=142, shuffle=True)

# Naive Bayes modelini eğitme ve hiperparametre ayarlama
nb_clf = MultinomialNB()
parameters = {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],'fit_prior': [True, False]}
grid_search = GridSearchCV(nb_clf, parameters, cv=5)
grid_search.fit(X_train, y_train)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi skor: {grid_search.best_score_}")

best_nb_clf = grid_search.best_estimator_
y_pred = best_nb_clf.predict(X_test)

# Modelin performansını değerlendirme
print(classification_report(y_test, y_pred))

# Confusion Matrix oluşturma ve gösterme
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
cm_display.plot(cmap=plt.cm.Blues)  # 'cmap' parametresi ile renk haritası belirleyebilirsiniz
plt.show()
#LOGİSTİC REGRESSYON 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Verileri yükle
df = pd.read_csv("/content/drive/MyDrive/btkakademi/Parkison_Dataset.csv")

# Özellikleri ve etiketi ayırma
X = df.drop(columns=['class', 'id'])
y = df['class'].values

# Verileri sıralama
indices = np.arange(y.shape[0])
rnd = np.random.RandomState(123)
shuffled_indices = rnd.permutation(indices)
X_shuffled, y_shuffled = X.iloc[shuffled_indices], y[shuffled_indices]

# Veriyi ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_shuffled)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_shuffled, test_size=0.2, random_state=142, shuffle=True)

# Grid Search ile en iyi hiperparametreleri bulma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("En iyi hiperparametreler:", grid_search.best_params_)
print("En iyi doğruluk:", grid_search.best_score_)

# En iyi model ile tahmin yapma
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Modelin performansını değerlendirme
print("\nEn İyi Model Performansı:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
cm_display.plot(cmap=plt.cm.Blues)
plt.show()
