import os
import numpy as np

negatif_yorumlar_path = "/content/drive/MyDrive/yeniolumsuz.txt"#verileri yükleme
pozitif_yorumlar_path = "/content/drive/MyDrive/yeni_pozitif_yorumlar.txt"

with open(negatif_yorumlar_path, "r", encoding="utf-8") as file:
    negatif_yorumlar = file.readlines()
with open(pozitif_yorumlar_path, "r", encoding="utf-8") as file:
    pozitif_yorumlar = file.readlines()

X_negatif = np.array(negatif_yorumlar)
X_pozitif = np.array(pozitif_yorumlar)

# Eğitim ve test verilerinin yüzde kaçını alacağımızı belirleyelim
test_orani = 0.2

# Negatif yorumlar için eğitim ve test verilerini ayırma
negatif_test_sayisi = int(len(X_negatif) * test_orani)
negatif_egitim_sayisi = len(X_negatif) - negatif_test_sayisi

# Pozitif yorumlar için eğitim ve test verilerini ayırma
pozitif_test_sayisi = int(len(X_pozitif) * test_orani)
pozitif_egitim_sayisi = len(X_pozitif) - pozitif_test_sayisi

# Eğitim ve test verilerini belirli oranda alarak klasörlere yazma işlemi
for klasor_index in range(5):
    print(f"Klasör {klasor_index + 1} için k-fold cross validation işlemi başlıyor...")

    # Klasörü oluştur
    klasor_adi = f"/content/drive/MyDrive/makineödev2/klasor_{klasor_index+1}"
    if not os.path.exists(klasor_adi):
        os.makedirs(klasor_adi)

    # Eğitim ve test verilerini ayır
    negatif_baslangic_index = klasor_index * negatif_test_sayisi
    negatif_bitis_index = negatif_baslangic_index + negatif_test_sayisi
    pozitif_baslangic_index = klasor_index * pozitif_test_sayisi
    pozitif_bitis_index = pozitif_baslangic_index + pozitif_test_sayisi

    test_X_negatif = X_negatif[negatif_baslangic_index:negatif_bitis_index]
    test_X_pozitif = X_pozitif[pozitif_baslangic_index:pozitif_bitis_index]
    test_X = np.concatenate((test_X_negatif, test_X_pozitif))

    egitim_X_negatif = np.concatenate((X_negatif[:negatif_baslangic_index], X_negatif[negatif_bitis_index:]))
    egitim_X_pozitif = np.concatenate((X_pozitif[:pozitif_baslangic_index], X_pozitif[pozitif_bitis_index:]))
    egitim_X = np.concatenate((egitim_X_negatif, egitim_X_pozitif))

    # Dosyalara yazma işlemi
    with open(f"{klasor_adi}/egitim_negatif.txt", "w", encoding="utf-8") as file:
        file.writelines(egitim_X_negatif)
    with open(f"{klasor_adi}/test_negatif.txt", "w", encoding="utf-8") as file:
        file.writelines(test_X_negatif)
    with open(f"{klasor_adi}/egitim_pozitif.txt", "w", encoding="utf-8") as file:
        file.writelines(egitim_X_pozitif)
    with open(f"{klasor_adi}/test_pozitif.txt", "w", encoding="utf-8") as file:
        file.writelines(test_X_pozitif)

    print(f"Klasör {klasor_index + 1} için eğitim ve test verileri dosyalara yazıldı.")
    
    
    
    
import os
import numpy as np
import pandas as pd

klasor_sayisi = 5

# Train verileri için TF matrisini oluşturun ve train.csv dosyalarını oluşturun
for klasor_index in range(1, klasor_sayisi+1):
    X_train_tum = []

    train_pozitif_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/egitim_pozitif.txt"
    train_negatif_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/egitim_negatif.txt"

    with open(train_pozitif_path, "r", encoding="utf-8") as file:
        train_pozitif = file.readlines()
    with open(train_negatif_path, "r", encoding="utf-8") as file:
        train_negatif = file.readlines()

    X_train_tum.extend(train_pozitif)
    X_train_tum.extend(train_negatif)

    train_tf_matrix = np.zeros((len(X_train_tum), len(features)))

    for i, kelime_satiri in enumerate(X_train_tum):
        kelimeler = kelime_satiri.split()
        for kelime in kelimeler:
            if kelime in features:
                kelime_index = features.index(kelime)
                train_tf_matrix[i, kelime_index] += 1

    # Eğitim verileriyle hesaplanan IDF vektörünü train verileriyle kullanarak TF-IDF matrisini oluşturun
    train_tfidf_matrix = train_tf_matrix * training_idf_vector

    # Train verilerinin sınıf etiketlerini oluştur
    train_classes = [1] * len(train_pozitif) + [0] * len(train_negatif) # Bu kısmı train verilerinizin etiketlerine göre düzenleyin.
    train_classes_array = np.array(train_classes)

    # TF-IDF matrisini ve sınıf etiketlerini birleştirerek bir DataFrame oluştur
    train_data = np.hstack((train_tfidf_matrix, train_classes_array.reshape(-1, 1)))
    train_df = pd.DataFrame(train_data, columns=features + ['sinif_label'])

    # Train.csv dosyasını oluşturulan dizine kaydedin
    output_directory = f'/content/drive/MyDrive/makineproje/klasor_{klasor_index}'
    os.makedirs(output_directory, exist_ok=True)
    train_df.to_csv(f'{output_directory}/tfidf_train.csv', index=False)

# Test verileri için TF matrisini oluşturun ve test.csv dosyalarını oluşturun
for klasor_index in range(1, klasor_sayisi+1):
    X_test_tum = []

    test_pozitif_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/test_pozitif.txt"
    test_negatif_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/test_negatif.txt"

    with open(test_pozitif_path, "r", encoding="utf-8") as file:
        test_pozitif = file.readlines()
    with open(test_negatif_path, "r", encoding="utf-8") as file:
        test_negatif = file.readlines()

    X_test_tum.extend(test_pozitif)
    X_test_tum.extend(test_negatif)

    test_tf_matrix = np.zeros((len(X_test_tum), len(features)))

    for i, kelime_satiri in enumerate(X_test_tum):
        kelimeler = kelime_satiri.split()
        for kelime in kelimeler:
            if kelime in features:
                kelime_index = features.index(kelime)
                test_tf_matrix[i, kelime_index] += 1

    # Eğitim verileriyle hesaplanan IDF vektörünü test verileriyle kullanarak TF-IDF matrisini oluşturun
    test_tfidf_matrix = test_tf_matrix * training_idf_vector

    # Test verilerinin sınıf etiketlerini oluştur
    test_classes = [1] * len(test_pozitif) + [0] * len(test_negatif) # Bu kısmı test verilerinizin etiketlerine göre düzenleyin.
    test_classes_array = np.array(test_classes)

    # TF-IDF matrisini ve sınıf etiketlerini birleştirerek bir DataFrame oluştur
    test_data = np.hstack((test_tfidf_matrix, test_classes_array.reshape(-1, 1)))
    test_df = pd.DataFrame(test_data, columns=features + ['sinif_label'])

    # Test.csv dosyasını oluşturulan dizine kaydedin
    output_directory = f'/content/drive/MyDrive/makineproje/klasor_{klasor_index}'
    os.makedirs(output_directory, exist_ok=True)
    test_df.to_csv(f'{output_directory}/tfidf_test.csv', index=False)

    klasor_sayisi = 5
    accuracies = []
    f1_scores = []

    # Train ve test verilerini yüklemek için boş listeler oluştur
    train_datas = []
    test_datas = []

    # Her bir klasör için
    for klasor_index in range(1, klasor_sayisi+1):
        print(f"Klasör {klasor_index} için işlem başlatılıyor...")

        # Eğitim verilerini yükle
        train_data_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/tfidf_train.csv"
        test_data_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/tfidf_test.csv"

        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
        except FileNotFoundError as e:
            print(f"Dosya bulunamadı: {e.filename}")
            continue

        # Train ve test verilerini listeye ekle
        train_datas.append(train_data)
        test_datas.append(test_data)

    # Train ve test verilerini bir kere yükledikten sonra işlemleri gerçekleştir
    for klasor_index, (train_data, test_data) in enumerate(zip(train_datas, test_datas), 1):
        print(f"Klasör {klasor_index} için işlem başlatılıyor...")

        X_train = train_data.drop('sinif_label', axis=1)
        y_train = train_data['sinif_label']
        X_test = test_data.drop('sinif_label', axis=1)
        y_test = test_data['sinif_label']

        # Hedef değişkeni integer olarak ayarla
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')

        # Random Forest modelini oluştur ve eğit
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Modeli test verileriyle değerlendir
        y_pred = model.predict(X_test)

        # Accuracy ve F1-score hesapla
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Sonuçları yazdır
        print(f"Klasör {klasor_index} için Accuracy: {accuracy}, F1-score: {f1}")

        # Sonuçları listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Klasör {klasor_index} için işlemler tamamlandı.")

    # Genel accuracy ve F1-score hesapla
    if accuracies and f1_scores:
        average_accuracy = sum(accuracies) / len(accuracies)
        average_f1 = sum(f1_scores) / len(f1_scores)

        print(f"Genel Accuracy: {average_accuracy}, Genel F1-score: {average_f1}")
    else:
        print("Hiçbir klasör için sonuç hesaplanamadı.")


import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer

# Veriyi yükleme
klasor_sayisi = 5

for klasor_index in range(1, klasor_sayisi+1):
    file_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/tfidf_train.csv"
    data = pd.read_csv(file_path)

    # Özellikleri (X) ve hedefi (y) ayırma
    X = data.drop('sinif_label', axis=1)
    y = data['sinif_label'].replace({1.0: 1, 0.0: 0})

    # Eksik değerleri ortalama ile doldurma
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Mutual Information kullanarak öznitelik seçimi
    def select_features(X, y, k):
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        cols = selector.get_support(indices=True)
        X_new = X.columns[cols]
        return X_new

    feature_counts = [250, 500, 1000, 2500, 5000]
    for count in feature_counts:
        X_new = select_features(X_imputed, y, count)
        output_path = f"/content/drive/MyDrive/makineproje/klasor_{klasor_index}/tfidf_train_{count}.txt"
        with open(output_path, 'w') as file:
            for feature in X_new:
                file.write(f"{feature}\n")

import os
import numpy as np
import pandas as pd

num_folders = 5
for folder_num in range(1, num_folders + 1):
    all_training_docs = []
    feature_set = []

    pos_train_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/egitim_pozitif.txt"
    neg_train_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/egitim_negatif.txt"

    with open(pos_train_path, "r", encoding="utf-8") as file:
        pos_train_docs = file.readlines()
    with open(neg_train_path, "r", encoding="utf-8") as file:
        neg_train_docs = file.readlines()

    folder_training_docs = pos_train_docs + neg_train_docs
    all_training_docs.extend(folder_training_docs)

    feature_counts = [250, 500, 1000, 2500, 5000]
    for count in feature_counts:
        tfidf_features_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/tfidf_train_{count}.txt"
        with open(tfidf_features_path, "r", encoding="utf-8") as file:
            selected_features = file.readlines()

        for word in selected_features:
            word = word.strip()
            if word not in feature_set:
                feature_set.append(word)
        print(list(feature_set))

        df_vectör = np.zeros(len(feature_set))
        for doc in folder_training_docs:
            words = doc.split()
            for word in feature_set:
                if word in words:
                    df_vectör[feature_set.index(word)] += 1
        print(df_vectör)

        total_docs = len(all_training_docs)
        print(total_docs)

        num_features = len(feature_set)
        print(num_features)

        idf_vector = np.zeros(len(feature_set))
        for i in range(len(feature_set)):
            idf_vector[i] = np.log10(total_docs / df_vectör[i])
        print(idf_vector)

        tf_matrix = np.zeros((len(all_training_docs), len(feature_set)))

        for i, doc in enumerate(folder_training_docs):
            words = doc.split()
            for word in words:
                word = word.strip()
                if word in feature_set:
                    word_index = feature_set.index(word)
                    tf_matrix[i, word_index] += 1
        print(tf_matrix)

        tfidf_matrix = tf_matrix * idf_vector
        print(tfidf_matrix)

        class_labels = [1] * len(pos_train_docs) + [0] * len(neg_train_docs)
        class_labels_array = np.array(class_labels)
        combined_data = np.hstack((tfidf_matrix, class_labels_array.reshape(-1, 1)))
        df = pd.DataFrame(combined_data, columns=list(feature_set) + ['sinif_label'])
        output_dir = f'/content/drive/MyDrive/makineproje/klasor_{folder_num}'
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f'{output_dir}/tfidf_train_{count}.csv', index=False)


import os
import numpy as np
import pandas as pd

num_folders = 5

def read_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()

def compute_tfidf(documents, feature_set):
    num_docs = len(documents)
    feature_count = len(feature_set)

    df = np.zeros(feature_count)
    for idx, feature in enumerate(feature_set):
        df[idx] = sum(1 for doc in documents if feature in doc.split())

    idf = np.log10(num_docs / (df + 1))

    tf_matrix = np.zeros((num_docs, feature_count))
    for doc_idx, doc in enumerate(documents):
        words = doc.split()
        for word in words:
            if word in feature_set:
                word_idx = feature_set.index(word)
                tf_matrix[doc_idx, word_idx] += 1

    tfidf_matrix = tf_matrix * idf
    return tfidf_matrix

for folder_num in range(1, num_folders + 1):
    training_docs_all = []
    features_set = []

    pos_train_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/egitim_pozitif.txt"
    neg_train_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/egitim_negatif.txt"

    pos_train_docs = read_lines(pos_train_path)
    neg_train_docs = read_lines(neg_train_path)

    training_docs = pos_train_docs + neg_train_docs
    training_docs_all.extend(training_docs)

    feature_sizes = [250, 500, 1000, 2500, 5000]
    for size in feature_sizes:
        tfidf_feature_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/tfidf_train_{size}.txt"
        tfidf_features = [line.strip() for line in read_lines(tfidf_feature_path)]

        for word in tfidf_features:
            if word not in features_set:
                features_set.append(word)

        print(f"Selected features: {list(features_set)}")

        tfidf_train_matrix = compute_tfidf(training_docs, features_set)

        pos_test_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/test_pozitif.txt"
        neg_test_path = f"/content/drive/MyDrive/makineproje/klasor_{folder_num}/test_negatif.txt"

        pos_test_docs = read_lines(pos_test_path)
        neg_test_docs = read_lines(neg_test_path)

        test_docs = pos_test_docs + neg_test_docs

        tfidf_test_matrix = compute_tfidf(test_docs, features_set)

        test_labels = [1] * len(pos_test_docs) + [0] * len(neg_test_docs)
        test_labels_array = np.array(test_labels)

        test_data = np.hstack((tfidf_test_matrix, test_labels_array.reshape(-1, 1)))
        test_df = pd.DataFrame(test_data, columns=list(features_set) + ['sinif_label'])

        output_dir = f'/content/drive/MyDrive/makineproje/klasor_{folder_num}'
        os.makedirs(output_dir, exist_ok=True)
        test_df.to_csv(f'{output_dir}/tfidf_test_{size}.csv', index=False)
        
        
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import numpy as np

klasor_sayisi = 5
feature_counts = [250, 500, 1000, 2500, 5000]

for count in feature_counts:
    accuracies = []
    f1_scores = []
    for klasor_numarasi in range(1, klasor_sayisi + 1):
        # Eğitim ve test verilerini yükleyin
        train_data = pd.read_csv(f"/content/drive/MyDrive/makineproje/klasor_{klasor_numarasi}/tfidf_train_{count}.csv")
        test_data = pd.read_csv(f"/content/drive/MyDrive/makineproje/klasor_{klasor_numarasi}/tfidf_test_{count}.csv")

        # Eğitim ve test verilerindeki eksik değerleri ortalama ile doldurmak için SimpleImputer kullanın
        imputer = SimpleImputer(strategy='mean')
        X_train = train_data.drop(columns="sinif_label")
        X_train = imputer.fit_transform(X_train)
        y_train = train_data["sinif_label"].replace({1.0: 1, 0.0: 0}).astype(int)

        X_test = test_data.drop(columns="sinif_label")
        X_test = imputer.transform(X_test)
        y_test = test_data["sinif_label"].replace({1.0: 1, 0.0: 0}).astype(int)

        # Verilerin doğru biçimde olduğunu kontrol edin
        assert y_train.dtype == int, "y_train contains non-integer values"
        assert y_test.dtype == int, "y_test contains non-integer values"

        # RandomForestClassifier'ı tanımlama ve eğitme
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Test seti üzerinde tahmin yapma
        y_pred = rf_classifier.predict(X_test)

        # Modelin doğruluğunu hesaplama
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Klasor {klasor_numarasi} için {count} features model doğruluğu:", accuracy)

        # F1-score hesaplama
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        print(f"Klasor {klasor_numarasi} için {count} features model F1-score değeri:", f1)

    # Klasör bazında doğruluk ve F1-score ortalamalarını hesaplama
    klasor_accuracy_avg = np.mean(accuracies)
    klasor_f1_score_avg = np.mean(f1_scores)

    print(f"{count} için tüm klasörlerdeki ortalama doğruluk: {klasor_accuracy_avg}")
    print(f"{count} için tüm klasörlerdeki ortalama F1-score: {klasor_f1_score_avg}")
    