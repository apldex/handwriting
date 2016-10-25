Handwriting Recognition Project based on DL4J API
=========================
Repositori ini berisi source code untuk pengenal tulisan tangan
dengan model pengenal Convolutional Neural Net. Bahasa yang digunakan adalah Java

Arsitektur ConvNet dibangun menggunakan API DeepLearning4Java.

---

## Build and Run

Untuk menjalankan project ini, terlebih dahulu download dan install [Maven](https://maven.apache.org/), [IntelliJ] (http://jetbrains.com/) dan Java. 

Pastikan versi Java yang digunakan adalah yang paling baru dengan perintah berikut
```
java -version
```

Pastikan versi Maven yang digunakan adalah yang paling baru dengan perintah berikut
```
mvn --version
```

Untuk mac, install Maven dengan perintah berikut
```
brew install maven
```


Selanjutnya, import project ini ke IntelliJ dengan cara File -> New -> Project from Existing Sources.
Pada window selanjutnya pilih Import project from external model dan pastikan Maven yang terpilih.

## Catatan

Pada file ConvNet.java, ada beberapa hal yang harus diperhatikan. Pastikan line berikut sesuai dengan direktori tempat penyimpanan dataset
```
File parentDir = new File("/Volumes/Data/adrian/Datasets/Capt/");
```
Line berikut juga harus diperhatikan untuk tempat penyimpanan hasil training. Sesuaikan dengan direktori pada penyimpanan lokal
```
File file = new File("/Volumes/Data/HandwritingProject/Train Result .txt");
```

