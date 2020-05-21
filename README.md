# Data Preprocessing and Determining Feature Importances of ANY GIVEN Dataset
My program makes automatic data preparation for any given dataset, than generate models using Decision Tree and Random Forest Classifiers and finds out the feature’s importances. And runs on Flask-RESTful API.


## Kodun içerisinde belirtildiği yere dosyanın kendisi verilir. Ardından Veriler otomatik olarak alttaki preprocessing ve uygun model oluşturma süreçlerinden geçerler:
(Bu süreçler kodun içerisinde de başlama bitiş olarak Comment’ler ile belirtilmişlerdir)

- 100% Missing Value sutün varsa düşür.
- Feature Importance işleminde Generic bir yazılım geliştirdiğimden ötürü, default olarak Date&Time sütunları üzerinden işlem yapılmamaktadır. Bu yüzden düşürülürler.- Kullanıcı isteğine göre ayrıca bir serviste Date verisine göre analiz yapılabilecektir.
- 90% dan daha çok “UNIQUE” değer içeren sütunlar analiz yapılacak verilerin arasından düşürülür (örneğin TC, telefon numarası vb.)
- Numeric Sütunlar için Missing Value Handling
- Nominal Sütunlar için Missing Value Handling
- Numeric sütunların önemini araştırmak için içerdikleri “UNIQUE” veri sayılarına göre ayrıştırılırlar ve en doğru şekilde “Binning” işleminden geçirilerek gruplara bölünürler.
- Tüm bu Preprocessing süreçlerinden geçmiş olan sütunların içerisindeki her “UNIQUE” data ayrı bir sütun olarak ayrıştırılır ve yeni sütunlar sadece Binary formatta veri barındırırlar (OneHotEncoding)
- Model oluşturma ve ilişki analizi işleminden önce son kez birkaç döngü ile her şeyin yolunda olduğundan emin olunur.
- Ardından kullanıcının seçtiği algoritmaya bakılır (Decision Tree Classifier || Random Forest Classifier) ve bu algoritmaya göre en optimal model oluşturulur, 
  Oluşturduğum bu model sayesinde: Sisteme girilen veri setindeki tüm sütunların barındırdığı unique değerler arasında bir ilişki yakalayabiliriz.
  
## Ayrıca oluşturduğum bu sistemde, sisteme verilen input ve output’lar tamamen JSON formatındadır. Bu da elde edilen sonuçların görselleştirme ve başka sistemlere entegre edilmesinde öngörülmüş bir unsurdur.



NOT:
- Eğer tüm veri setinde analiz isteniyorsa ->”/UniqueDataImportances” servisi
- Belirlenmiş iki sütun arasındaki verilerin analizi isteniyorsa ->”/DualImportances” servisi kullanılmalıdır.
