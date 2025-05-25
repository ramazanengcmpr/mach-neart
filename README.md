# mach-neart
makıneılesanat

Bu Python uygulaması, gerçek zamanlı kamera görüntüsüne:
Neural Style Transfer (NST) ile sanatsal bir stil uygulamak
DeepDream ile soyut ve hayal benzeri bir efekt eklemek
görevlerini yerine getiriyor.
Ayrıca kullanıcıdan bir stil görseli seçiliyor ve o görsele de DeepDream uygulanıyor. Sonuçta 3 görüntü yan yana sunuluyor:
NST (kamera + stil)
Kamera + DeepDream
Stil görseli + DeepDream

 KULLANILAN MODELLER
1- Neural Style Transfer (NST)
Kaynak: TensorFlow Hub
Model: arbitrary-image-stylization-v1-256
Amaç: Kamera görüntüsünü kullanıcı tarafından seçilen bir stil görseli ile birleştirerek sanatsal bir görünüm elde etmek.
2-DeepDream (InceptionV3 Temelli)
Model: InceptionV3 (Imagenet ağırlıklı)
Katmanlar: mixed3 ve mixed5
Amaç: Görselin aktivasyon haritalarını kuvvetlendirerek soyut ve rüya benzeri bir efekt oluşturmak.
q: Uygulamayı kapatır.
s: O anki çıktıyı outputs/ klasörüne .jpg olarak kaydeder.
r: Yeni stil seçmek için döngüyü baştan başlatır.









Bileşen	Açıklama
NST Modeli	:TensorFlow Hub'dan yüklenen pre-trained model ile kamera görüntüsüne stil transferi yapılır
DeepDream Modeli	:InceptionV3'in ara katmanlarından aktivasyonlar kullanılarak "rüya" efektleri eklenir
Kamera	:OpenCV ile açılır ve görüntü işleme yapılır
GUI:	Tkinter ile stil görseli seçimi yapılır
Kaydetme:	Tuşla kayıt yapılır, kullanıcı dostu klasör sistemiyle organize edilir
