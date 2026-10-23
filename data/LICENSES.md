# Veri ve model lisans envanteri

Belirsiz lisanslı kaynak projede kullanılmaz. “Ticari kullanım” alanı doğrulanmadan hiçbir
veri eğitim, değerlendirme veya demo hattına alınmaz.

| Kaynak | Resmî adres | Lisans | Ticari kullanım | Kullanım amacı | Durum |
| --- | --- | --- | --- | --- | --- |
| BDD100K | https://www.bdd100k.com/ | Doğrulama bekliyor | Belirsiz | YOLO ince ayarı | **Kullanıma kapalı** |
| COCO128 | https://docs.ultralytics.com/datasets/detect/coco128/ | COCO anotasyonları CC BY 4.0; görüntüler kendi Flickr lisanslarına tabidir | Görüntü bazında kontrol gerekir | Yalnızca eğitim hattı smoke testi; veri dağıtılmaz | **Etkin** |
| Qwen2.5-VL-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | Qwen Research License | Hayır; ticari kullanım ayrıca izin gerektirir | Kullanılmıyor | **Devre dışı** |
| Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | Evet | Olay açıklama | **Etkin** |
| Ultralytics YOLO | https://github.com/ultralytics/ultralytics | AGPL-3.0 | AGPL koşullarıyla | Nesne tespiti | Kayıtlı |

Depodaki `yolo11n.onnx`, Ultralytics YOLO11n ağırlığından bu proje için dışa aktarılmıştır
ve AGPL-3.0 proje koşulları altında dağıtılır. COCO128 görüntüleri depoya eklenmez; ilk yerel
çalıştırmada resmî arşiv indirilir ve aşağıdaki bütünlük değeriyle doğrulanır.

İhlal klipleri eklendikçe her klip veya koleksiyon için aynı beş alan kaydedilmelidir. BDD100K
koşulları kullanıcı tarafından resmî metinden okunup onaylanana kadar veri demosu yalnızca
kurulum yönlendirmesi verir.

## COCO128 bütünlük kaydı

- Kaynak: `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`
- İndirme tarihi: 2026-10-22
- Arşiv boyutu: 6,983,030 bayt
- SHA-256: `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`
- İçerik: 128 görüntü; YOLO biçimli etiketler
- Kapsam: CI/eğitim hattı doğrulaması. Aynı görüntüler eğitim ve doğrulamada kullanıldığı için
  sonuçlar ürün performansı veya genelleme metriği olarak raporlanmaz.
- Not: Arşivdeki GPL-3.0 metni Ultralytics dönüştürme yazılımına aittir. COCO anotasyonları
  CC BY 4.0'dır; COCO görüntülerinin telifleri kaynak Flickr sahiplerinde kalır.

