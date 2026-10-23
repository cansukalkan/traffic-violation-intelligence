# Veri hazırlığı

Veri dosyaları Git'e eklenmez. BDD100K'yı yalnızca resmî kaynaktan indirin ve aşağıdaki
yerleşimi kullanın:

```text
<bdd-root>/
  images/100k/train/*.jpg
  images/100k/val/*.jpg
  labels/...
```

Alt küme üretici yalnızca mutlak görüntü yollarını `train.txt` ve `val.txt` manifestlerine
yazar; görüntü kopyalamaz. Aynı seed ve kaynak dizin her zaman aynı listeyi üretir.

İhlal etiketi örneği `data/labels/example.json` dosyasındadır. Desteklenen ilk sürüm olayları
`red_light`, `stop_line` ve `crosswalk` değerleridir. `frame_t` saniye cinsindendir.

Ham karelerin hedef saklama süresi yedi gündür. İlk sürümde plaka tanıma kapsam dışıdır.

