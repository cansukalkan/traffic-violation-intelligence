# Scrum çalışma modeli

## Kadans

Bu proje kısa, demo odaklı sprintlerle ilerler. Her Codex çalışma oturumu bir sprint olarak
ele alınır. Oturum başında hedef ve kabul ölçütleri, sonunda ise çalışan demo, test sonucu ve
bilinen riskler sunulur.

## Roller

- Product Owner: proje sahibi; ürün önceliğini ve kabul kararını verir.
- Scrum Master / geliştirici: Codex; kapsamı küçük tutar, engelleri görünür kılar ve kalite
  kapılarını uygular.
- Paydaş: işe alım uzmanı veya teknik görüşmeci; yeniden üretilebilirlik, ölçüm ve mimari
  kararları inceler.

## İş akışı

`Backlog -> Ready -> In progress -> Review -> Done`

Bir işin Ready olması için amacı, arayüzü, testi ve demo çıktısı tanımlı olmalıdır. Aynı anda
tek ana görev In progress tutulur. Bir görev ancak Definition of Done koşullarını sağladığında
Done olur.

## Definition of Done

1. Genel arayüzler tip ipuçlu ve konfigürasyon kod dışındadır.
2. Birim testleri GPU/model/veri indirmeden geçer.
3. `pytest -v -m "not gpu"` başarılıdır.
4. Göreve ait demo çalışır ve `reports/demos/` altında çıktı üretir.
5. Lisans, gizlilik ve performans etkileri belgelenmiştir.
6. Kullanıcıya demo ürün ağacı, çalıştırma komutu ve ölçülen sonuç gösterilir.

## Yol haritası

- Sprint 0: veri/lisans temeli, proje iskeleti, konfigürasyon ve Docker ön kontrolü.
- Sprint 1: geometri ve trafik ışığı durumu.
- Sprint 2: YOLO sarmalayıcısı, sahte dedektör ve iz geçmişi.
- Sprint 3: ihlal kuralları, VLM ayrıştırma ve yönlendirme politikası (Kontrol Noktası A).
- Sprint 4: Qwen2.5-VL ve gerçek model duman testi (Kontrol Noktası B).
- Sprint 5: SQLite `EventStore`, kare kaynağı ve senkron uçtan uca boru hattı.
- Sprint 6: 31 işçili ajan katmanı ve gözlemlenebilirlik (Kontrol Noktası C).
- Sprint 7: donanım yolları, değerlendirme ve kıyaslama (Kontrol Noktaları D/E).
- Sprint 8: tam CLI, Docker teslimi ve tek HTML portföy demosu.
- Final Sprint: harici medya yükleme, FastAPI dashboard, SQLite analiz geçmişi ve kalıcı
  Docker servisi.

PostgreSQL ayrı bir sonraki sürüm backlog öğesidir. `EventStore` Protocol değişmeden,
`psycopg` tabanlı uygulama ve isteğe bağlı Compose profili eklenecektir.

