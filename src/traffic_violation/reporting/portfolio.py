"""Generate a self-contained HTML portfolio report from verified demo outputs."""

from __future__ import annotations

import base64
import html
from pathlib import Path


def _image_data(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _text(path: Path) -> str:
    return html.escape(path.read_text(encoding="utf-8").strip())


def build_portfolio_report(root: Path, output: Path) -> Path:
    demos = root / "reports" / "demos"
    images = {
        "detection": _image_data(demos / "04_detect.jpg"),
        "signal": _image_data(demos / "03_signal_preview.png"),
        "tracking": _image_data(demos / "05_tracking_preview.png"),
        "pipeline": _image_data(demos / "11_pipeline_preview.png"),
        "benchmark": _image_data(demos / "13_bench.png"),
        "evaluation": _image_data(demos / "14_eval.png"),
    }
    benchmark = _text(demos / "13_bench.txt")
    evaluation = _text(demos / "14_eval.txt")
    agents = _text(demos / "12_agents.txt")
    document = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trafik İhlali Tespit ve Açıklama Sistemi</title>
<style>
:root{{--bg:#0b1020;--panel:#141b2d;--text:#eef2ff;--muted:#aab4cc;--green:#38d996;--line:#2b3650}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:36px 22px 70px}} h1{{font-size:clamp(30px,5vw,54px);line-height:1.05;margin:.2em 0}} h2{{margin-top:52px}}
.lead{{max-width:820px;color:var(--muted);font-size:18px}} .status{{color:var(--green);font-weight:700}}
.metrics,.gallery,.flow{{display:grid;gap:16px}} .metrics{{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin:30px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px}} .value{{font-size:30px;font-weight:800}} .label,.note{{color:var(--muted)}}
.flow{{grid-template-columns:repeat(auto-fit,minmax(130px,1fr))}} .node{{background:#13233a;border-radius:12px;padding:14px;text-align:center}}
.gallery{{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}} figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden}}
img{{display:block;width:100%;height:auto}} figcaption{{padding:12px 16px;color:var(--muted)}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#090e19;border:1px solid var(--line);padding:18px;border-radius:12px;color:#dce5f8}}
.decision{{border-left:4px solid var(--green);padding-left:16px}} footer{{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}}
</style></head><body><main>
<div class="status">● Docker ile doğrulandı · Sprint 8 teslimi</div>
<h1>Trafik İhlali Tespit ve Açıklama Sistemi</h1>
<p class="lead">YOLO nesne tespiti, iz takibi, deterministik trafik kuralları, kontrollü VLM yönlendirmesi, SQLite olay kaydı ve 31 işçili asenkron orkestrasyonu bir araya getiren denetlenebilir bilgisayarlı görü projesi.</p>
<div class="metrics"><div class="card"><div class="value">97/97</div><div class="label">Geçen test</div></div><div class="card"><div class="value">12 / 31</div><div class="label">Rol / işçi</div></div><div class="card"><div class="value">7,29 FPS</div><div class="label">YOLO11n PyTorch CPU</div></div><div class="card"><div class="value">%57,1</div><div class="label">Sentetik olay F1</div></div></div>
<h2>Uçtan uca mimari</h2><div class="flow"><div class="node">Kamera</div><div class="node">YOLO</div><div class="node">ByteTrack izi</div><div class="node">Kural motoru</div><div class="node">VLM Router</div><div class="node">Qwen3-VL</div><div class="node">SQLite</div></div>
<h2>Çalışan görsel demolar</h2><div class="gallery">
<figure><img src="{images['detection']}" alt="YOLO nesne tespit kutuları"><figcaption>Gerçek YOLO nesne tespiti</figcaption></figure>
<figure><img src="{images['signal']}" alt="Trafik ışığı durumları"><figcaption>HSV tabanlı sinyal durumu: 15/15</figcaption></figure>
<figure><img src="{images['tracking']}" alt="Araç iz geçmişi"><figcaption>Kalıcı track ID ve hareket izi</figcaption></figure>
<figure><img src="{images['pipeline']}" alt="Kırmızı ışıkta çizgi geçişi"><figcaption>Uçtan uca kırmızı ışık adayı</figcaption></figure></div>
<h2>Performans kararı</h2><figure><img src="{images['benchmark']}" alt="PyTorch ve ONNX karşılaştırması"><figcaption>32 kare × 3 tekrar, 320×320 YOLO11n CPU benchmark'ı</figcaption></figure>
<p class="decision">Bu donanımda PyTorch FP32, ONNX Runtime CPU'dan daha hızlı ölçüldü. Varsayım yerine ölçüm kullanıldı ve PyTorch yolu korundu.</p><pre>{benchmark}</pre>
<h2>Model ne kadar doğru?</h2><figure><img src="{images['evaluation']}" alt="Precision recall ve F1 grafiği"><figcaption>Sentetik kabul senaryosu; gerçek ürün accuracy'si değil</figcaption></figure><pre>{evaluation}</pre>
<p class="note">Gerçek accuracy için ihlal olmayan negatif klipler dahil lisanslı 30–60 klibin insan tarafından etiketlenmesi gerekir.</p>
<h2>Ajan katmanı</h2><pre>{agents}</pre>
<h2>Mühendislik güvenceleri</h2><div class="gallery"><div class="card"><strong>Model izolasyonu</strong><p class="note">YOLO ve VLM Protocol arkasında; testler model indirmeden çalışır.</p></div><div class="card"><strong>Denetlenebilirlik</strong><p class="note">Kural kanıtı, routing nedeni ve VLM JSON sonucu olayla birlikte saklanır.</p></div><div class="card"><strong>Geri basınç</strong><p class="note">Sınırlı kuyruk bellek büyümesini engeller; kayıplar ölçülür.</p></div><div class="card"><strong>Lisans disiplini</strong><p class="note">Belirsiz veri kullanılmaz; Qwen3-VL Apache-2.0, uygulama AGPL-3.0.</p></div></div>
<footer>Python 3.11 · PyTorch · Ultralytics YOLO · OpenCV · Qwen3-VL · SQLite · Docker</footer>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
