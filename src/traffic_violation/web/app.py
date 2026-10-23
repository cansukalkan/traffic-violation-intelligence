"""FastAPI application factory for the local product dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from traffic_violation.web.service import (
    MediaAnalysisService,
    MediaValidationError,
    SampleCatalog,
)


def create_app(
    *,
    data_dir: Path | None = None,
    mode: str | None = None,
    sample_dir: Path | None = None,
) -> FastAPI:
    root = data_dir or Path(os.environ.get("TV_DATA_DIR", "reports/dashboard"))
    analyzer_mode = mode or os.environ.get("TV_ANALYZER_MODE", "source")
    if analyzer_mode not in {"source", "onnx", "yolo"}:
        raise ValueError("TV_ANALYZER_MODE must be source, onnx or yolo")
    service = MediaAnalysisService(root, mode=analyzer_mode)
    samples = SampleCatalog(
        sample_dir
        or Path(os.environ.get("TV_SAMPLE_DIR", "data/coco128/images/train2017"))
    )
    app = FastAPI(
        title="Traffic Violation Intelligence",
        version="1.0.0",
        description="Local-first traffic media ingestion and analysis dashboard",
    )
    app.state.analysis_service = service
    app.mount("/media", StaticFiles(directory=service.preview_dir), name="media")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _dashboard_html(analyzer_mode)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "storage": "sqlite", "analyzer_mode": analyzer_mode}

    @app.get("/api/jobs")
    def list_jobs(limit: int = 25) -> dict[str, object]:
        try:
            jobs = service.store.list(limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"items": [job.to_dict() for job in jobs], "count": len(jobs)}

    @app.get("/api/samples")
    def list_samples() -> dict[str, object]:
        items = samples.list()
        return {"items": [item.to_dict() for item in items], "count": len(items)}

    @app.get("/api/samples/{sample_id}/thumbnail", response_class=FileResponse)
    def sample_thumbnail(sample_id: str) -> FileResponse:
        sample = samples.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="Örnek veri bulunamadı.")
        return FileResponse(sample.path)

    @app.post("/api/samples/{sample_id}/analyze", status_code=201)
    def analyze_sample(sample_id: str) -> dict[str, object]:
        sample = samples.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="Örnek veri bulunamadı.")
        try:
            with sample.path.open("rb") as stream:
                job = service.analyze_upload(stream, f"sample-{sample.name}")
        except MediaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        job = service.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analiz kaydı bulunamadı.")
        return job.to_dict()

    @app.post("/api/analyze", status_code=201)
    def analyze(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
        try:
            job = service.analyze_upload(file.file, file.filename or "upload")
        except MediaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/preview", response_class=FileResponse)
    def preview(job_id: str) -> FileResponse:
        job = service.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analiz kaydı bulunamadı.")
        return FileResponse(service.preview_dir / job.preview_name, media_type="image/jpeg")

    return app


def _dashboard_html(mode: str) -> str:
    mode_label = "YOLO11n ONNX nesne tespiti" if mode == "onnx" else (
        "YOLO nesne tespiti" if mode == "yolo" else "Kaynak doğrulama"
    )
    model_note = (
        "Model aktif: araç, insan ve trafik nesneleri kutu ve güven skoruyla işaretlenir."
        if mode == "onnx"
        else
        "Model aktif: önizlemede bulunan nesneler işaretlenir. İhlal kararı için kamera geometrisi gerekir."
        if mode == "yolo"
        else "Model kapalı: dosya doğrulanır, video hareketi ölçülür; sahte ihlal üretilmez."
    )
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic Violation Intelligence</title><style>
:root{{--bg:#07111f;--panel:#0d1b2d;--line:#20334d;--text:#ecf4ff;--muted:#94a9c6;--green:#32dda3;--cyan:#55c7ff;--red:#ff6b7a}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 70% -10%,#163a55 0,transparent 38%),var(--bg);color:var(--text);font:15px Inter,Segoe UI,sans-serif}}
.shell{{max-width:1180px;margin:auto;padding:34px 22px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:28px}}h1{{font-size:clamp(28px,5vw,52px);line-height:1;margin:8px 0}}.eyebrow{{color:var(--green);letter-spacing:.14em;font-weight:700}}.status{{padding:11px 15px;border:1px solid #2c5b50;border-radius:999px;background:#10342c;color:#8ef0ca;white-space:nowrap}}
.grid{{display:grid;grid-template-columns:1.05fr .95fr;gap:20px}}.card{{background:linear-gradient(145deg,#102139e6,#0a1627e6);border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 18px 50px #0005}}h2{{margin:0 0 8px;font-size:20px}}p{{color:var(--muted);line-height:1.55}}.drop{{display:block;border:1.5px dashed #3d6384;border-radius:16px;padding:34px;text-align:center;cursor:pointer;margin:20px 0;background:#091827}}.drop:hover{{border-color:var(--cyan)}}input[type=file]{{display:none}}button{{border:0;border-radius:12px;background:linear-gradient(90deg,var(--green),var(--cyan));color:#06111d;font-weight:800;padding:13px 19px;cursor:pointer}}button:disabled{{opacity:.5}}.small{{font-size:13px}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px}}.metric{{padding:14px;background:#091827;border-radius:13px}}.metric b{{display:block;font-size:23px;margin-top:5px}}img{{width:100%;height:330px;object-fit:contain;background:#050b13;border-radius:14px;border:1px solid var(--line)}}.empty{{height:330px;display:grid;place-items:center;border:1px solid var(--line);border-radius:14px;color:var(--muted)}}table{{width:100%;border-collapse:collapse;margin-top:14px}}td,th{{padding:12px 8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}.wide{{grid-column:1/-1}}.pill{{color:#8ef0ca}}#error{{color:#ff9aa5;min-height:20px}}.samples{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}.sample{{padding:0;overflow:hidden;background:#091827;border:1px solid var(--line);color:var(--text);font-weight:600}}.sample img{{height:92px;border:0;border-radius:0;object-fit:cover}}.sample span{{display:block;padding:9px;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}@media(max-width:760px){{header{{display:block}}.grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr}}.samples{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main class="shell"><header><div><div class="eyebrow">LOCAL-FIRST • DOCKER • SQLITE</div><h1>Traffic Violation<br>Intelligence</h1><p>Harici trafik görüntüsünü yükle, doğrula ve analiz kaydını denetlenebilir biçimde sakla.</p></div><div class="status">● Sistem çalışıyor</div></header>
<section class="grid"><article class="card"><h2>Yeni analiz</h2><p><b>{mode_label}</b> — {model_note}</p><form id="form"><label class="drop" for="file"><b id="fileLabel">Görsel veya videoyu buraya bırak / seç</b><br><span class="small">JPG, PNG, MP4, AVI, MOV, MKV • en fazla 100 MB</span></label><input id="file" name="file" type="file" accept="image/jpeg,image/png,video/*" required><button id="submit">Analizi başlat</button></form><p id="error"></p><hr style="border:0;border-top:1px solid var(--line);margin:22px 0"><h2>Hazır örnek veri</h2><p class="small">Dosyan yoksa COCO128 demo galerisinden seç. Bunlar trafik ihlali etiketi değil, ürün akışı örnekleridir.</p><div id="samples" class="samples"><span class="small">Örnekler yükleniyor…</span></div></article>
<article class="card"><h2>Sonuç önizlemesi</h2><div id="preview" class="empty">Henüz analiz seçilmedi</div><div class="metrics"><div class="metric"><span>Boyut</span><b id="size">—</b></div><div class="metric"><span>Süre / kare</span><b id="duration">—</b></div><div class="metric"><span>Hareket</span><b id="motion">—</b></div></div></article>
<article class="card wide"><h2>SQLite analiz geçmişi</h2><p>Her yükleme benzersiz kimlikle saklanır. Ham dosya adı çalıştırma yolu olarak kullanılmaz.</p><div style="overflow:auto"><table><thead><tr><th>Zaman</th><th>Dosya</th><th>Tür</th><th>Mod</th><th>Tespit</th><th>Durum</th></tr></thead><tbody id="jobs"></tbody></table></div></article></section></main>
<script>
const file=document.querySelector('#file'), form=document.querySelector('#form'), submit=document.querySelector('#submit'), error=document.querySelector('#error');
file.addEventListener('change',()=>document.querySelector('#fileLabel').textContent=file.files[0]?.name||'Dosya seç');
function esc(v){{return String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function show(j){{document.querySelector('#preview').className='';document.querySelector('#preview').innerHTML=`<img alt="Analiz önizlemesi" src="/api/jobs/${{j.id}}/preview?t=${{Date.now()}}"><p><b>${{j.detection_count}} nesne bulundu</b>${{j.detected_classes.length?' • '+j.detected_classes.join(', '):''}}</p>`;document.querySelector('#size').textContent=`${{j.width}}×${{j.height}}`;document.querySelector('#duration').textContent=j.media_kind==='video'?`${{j.duration_s.toFixed(1)}} sn`:`${{j.frame_count}} kare`;document.querySelector('#motion').textContent=`${{(j.motion_score*100).toFixed(1)}}%`;}}
async function load(){{const r=await fetch('/api/jobs');const d=await r.json();document.querySelector('#jobs').innerHTML=d.items.map(j=>`<tr data-id="${{j.id}}"><td>${{new Date(j.created_at*1000).toLocaleString('tr-TR')}}</td><td>${{esc(j.original_name)}}</td><td>${{j.media_kind}}</td><td>${{j.mode}}</td><td>${{j.detection_count}}</td><td class="pill">${{j.status}}</td></tr>`).join('')||'<tr><td colspan="6">Henüz kayıt yok.</td></tr>';document.querySelectorAll('tr[data-id]').forEach(row=>row.onclick=()=>{{const j=d.items.find(x=>x.id===row.dataset.id);if(j)show(j)}});if(d.items[0])show(d.items[0]);}}
async function loadSamples(){{const r=await fetch('/api/samples');const d=await r.json();document.querySelector('#samples').innerHTML=d.items.map(s=>`<button class="sample" type="button" data-sample="${{s.id}}"><img alt="${{esc(s.name)}}" src="${{s.thumbnail_url}}"><span>${{esc(s.name)}}</span></button>`).join('')||'<span class="small">Örnek veri dizini bulunamadı.</span>';document.querySelectorAll('[data-sample]').forEach(b=>b.onclick=()=>analyzeSample(b.dataset.sample,b));}}
async function analyzeSample(id,button){{document.querySelectorAll('[data-sample]').forEach(b=>b.disabled=true);const old=button.querySelector('span').textContent;button.querySelector('span').textContent='Analiz ediliyor…';error.textContent='';try{{const r=await fetch(`/api/samples/${{id}}/analyze`,{{method:'POST'}});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Örnek analiz edilemedi');show(d);error.style.color='#8ef0ca';error.textContent=`Tamamlandı: ${{d.detection_count}} nesne bulundu.`;await load();}}catch(e){{error.style.color='#ff9aa5';error.textContent=e.message}}finally{{button.querySelector('span').textContent=old;document.querySelectorAll('[data-sample]').forEach(b=>b.disabled=false)}}}}
form.addEventListener('submit',async e=>{{e.preventDefault();if(!file.files[0])return;submit.disabled=true;submit.textContent='Analiz ediliyor…';error.textContent='';const body=new FormData();body.append('file',file.files[0]);try{{const r=await fetch('/api/analyze',{{method:'POST',body}});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Analiz başarısız');show(d);await load();}}catch(e){{error.textContent=e.message}}finally{{submit.disabled=false;submit.textContent='Analizi başlat'}}}});load();loadSamples();
</script></body></html>"""
