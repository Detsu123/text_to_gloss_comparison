# German Text → DGS Gloss судалгааны ажлын хураангуй

**Бодит PHOENIX14T өгөгдөл дээр дөрвөн загварыг тус бүр гурван seed-ээр сургаж, нийт 12 туршилт хийсэн. Энэ нөхцөлд GRU + Attention хамгийн сайн үр дүнтэй байлаа.**

## 1. Бодитоор хийсэн зүйл

- PHOENIX14T-ийн текст annotation-уудыг татаж, train **7,096**, dev **519**, test **642** мөрийг шалгасан. Видео татаж боловсруулаагүй.
- German `translation` → DGS `orth` чиглэлээр нийтлэг preprocessing хийсэн.
- Vocabulary-г зөвхөн train хэсгээс үүсгэсэн: **2,887 German үг, 1,085 gloss**; тус бүрд дөрвөн special token нэмэгдэнэ.
- RNN + Attention, GRU + Attention, LSTM + Attention, Transformer encoder–decoder-ийг PyTorch-оор хэрэгжүүлсэн.
- **128 хэмжээст, 10 epoch, batch 32, seed 42/43/44, CPU FP32** нөхцөлд сургасан. Нэг загварын нэг seed нэг тусдаа сургалт болно.
- Checkpoint-ийг dev BLEU-4-өөр сонгосон. Бүх сургалт дууссаны дараа test үнэлгээ хийсэн; нийт **7,704 prediction** хадгалсан.
- BLEU-1/2/3/4, chrF, TER, exact match, token error rate, parameter count, хугацаа, уртын бүлгийн болон алдааны шинжилгээ гаргасан.
- Найман төрлийн графикийг 300 dpi PNG, PDF, SVG хэлбэрээр гаргасан.
- **16 автомат шалгалт тэнцсэн**: дөрвөн загварын forward/backward, masks, tiny overfit, inference/EOS, metric, checkpoint, train-only vocabulary болон бүтэн pipeline-ийн шалгалт.

## 2. Хийгдээгүй зүйл ба шалтгаан

Энэ орчинд CUDA GPU байгаагүй. Иймээс RTX 4050 дээрх VRAM, AMP болон GPU хугацааг хэмжээгүй. Богино CPU туршилтыг **pilot study буюу урьдчилсан туршилт** гэж тэмдэглэсэн. 256 хэмжээст, 40 хүртэл epoch-ийн бүрэн тохиргоог бэлдсэн боловч энд сургаагүй.

Өгөгдөл нь PHOENIX14T-ийн хувилбарыг тогтоосон public annotation mirror-оос ирсэн. Албан ёсны split хэмжээ, бүтэц, үгийн санг шалгаж, файлын SHA-256-ийг хадгалсан. Бүтэн 39 GB албан ёсны архивтай byte-for-byte тулгаагүй. Лицензийн эх сурвалжуудын хувилбарын зөрүүг `data/README.md`-д тайлбарласан; өгөгдлийг хязгаарлалтгүй арилжааны дата гэж үзээгүй.

DGS мэргэжилтний хэлзүйн үнэлгээ, beam search, баталгаажсан дүрэмт baseline болон Монгол дохионы хэлний моделийг хийгээгүй. Дүрэмт baseline-д үндэслэлгүй DGS дүрэм зохиож оруулаагүй.

## 3. Аль загвар хамгийн сайн байсан бэ?

**Test-ийн 642 өгүүлбэр дээрх гурван seed-ийн дундаж ± стандарт хазайлт:**

| Загвар | BLEU-4 ↑ | chrF ↑ | TER ↓ | Parameter |
|---|---:|---:|---:|---:|
| RNN + Attention | 7.66 ± 0.35 | 34.15 ± 0.64 | 73.79 ± 0.90 | 765,249 |
| **GRU + Attention** | **14.28 ± 0.42** | **44.98 ± 0.62** | **63.38 ± 0.48** | **897,345** |
| LSTM + Attention | 12.32 ± 0.89 | 43.64 ± 0.69 | 64.29 ± 0.83 | 963,393 |
| Transformer | 8.30 ± 0.33 | 36.11 ± 1.27 | 77.57 ± 6.77 | 1,312,961 |

GRU нь BLEU-4, chrF, TER гурваар хамгийн сайн дундажтай. Гэхдээ BLEU-4 **14.28 гэдэг нь “14.28% зөв орчуулсан” гэсэн үг биш**. Энэ бол reference gloss-той n-gram давхцлыг үнэлэх оноо. GRU-ийн бүтэн дарааллын exact match дунджаар 3.27% байсан.

Train-д байсан яг ижил text/gloss хос test-ийн 14 мөрд давтагдсан. Тэдгээрийг хассан 628 мөр дээр ч GRU хамгийн өндөр дундаж BLEU-4 (**13.83**) авсан. Үндсэн benchmark-д албан ёсны бүх 642 мөрийг хэвээр үлдээсэн.

## 4. Анхны таамаглалыг дэмжсэн үү?

**“Transformer заавал илүү, ялангуяа урт өгүүлбэр дээр илүү” гэсэн таамаглалыг энэ туршилт дэмжээгүй.**

Уртын босгыг train өгөгдлийн тархалтаас тогтоосон. Доорх нь test BLEU-4-ийн гурван seed-ийн дундаж:

| Загвар | Богино ≤11 үг, n=328 | Дунд 12–16 үг, n=200 | Урт ≥17 үг, n=114 |
|---|---:|---:|---:|
| RNN | 13.08 | 4.36 | 4.15 |
| GRU | 21.07 | 11.98 | 9.18 |
| LSTM | 18.25 | 10.22 | 7.87 |
| Transformer | 14.12 | 4.96 | 4.08 |

Бүх загвар урт өгүүлбэрийн бүлэгт доогуур оноотой. Энэ нь уртын бүлгүүдийн ажиглалт; урт дангаараа шалтгаан гэдгийг нотлоогүй. Бүлгүүдийн үг, агуулга, reference-ийн урт болон хэмжээ өөр.

Transformer seed 42 нь зургаан өгүүлбэр дээр EOS гаргалгүй 80-token хязгаарт хүрсэн. Нэг бодит жишээнд `NAECHSTE WOCHE` гэсэн хоёр token-ийг 40 удаа давтсан. Энэ алдааг prediction болон qualitative analysis файлд хадгалсан; нууж хасаагүй.

## 5. Хурал дээр юу танилцуулах вэ?

Санал болгох гарчиг:

**A Controlled Comparison of Recurrent and Transformer Models for German Text-to-Gloss Translation**

Илтгэлийн гол дараалал:

1. Асуудал: German текстээс DGS gloss-ийн зөв дараалал гаргах; видео таних ажил биш.
2. Арга: ижил split, preprocessing, optimizer, greedy decoding, metric; гурван seed.
3. Үр дүн: GRU-ийн энэ тохиргоон дахь давуу үр дүн, parameter count, уртын бүлгийн ялгаа.
4. Бодит алдаа: gloss орхих, өөр token сонгох, урт өгүүлбэрийн алдаа, давталтын loop.
5. Хязгаар ба дараагийн алхам: 10 epoch-ийн pilot, ялгаатай parameter count, нэг weather-domain corpus, DGS эксперт үнэлгээ хийгдээгүй; дараа нь урт сургалт, parameter-matched туршилт, Монгол корпус.

Үүнийг “шинэ архитектур зохиосон” гэж танилцуулах үндэслэлгүй. Харин **нэгдсэн нөхцөлтэй, давтагдахуйц харьцуулсан туршилт** гэж тайлбарлах нь зөв. Хурлын эцсийн хувилбарт оруулахаасаа өмнө RTX дээр урт сургалт хийж, хэлзүйн жишээнүүдээ мэргэжилтнээр хянуулах нь судалгааг бэхжүүлнэ.

## 6. RTX 4050 6 GB дээр ажиллуулах команд

ZIP-ийг задлаад `text_to_gloss_comparison` хавтас дотор terminal нээнэ.

Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_all.py --prepare --config config/default.yaml --seeds 42 43 44
```

Windows PowerShell-д эхний хоёр мөрийн оронд:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Бусад `python ...` команд адил. CUDA ашиглаж байгааг шалгах:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Хадгалсан pilot GRU-г шууд ашиглах:

```bash
python -m src.inference --config config/cpu_pilot.yaml --model gru --text "morgen ist es im norden bewölkt ."
```

Бодитоор гарсан seed-42 prediction: `MORGEN NORD WOLKE HIMMEL`.

Санах ой хүрэхгүй бол `config/default.yaml`-д `batch_size: 8`, `grad_accumulation: 2` болгоно. Өөрчлөлтийг бүх загварт ижил хэрэглэнэ. Нэг batch-ийн GPU peak шалгах команд: `python scripts/profile_memory.py`.

Шинэ бүрэн туршилт `runs/phoenix14t_full/` дотор үүснэ. Одоогийн бодит pilot үр дүн `runs/phoenix14t_cpu_pilot/` дотор тусдаа хадгалагдсан. Код existing checkpoint-ийг санамсаргүй дарж бичихгүй; шинэ туршилтад YAML-ийн `output`-ыг шинэ хавтас болгоно.
