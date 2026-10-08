# turbo-roformer (Kurzfassung auf Deutsch)

**BS-RoFormer-Gesangstrennung in Echtzeit – auf der CPU.**

Ein 4:12 langer Song wird auf einem 8-Kern-Laptop-Prozessor (AMD Ryzen 7 8845HS) in **3 Minuten 5 Sekunden** in
Gesang und Instrumental zerlegt – ohne Grafikkarte, mit 2,4 GB Arbeitsspeicher und in derselben Qualität wie das
unveränderte PyTorch-Modell. Das entspricht einem Echtzeitfaktor (RTF) von **0,73** und ist **6,7-mal schneller**
als python-audio-separator mit demselben Modell auf derselben Maschine (20:32 Minuten).

| Ryzen 7 8845HS, 4:12-Song, Overlap 2 | Laufzeit | RTF | RAM | Qualität ggü. Overlap-8-Referenz |
|---|---:|---:|---:|---:|
| python-audio-separator 0.47 (fp32) | 20:32 | 4,88 | 3,0 GB | 31,37 dB |
| **turbo-roformer (bf16-Kernel)** | **3:05** | **0,73** | **2,4 GB** | **31,34 dB** |

Der Testsong (kommerzieller Pop-/Worship-Titel) ist aus urheberrechtlichen Gründen nicht enthalten; ein
reproduzierbarer Benchmark mit synthetischem Signal liegt unter `benchmarks/`.

## Schnellstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "turbo-roformer @ git+https://github.com/LoggeL/turbo-roformer@v0.1.0"
turbo-roformer song.flac -o getrennt/
```

```python
from turbo_roformer import Separator

sep = Separator(model="resurrection", precision="bf16", threads=8)  # einmal laden, oft verwenden
inst = sep.separate(mix, sr=44100)  # float32 (Kanäle, Samples) -> Instrumental (2, Samples)
vocals = mix - inst  # Gesang = Mischung minus Instrumental (bei Stereo-Eingang)
```

Beim ersten Aufruf werden die Modellgewichte (204 MB, per SHA-256 geprüft) vom Original-Upload des Autors geladen
und – falls die CPU AVX512-BF16 kann – die C++-Kernel einmalig kompiliert (ca. 10 s, benötigt `g++`).

## Wie funktioniert das?

Pro 17-Sekunden-Abschnitt sinkt die Rechenzeit in mehreren Schritten von 40,3 s auf 6,3 s:

1. **Speicherverwaltung:** glibc hat für jede große Zwischenmatrix frischen Speicher per `mmap` angefordert und
   wieder freigegeben – über eine Milliarde Seitenfehler pro Song. Ein `mallopt`-Aufruf behebt das (40,3 → 31,4 s).
2. **Umstrukturierter Vorwärtsdurchlauf:** Normierungen und Skalierungen in die Gewichte gefaltet, q/k/v und Gates in
   einer Matrixmultiplikation, keine Umsortier-Kopien zwischen Zeit- und Frequenz-Attention – in fp32 exakt.
3. **bf16 statt fp32** für die Operanden der Matrixmultiplikationen, Akkumulation und Residualpfad bleiben fp32.
4. **Eigener Attention-Kernel:** PyTorchs bf16-Attention ist auf CPUs ohne AMX langsam (~0,6 TFLOP/s). Ein
   handgeschriebener `vdpbf16ps`-Kernel mit exaktem Softmax schafft ~1,6 TFLOP/s.
5. **Fusionierte Kernel:** Rotary-Embedding, Residual-Addition und die nächste Normierung werden direkt beim
   Schreiben der Matrixmultiplikations-Ergebnisse erledigt.

Der bf16-Rundungsfehler liegt ~54 dB unter dem Signal (gegenüber dem exakten fp32-Ergebnis), der Fehler durch die
abschnittsweise Verarbeitung selbst dagegen nur ~31 dB (gegenüber einer Overlap-8-Referenz) – bf16 fällt also nicht ins
Gewicht. Ein MP3-Export mit
320 kbit/s verändert das Signal stärker (36,5 dB) als die bf16-Rechnung.

Details: [how-it-works.md](how-it-works.md) und [benchmarks.md](benchmarks.md) (auf Englisch).

## Hardware

- **Schneller Pfad:** Linux x86-64 mit AVX512-BF16 – AMD Zen 4/Zen 5 (Ryzen 7000/8000/9000, EPYC 9004/9005),
  Intel Xeon ab Sapphire Rapids. Gemessen wurde nur auf dem Ryzen 7 8845HS.
- **Alle anderen Systeme** (Intel-Desktop-CPUs, ältere AMD-CPUs, Apple Silicon, ARM, Windows): exakter
  fp32-Pfad mit PyTorch, auf dem 8845HS etwa 3,9-mal langsamer.
- Threads = Anzahl physischer Kerne; Hyperthreading bringt nichts.

## Grenzen

- Ein `separate()`-Aufruf pro Instanz gleichzeitig (gleichzeitige Aufrufe werden per Lock nacheinander abgearbeitet).
- Validiert ist nur das Modell „BS Roformer Instrumental Resurrection“ von unwa.
- Clips unter 17 Sekunden werden nur minimal aufgefüllt und ohne Kontext verarbeitet; das Ergebnis weicht leicht von
  derselben Stelle im ganzen Song ab.
- Keine GPU-Unterstützung.

## Lizenz und Dank

MIT-Lizenz. Modell: „BS Roformer Instrumental Resurrection“ von **unwa** (Gewichte werden nicht mitgeliefert).
Architektur: Lu et al., *Music Source Separation with Band-Split RoPE Transformer*. Referenzimplementierung:
lucidrains/BS-RoFormer und nomadkaraoke/python-audio-separator (beide MIT). Details in [NOTICE](../NOTICE).
