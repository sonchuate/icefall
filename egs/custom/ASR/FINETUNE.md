# Finetune gipformer trên dữ liệu ZAC 2026

Tài liệu này mô tả hai cách finetune model tiếng Việt gipformer (Zipformer RNN-T, vocab 2000) trên tập train của Zalo AI Challenge 2026:

| | Full finetune | LoRA |
|---|---|---|
| Thư mục code | `zipformer/` | `zipformer_lora/` |
| Script | `finetune_zac.sh` | `finetune_zac_lora.sh` |
| Tham số được train | Toàn bộ, khoảng 69,7 triệu | Chỉ adapter, 1,78% model (`LORA_R=8`) |
| Learning rate mặc định | `0.001` | `0.0045` |
| Số epoch mặc định | 10 | 20 |
| Bước thêm sau train | Không | Merge adapter vào model gốc |
| WER trên dev | **6,29%** (model gốc 14,41%) | Chưa có |

Nên bắt đầu với full finetune vì đây là cách đã có kết quả. LoRA đáng thử nếu full finetune bị overfit, nhưng chưa có bằng chứng là nó giảm WER.

Mọi lệnh bên dưới chạy từ `egs/custom/ASR`, trừ khi có ghi chú khác.

## 1. Cài môi trường

Cần Linux hoặc WSL với GPU NVIDIA. `k2` phải khớp đúng phiên bản torch, CUDA và Python, xem danh sách wheel tại <https://k2-fsa.github.io/k2/cuda.html>.

Hai tổ hợp đã chạy được:

| Môi trường | Python | torch | Wheel `k2` |
|---|---|---|---|
| WSL, driver CUDA 12.2 | 3.10 | `2.5.1+cu121` | [`cuda12.1.torch2.5.1`](https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250715+cuda12.1.torch2.5.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl) |
| Kaggle, 2 GPU T4 | 3.13 | `2.11.0+cu128` | [`cuda12.8.torch2.11.0`](https://huggingface.co/csukuangfj2/k2/resolve/main/ubuntu-cuda/1.24.4.dev20260625/k2-1.24.4.dev20260626+cuda12.8.torch2.11.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl) |

Sau khi cài torch và `k2`, cài các thư viện còn lại:

```bash
pip install lhotse kaldifst kaldialign sentencepiece tensorboard typeguard dill \
  soundfile num2words pypinyin==0.50.0 packaging urllib3 requests
```

Đặt biến môi trường để `k2` tìm thấy thư viện CUDA và `icefall` import được:

```bash
export PYTHONPATH=/duong/dan/toi/icefall:$PYTHONPATH
export LD_LIBRARY_PATH=$(python - <<'E'
import glob, os, torch
site = os.path.dirname(os.path.dirname(torch.__file__))
names = ["libcudart.so*", "libnvrtc.so*", "libnvToolsExt.so*", "libtorch.so"]
print(":".join({os.path.dirname(p) for n in names for p in glob.glob(f"{site}/**/{n}", recursive=True)}))
E
):$LD_LIBRARY_PATH

python -c "import torch,k2,lhotse,icefall; print(torch.__version__, torch.cuda.device_count())"
```

Trong notebook, đặt các biến này qua `os.environ`, vì `export` trong ô `!` không giữ lại cho ô sau.

**Clone vào ổ Linux.** Nếu clone trên Windows với `core.symlinks=false`, các file symlink trong `icefall/` như `icefall/transformer_lm/scaling.py` sẽ thành file text và gây `SyntaxError`.

## 2. Chuẩn bị dữ liệu

Git không chứa dữ liệu và model. README của bộ dữ liệu cấm phát tán, nên không đưa chúng lên nơi công khai.

```text
icefall/data/gipformer/          # model.pt, bpe.model, tokens.txt, ...
icefall/data/train+public_test/  # chỉ dùng train/
```

Chỉ `train/` có nhãn `audio_text`. `public_test` không có nhãn nên không dùng để train được.

Tạo `metadata.csv` với hai cột `text,audio_path`:

```bash
python local/prepare_zac_csv.py \
  --train-dir ../../../data/train+public_test/train \
  --output ../../../data/metadata.csv
```

Kết quả đúng là 5.016 dòng, loại 20 câu, tổng khoảng 6,8 giờ audio. Script dùng cả các lượt user trong `history` và chuẩn hóa text cho khớp tokenizer:

- Viết hoa, bỏ dấu câu.
- Đổi số thành chữ, ví dụ `VTV3` thành `VTV BA`.
- Loại các câu có mã không rõ cách đọc như `RV123`.

`audio_path` là đường dẫn tương đối so với thư mục chứa `metadata.csv`. Nếu dữ liệu nằm ở nơi khác và không tạo được symlink, chẳng hạn `/kaggle/input` trên Kaggle, hãy đổi cột này sang đường dẫn tuyệt đối:

```python
import csv
from pathlib import Path

METADATA = Path("/kaggle/working/icefall/data/metadata.csv")
DATA_ROOT = Path("/kaggle/input/.../train+public_test")

with open(METADATA, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    row["audio_path"] = str(DATA_ROOT / row["audio_path"][row["audio_path"].index("train/audio/"):])

missing = [r["audio_path"] for r in rows if not Path(r["audio_path"]).is_file()]
assert not missing, missing[:3]

with open(METADATA, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "audio_path"])
    writer.writeheader()
    writer.writerows(rows)
```

`finetune.py` và `decode.py` tách tập dev ngay trong code: 8% hội thoại, seed 42, các lượt cùng hội thoại luôn nằm cùng một tập. Muốn so WER giữa các lần chạy thì giữ nguyên `--dev-ratio` và `--split-seed`.

## 3. Full finetune

```bash
WORLD_SIZE=2 MAX_DURATION=120 NUM_EPOCHS=20 \
GIPFORMER_DIR=../../../data/gipformer \
EXP_DIR=zipformer/exp_zac \
bash finetune_zac.sh
```

Biến môi trường của script:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `WORLD_SIZE` | `1` | Số GPU |
| `MAX_DURATION` | `60` | Số giây audio mỗi batch trên một GPU |
| `NUM_EPOCHS` | `10` | Số epoch |
| `BASE_LR` | `0.001` | Learning rate |
| `GIPFORMER_DIR` | `../../../data/gipformer` | Thư mục chứa `model.pt` và `bpe.model` |
| `METADATA_CSV` | `../../../data/metadata.csv` | File dữ liệu |
| `EXP_DIR` | `zipformer/exp_zac` | Nơi lưu checkpoint và log |

**Chọn `MAX_DURATION`.** Tập này nhỏ nên batch lớn làm giảm số bước cập nhật. Với 2 GPU, `MAX_DURATION=120` cho khoảng 90 bước mỗi epoch, còn `300` chỉ khoảng 38 bước. GPU 4 GB đã chạy được với `60` trên 1 GPU.

**Dung lượng.** Mỗi `epoch-*.pt` nặng khoảng 1,1 GB vì chứa cả optimizer và `model_avg`. Trên Kaggle, 20 epoch dễ vượt giới hạn của `/kaggle/working`. Hãy đặt `EXP_DIR` tới `/kaggle/temp/...`, rồi lưu checkpoint tốt nhất trước khi hết phiên.

Finetune luôn bắt đầu từ epoch 1 và ghi đè checkpoint cũ, nên mỗi lần chạy nên dùng một `EXP_DIR` riêng.

Theo dõi quá trình train:

```bash
tensorboard --logdir zipformer/exp_zac/tensorboard
```

Checkpoint tốt nhất trên dev được lưu tại `$EXP_DIR/best-valid-loss.pt`.

## 4. LoRA

LoRA giữ nguyên model gốc và chỉ train các adapter nhỏ gắn vào lớp self-attention và feed-forward.

```bash
WORLD_SIZE=2 MAX_DURATION=120 NUM_EPOCHS=20 LORA_R=8 \
GIPFORMER_DIR=../../../data/gipformer \
EXP_DIR=/kaggle/temp/exp_lora \
bash finetune_zac_lora.sh
```

Ngoài các biến ở mục 3, script có thêm `LORA_R` (kích thước bottleneck, mặc định `8`). Giá trị mặc định khác với full finetune là `BASE_LR=0.0045`, `NUM_EPOCHS=20` và `EXP_DIR=zipformer_lora/exp_zac`.

Sau khi train, gộp adapter vào model gốc:

```bash
python zipformer_lora/merge_lora.py \
  --checkpoint /kaggle/temp/exp_lora/epoch-20.pt \
  --base-model ../../../data/gipformer/model.pt \
  --output zipformer/exp_lora_merged/epoch-1.pt
```

File kết quả có cùng key và định dạng với `model.pt` gốc, nên dùng được cho decode và export ở các mục sau mà không cần code LoRA. Thêm `--key model_avg` nếu muốn merge bộ trọng số trung bình.

**Về `--param-min-rms`.** `zipformer_lora/finetune.py` đặt mặc định `1e-3` thay vì `1e-5` của `ScaledAdam`. `lora_B` khởi tạo bằng 0 mà bước cập nhật của `ScaledAdam` tỉ lệ với độ lớn tham số, nên với `1e-5` adapter gần như không học dù log vẫn trông bình thường. Trong phép đo 40 batch, norm trung bình của `lora_B` tăng từ `8e-6` lên `8e-4` sau khi đổi. Giá trị `1e-3` được suy ra từ công thức, chưa được dò thực nghiệm. Nếu WER gần như không đổi, hãy thử `1e-2` bằng cách sửa `default` trong `finetune.py`.

Trạng thái kiểm chứng của LoRA:

- Đã kiểm tra trên 1 GPU, 40 batch: checkpoint gốc nạp đủ và kết quả merge khớp cách model tự merge ở chế độ eval.
- Chưa chạy trọn một lần train nào, chưa chạy 2 GPU và chưa có WER sau khi train xong.

## 5. Đánh giá trên dev

`decode.py` đọc checkpoint theo tên `epoch-N.pt`, nên hãy đưa checkpoint muốn đánh giá về tên đó. Bước trích dưới đây giữ lại đúng phần trọng số:

```python
import pathlib, sys, torch

# Checkpoint lưu từ Python 3.13 không mở được trên bản cũ hơn nếu thiếu dòng này.
if sys.version_info < (3, 13):
    sys.modules["pathlib._local"] = pathlib

SOURCE = "zipformer/exp_zac/best-valid-loss.pt"
OUTPUT = pathlib.Path("zipformer/exp_best/epoch-1.pt")

checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
state_dict = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": state_dict}, OUTPUT)
```

Với LoRA, dùng trực tiếp file đã merge ở mục 4 và bỏ qua bước này.

```bash
python zipformer/decode.py --epoch 1 --avg 1 --use-averaged-model 0 \
  --exp-dir zipformer/exp_best \
  --bpe-model ../../../data/gipformer/bpe.model \
  --use-ctc 1 --use-transducer 1 \
  --decoding-method greedy_search \
  --on-the-fly-feats 1 --max-duration 120
```

Kết quả nằm trong `$EXP_DIR/greedy_search/`. Đổi `checkpoint["model"]` thành `checkpoint["model_avg"]` để thử bộ trọng số trung bình.

Luôn giữ `--use-ctc 1` vì checkpoint gipformer có CTC head.

## 6. Export ONNX

Cài thêm thư viện. Hai phiên bản dưới đây đã chạy được với torch 2.5.1. Với torch mới hơn có thể cần phiên bản khác.

```bash
pip install onnx==1.17.0 onnxruntime==1.20.1 sherpa-onnx
```

`export_zac.sh` trích trọng số từ checkpoint, export ONNX rồi gom vào `$GIPFORMER_DIR/<tên>` với cấu trúc giống model gốc:

```bash
# Full finetune
bash export_zac.sh ../../../data/gipformer/best-valid-loss.pt finetune_best

# LoRA: dùng file đã merge ở mục 4
bash export_zac.sh zipformer/exp_lora_merged/epoch-1.pt lora_best
```

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GIPFORMER_DIR` | `../../../data/gipformer` | Thư mục có `tokens.txt`, `bpe.model`, `bpe.vocab`, `config.json` |
| `WORK_DIR` | `~/gip_export` | Thư mục trung gian, chứa cả các bản `.int8.onnx` |
| `KEY` | `model` | Bộ trọng số cần export, `model` hoặc `model_avg` |

Script báo lỗi nếu checkpoint LoRA chưa được merge. Quy trình này đã chạy với Python 3.10 và torch 2.5.1. Chưa thử export với torch 2.11, nên nên tải checkpoint từ Kaggle về rồi export ở môi trường WSL.

## 7. Infer

```python
import sherpa_onnx
import soundfile as sf

d = "data/gipformer/finetune_best"
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{d}/encoder.onnx",
    decoder=f"{d}/decoder.onnx",
    joiner=f"{d}/joiner.onnx",
    tokens=f"{d}/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

audio, sample_rate = sf.read("duong/dan/file.wav", dtype="float32")
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio)
recognizer.decode_stream(stream)
print(stream.result.text)
```

Audio phải là 16 kHz mono. Text đầu ra viết hoa, không có dấu câu và số ở dạng chữ.

## 8. Kết quả

Đo trên 417 câu dev (6.141 từ), `greedy_search`:

| Model | PyTorch | ONNX qua `sherpa-onnx` |
|---|---|---|
| gipformer gốc | 14,41% | 14,46% |
| Full finetune, `model` | **6,29%** | 6,30% |
| Full finetune, `model_avg` | 6,61% | Chưa đo |

Checkpoint full finetune được lưu ở epoch 9, batch 820.

Tập dev đã được dùng để chọn checkpoint, nên số trên `public_test` có thể cao hơn.

## 9. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ImportError: libnvToolsExt.so.1` hoặc `libnvrtc.so.12` | `k2` không tìm thấy thư viện CUDA | Đặt `LD_LIBRARY_PATH` như mục 1 |
| `SyntaxError` ở file chỉ có một đường dẫn | Symlink thành file text khi clone trên Windows | Clone vào ổ Linux |
| `NOT_FOUND: .../bpe.model` | Sai đường dẫn model | Đặt `GIPFORMER_DIR` đúng |
| `AudioLoadingError ... No such file or directory` | `audio_path` trong CSV không trỏ tới file thật | Đổi sang đường dẫn tuyệt đối như mục 2 |
| `CUDNN_STATUS_EXECUTION_FAILED` ở CTC loss | cuDNN CTC lỗi trong WSL | `zipformer/model.py` đã tắt cuDNN cho đoạn này, kiểm tra đang dùng đúng nhánh |
| `No module named 'pathlib._local'` | Checkpoint lưu từ Python 3.13, mở bằng bản cũ hơn | Dùng đoạn trích trọng số ở mục 5 |
| `WER` LoRA gần như không đổi | Adapter chưa học được | Xem phần `--param-min-rms` ở mục 4 |
