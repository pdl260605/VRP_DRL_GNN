# CVRP Solver with Graph Neural Network

PyTorch implementation of **"Attention, Learn to Solve Routing Problems!"** (Kool et al., 2019) ([arXiv](https://arxiv.org/pdf/1803.08475.pdf)), mở rộng với kiến trúc **GNN Light** nhẹ hơn và giao diện so sánh trực quan.

<img src="https://user-images.githubusercontent.com/51239551/88506411-cd450f80-d014-11ea-84eb-12e7ab983780.gif" width="650"/>

---

## Mô tả

Dự án gồm **hai kiến trúc model**:

| Model | Script vẽ | Kiến trúc | Embed dim | Layers |
|---|---|---|---|---|
| `VRP{N}_train_epoch{E}.pt` | `plot.py` | Multi-Head Attention | 128 | 3 |
| `VRP{N}_train_GNN_Light_epoch{E}.pt` | `plot_light.py` | GNN Light | 64 | 2 |

---

## Cài đặt

```bash
pip install torch numpy matplotlib plotly tqdm
```

Yêu cầu: Python ≥ 3.8, PyTorch ≥ 1.9

---

## Cấu trúc thư mục

```
VRP_DRL_GNN/
├── PyTorch_GNN/
│   ├── run_gui.py          ← Giao diện so sánh (MỚI)
│   ├── plot.py             ← Vẽ model MHA (CLI)
│   ├── plot_light.py       ← Vẽ model GNN Light (CLI)
│   ├── train.py / train_light.py
│   ├── model.py / model_light.py
│   ├── Weights/            ← File .pt (model weights)
│   ├── Csv/                ← Log training
│   └── Pkl/                ← Hyperparameter configs
├── OpenData/               ← Dữ liệu chuẩn Augerat et al.
│   ├── A-n45-k7.txt
│   └── A-n53-k7.txt
└── README.md
```

---

## Giao diện so sánh (MỚI)

File `run_gui.py` cung cấp giao diện đồ họa để **chạy và so sánh nhiều model cùng lúc**.

### Chạy GUI

```bash
cd PyTorch_GNN
python run_gui.py
```

### Tính năng

- **Chọn tối đa 6 model** từ thư mục `Weights/` (tự động nhận diện Multi-Head Attention vs GNN Light)
- **Chọn 1 file dữ liệu** từ `OpenData/` hoặc duyệt file tùy chọn
- Tất cả model chạy cùng dữ liệu để so sánh công bằng
- **Tab "Bản đồ Routes"**: hiển thị đường đi từng model cạnh nhau
- **Tab "So sánh Chi phí"**: biểu đồ Best Cost / Avg Cost / Số tuyến + bảng tóm tắt
- Hỗ trợ CPU và GPU tự động

### Tham số GUI

| Tham số | Mô tả | Mặc định |
|---|---|---|
| Batch size | Số giải pháp sinh ra (chọn nghiệm tốt nhất) | 128 |
| Decode | `sampling` (đa dạng) hoặc `greedy` (nhanh) | sampling |

---

## Chạy từng model qua dòng lệnh

### Model MTH Attention

```bash
cd PyTorch_GNN
python plot.py -p Weights/VRP50_train_epoch19.pt -t ../OpenData/A-n45-k7.txt -d sampling -b 128
```

### Model GNN Light

```bash
cd PyTorch_GNN
python plot_light.py -p Weights/VRP20_train_GNN_Light_epoch19.pt -t ../OpenData/A-n45-k7.txt -d sampling -b 128
```

**Tham số:**

| Flag | Mô tả |
|---|---|
| `-p` | Đường dẫn file weights `.pt` (bắt buộc) |
| `-t` | File dữ liệu txt (không bắt buộc) |
| `-b` | Batch size |
| `-d` | Decode: `greedy` hoặc `sampling` |
| `-n` | Số khách hàng (chỉ dùng khi không có `-t`) |

---

## Training

### Bước 1: Sinh config

```bash
cd PyTorch_GNN
python config.py -n 20    # VRP20
python config.py -n 50    # VRP50
```

### Bước 2: Train

```bash
# GNN Attention
python train.py -p Pkl/VRP20_train.pkl

# GNN Light
python train_light.py -p Pkl/VRP20_train.pkl
python train_light.py -p Pkl/VRP50_train.pkl
```

---

## Dữ liệu chuẩn (OpenData)

Dữ liệu từ **Augerat et al. (1995)**:

| File | Khách hàng | Xe |
|---|---|---|
| `A-n45-k7.txt` | 44 | 7 |
| `A-n53-k7.txt` | 52 | 7 |

Nguồn: [NEO Research Group](https://neo.lcc.uma.es/vrp/vrp-instances/capacitated-vrp-instances/)

---

## Tham khảo

- [Kool et al., 2019 — Attention, Learn to Solve Routing Problems!](https://arxiv.org/pdf/1803.08475.pdf)
- [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)
- [d-eremeev/ADM-VRP](https://github.com/d-eremeev/ADM-VRP)
