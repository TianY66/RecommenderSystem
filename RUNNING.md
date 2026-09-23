# 推荐系统运行说明

本文档说明如何运行本仓库，并区分两个入口：

- **本地 CSV 演示（推荐先跑）**：读取 `data/*.csv`，加载仓库自带的模型权重，最终为示例用户输出推荐书籍 ID。
- **FastAPI + MySQL 服务（进阶集成）**：从两个 MySQL 数据库读取数据，对外提供推荐、微调和健康检查接口。

> 结论：仓库当前可以优先按“本地 CSV 演示”准备环境。FastAPI 版本不是只修改数据库密码就能直接运行，详见“FastAPI 服务”一节中的已知阻塞。

## 1. 运行前须知

### 1.1 项目规模

本地演示会读取：

- 10,000 个用户；
- 6,262 个物品；
- 94,934 条交互记录；
- 约 28 MB 的预训练模型权重。

初始化时，程序还会逐个访问 6,262 个封面 URL，并用 CLIP 提取图片和文本特征。因此第一次运行需要稳定的外网连接，耗时可能达到数十分钟，CPU 运行会更慢。

不能为了快速测试而直接截取 CSV 前几十行：模型词表大小会改变，随后加载预训练权重时会出现 `size mismatch`。

### 1.2 推荐环境

- Windows 10/11；
- Conda；
- Python 3.12（仓库的环境文件固定为 Python 3.12）；
- NVIDIA GPU，建议 8 GB 以上显存；
- 足够的磁盘空间用于 Conda 环境、PyTorch、CLIP 模型缓存；
- 能访问 GitHub、PyTorch 下载源、CLIP 模型下载地址和数据中的阿里云 OSS 图片地址。

代码会在 CUDA 不可用时退回 CPU，但完整处理 6,262 个物品会很慢。

## 2. 跑通本地 CSV 演示

以下命令均在 **Anaconda Prompt 或 PowerShell** 中执行。

### 2.1 进入仓库根目录

```powershell
# 在你克隆的 RecommenderSystem 仓库根目录执行后续命令
```

后续必须保持当前目录为仓库根目录，因为代码使用了 `data/...` 和 `model_weights/...` 相对路径。

### 2.2 创建 Conda 环境

优先使用仓库提供的完整环境文件：

```powershell
conda env create -f dependency\environment.yml
conda activate rs_project
python --version
```

预期 Python 版本为 3.12.x。

该 `environment.yml` 是 Windows 环境的完整导出，包含很多并非推荐主流程必需的包。如果精确构建失败，可先清理同名的失败环境，再采用最小环境：

```powershell
conda create -n rs_project python=3.12 -y
conda activate rs_project
python -m pip install pandas numpy scipy scikit-learn pillow requests
python -m pip install faiss-cpu
python -m pip install fastapi uvicorn sqlalchemy aiomysql pymysql
```

本项目只使用 FAISS 的 CPU 索引类，因而 `faiss-cpu` 可以满足代码中的 `import faiss`。如果已有可用的 `faiss-gpu`，也可以继续使用。

### 2.3 安装 PyTorch

应从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与显卡和驱动匹配的命令，不要机械照搬旧 CUDA 版本。

对于 RTX 50 系显卡，建议使用支持 CUDA 12.8 或更高版本的较新 PyTorch。例如官方仍提供该索引时，可执行：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安装后检查：

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`cuda: True` 表示程序会使用 GPU。若为 `False`，项目仍能启动，但 CLIP 特征提取会很慢。

### 2.4 安装 OpenAI CLIP

代码需要能够提供 `clip.load()` 和 `clip.tokenize()` 的 OpenAI CLIP 包。为避免安装到同名但不兼容的 PyPI 包，建议直接安装官方仓库版本：

```powershell
python -m pip uninstall -y clip
python -m pip install "git+https://github.com/openai/CLIP.git"
```

### 2.5 做依赖自检

```powershell
python -c "import torch, faiss, clip, pandas, sklearn, PIL, requests; print('核心依赖导入成功'); print('CUDA:', torch.cuda.is_available())"
```

如果要运行 FastAPI 入口，再检查：

```powershell
python -c "import fastapi, uvicorn, sqlalchemy, aiomysql; print('服务端依赖导入成功')"
```

### 2.6 检查封面网络访问

本地初始化依赖远程封面。至少先确认第一张图片能访问：

```powershell
Invoke-WebRequest -Method Head "https://haut-campus-knowledge-sharing-and-trading-platform.oss-cn-beijing.aliyuncs.com/book-images-2026/1.jpg"
```

如果此处超时或返回非 200，直接运行项目通常会在内容特征提取阶段失败。

### 2.7 启动本地演示

```powershell
python main.py
```

首次运行时会依次完成：

1. 下载 CLIP `ViT-B/32` 权重（本机未缓存时）；
2. 读取三个 CSV 文件；
3. 下载封面并计算 6,262 个内容特征；
4. 建立协同过滤、聚类和 FAISS 索引；
5. 加载双塔、LightGCN、三塔和多目标精排模型权重；
6. 为用户 `U00001` 在“15 点、周末、非节假日”场景下生成推荐。

成功时日志末尾会出现类似内容：

```text
recall finished
rough ranking finished
fine ranking finished
最终推荐的物品ID列表： [...]
recommend finished
```

示例用户和场景位于 `main.py` 的 `main()` 函数中。若需更换用户，应使用 `data/users_new.csv` 中真实存在的 `user_id`。

## 3. FastAPI + MySQL 服务

### 3.1 API 结构

服务入口为 `interface/main.py`，包括：

- `POST /recommend`：生成推荐；
- `POST /fine_tuning`：按时间范围查询新数据并微调；
- `GET /health`：健康检查；
- `/docs`：FastAPI 自动接口文档。

### 3.2 当前仓库的已知阻塞

在尝试启动服务前，需要先处理以下问题：

1. `sql/kl_trade.sql` 在 `user_item_interaction` 表中定义的是 `create_time`，但索引引用了不存在的 `datetime` 字段。需要将：

   ```sql
   KEY idx_datetime (datetime),
   KEY idx_user_item_time (user_id, item_id, datetime)
   ```

   改为：

   ```sql
   KEY idx_create_time (create_time),
   KEY idx_user_item_time (user_id, item_id, create_time)
   ```

2. `interface/database.py` 中仍是占位连接串，必须填写真实的 MySQL 用户名、密码、主机和数据库名。
3. SQL 文件只有 DDL，没有插入数据。空数据库无法初始化推荐器。
4. CSV 使用 `U00001`、`I0001` 形式的字符串 ID，SQL 表使用 `BIGINT`，不能原样导入。
5. 预训练权重与训练时的完整用户数、物品数、类别词表及编码顺序绑定。数据库数据仅“字段一致”还不够；数据规模或词表变化会使 `load_state_dict()` 报 `size mismatch`，或者产生错误的 ID 映射。
6. `interface/main.py` 当前引用旧的 `twin_towers_model.pth`，而现有双塔代码已经换成 DCN 结构。应改用 `../model_weights/improved_twin_towers_model.pth`。
7. `/recommend` 当前使用 `if not (user_id and hour and is_weekend and is_holiday)` 校验参数，会错误拒绝 `hour=0` 和布尔值 `false`。正式使用前应改为范围/空值校验。
8. 接口目录的导入写法混用了顶层导入和包导入；启动目录不正确时会出现 `ModuleNotFoundError`。

因此，仓库自带材料不足以直接完成数据库版端到端启动。若只是验证推荐算法，请先跑通 `python main.py`；数据库版应视为需要接入真实业务数据的二次开发入口。

### 3.3 数据库准备

修正上述 SQL 字段后，在 MySQL 8 中创建两个数据库：

```powershell
cmd /c "mysql -u root -p < sql\kl_trade.sql"
cmd /c "mysql -u root -p < sql\kl_user.sql"
```

之后必须向以下表导入相互匹配的数据：

- `kl_trade.book_info`；
- `kl_trade.item_interaction`；
- `kl_trade.user_item_interaction`；
- `kl_user.user_profile_behavior`。

如果希望直接使用仓库预训练权重，需要编写确定性的 CSV 转换/导入流程，将 `U00001`、`I0001` 转为整数，并确保排序、全集、类别取值与训练数据保持一致。否则应使用数据库中的完整数据重新训练模型，而不是加载仓库权重。

### 3.4 配置连接串

编辑 `interface/database.py`：

```python
BOOK_INTERACTION_DATABASE_URL = "mysql+aiomysql://用户名:密码@127.0.0.1:3306/kl_trade"
USER_DATABASE_URL = "mysql+aiomysql://用户名:密码@127.0.0.1:3306/kl_user"
```

密码中如果包含 `@`、`:`、`/` 等字符，需要先做 URL 编码。

### 3.5 启动服务

处理完数据库数据、权重和校验问题后，推荐从 `interface` 目录启动，使当前相对路径能够正确找到模型权重：

```powershell
# 当前目录是仓库根目录
Set-Location .\interface
$env:PYTHONPATH = (Resolve-Path "..").Path
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

启动阶段会读取全量数据库、下载封面、计算 CLIP 特征并建立索引，因此 `/health` 不会立刻可用。

浏览器访问：

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

推荐请求示例：

```powershell
$body = @{
    user_id = 1
    hour = 15
    is_weekend = $true
    is_holiday = $false
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/recommend" `
    -ContentType "application/json" `
    -Body $body
```

只有在修正参数校验后，示例中的 `$false` 才会被正确接受。

## 4. 常见错误

| 报错或现象 | 原因 | 处理方法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'clip'` | 未安装 OpenAI CLIP，或安装了同名错误包 | 按 2.4 节从 OpenAI GitHub 安装 |
| `ModuleNotFoundError: No module named 'faiss'` | FAISS 未安装 | 安装 `faiss-cpu`，或使用环境文件中的 FAISS |
| `Torch not compiled with CUDA enabled` | 安装了 CPU 版 PyTorch | 从 PyTorch 官方 CUDA 索引重新安装 |
| `no kernel image is available` | PyTorch/CUDA 版本不支持当前显卡架构 | RTX 50 系改用支持 CUDA 12.8+ 的较新 PyTorch |
| CLIP 下载失败 | 无法访问模型下载源或缓存文件损坏 | 检查代理/网络后重新下载 CLIP `ViT-B/32` |
| `UnboundLocalError: rgb_img` | 某个封面 URL 返回非 200，代码没有为失败图片准备回退图 | 检查该 URL；正式运行建议为失败请求增加超时、重试和默认图片 |
| `size mismatch for ...` | 当前数据的用户、物品或类别词表与预训练权重不一致 | 使用完整原始 CSV，或用当前数据重新训练全部模型 |
| 找不到 `model_weights` 或 `data` | 当前工作目录不正确 | 本地入口在仓库根目录运行；服务入口按 3.5 节运行 |
| MySQL `Unknown column 'datetime'` | DDL 的索引字段名写错 | 按 3.2 节改为 `create_time` |
| 请求中的 `false` 被判定为缺参 | `/recommend` 使用了 truthy 校验 | 改成显式空值判断和 `0 <= hour <= 23` 范围校验 |

## 5. 建议的验收顺序

1. 核心依赖全部可以导入；
2. `torch.cuda.is_available()` 符合预期；
3. 第一张 OSS 封面能访问；
4. `python main.py` 能输出最终推荐 ID；
5. 再修正和准备数据库版；
6. `/health` 返回 `healthy`；
7. `/recommend` 返回有序的 `recommendations` 数组；
8. 最后验证 `/fine_tuning`。

## 6. 本次检查结果

检查时仓库内 24 个 Python 文件均通过语法编译。当前机器具有 NVIDIA GeForce RTX 5070，CUDA 可被已有 PyTorch 环境识别，但现有 `pytorch` Conda 环境缺少 `faiss`、`clip`、`fastapi`、`sqlalchemy` 和 `aiomysql`，不能直接运行本项目，需要按本文补齐依赖。

本次没有声称数据库版已端到端运行成功：仓库没有数据库初始数据，且存在第 3.2 节列出的代码与 DDL 阻塞。本文给出的本地演示路径是当前最短、最可验证的跑通方式。
