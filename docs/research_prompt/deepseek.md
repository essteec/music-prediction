# 音乐信息检索数据集特征扩展计划


## 1. 执行摘要

**立即优先测试（Tier 0–1）：**

1. **MuQ（Mel Residual Vector Quantization）** — 当前最前沿的音乐自监督模型，在零样本音乐标注任务上达到SOTA。相比MERT v1 95M，MuQ采用Mel-RVQ预训练目标，可能编码互补的声学和音乐结构信息。95M参数量，与MERT规模相当，6GB VRAM可运行。

2. **LAION-CLAP music-specialized变体** — 多模态音乐-文本联合嵌入模型，在音色语义与人类感知对齐方面表现最佳。特别适合跨模态检索和基于自然语言描述的歌曲发现。

3. **结构化MIR描述符（节奏稳定性、节拍置信度、chroma/HPCP、MFCC统计、动态范围、响度曲线）** — 计算成本低（CPU即可），可解释性强，与黑箱嵌入互补，且可直接用于用户界面的"为什么相似"解释。

4. **BGE-M3歌词嵌入** — 多语言文本嵌入模型，支持100+语言，8192 token上下文。取代现有的MiniLM/MPNet，显著提升多语言歌词检索能力。

5. **歌词语言识别与清洗管线** — 低成本、高收益，是后续所有歌词处理的前提。

**坚决避免/推迟（Tier 3）：**

6. **Jukebox-derived embeddings** — 5B参数模型，6GB VRAM无法运行。已有研究表明其音频表示并未显著优于mel spectrogram。

7. **全词干分离（source separation）存储** — Demucs/Open-Unmix在6GB VRAM上处理10k首4分钟歌曲预计需要200+小时，且分离误差可能削弱价值。建议仅提取分离后聚合统计量而非存储波形。

8. **生成式LLM歌词标注（GPT-4/Claude级别）** — 10k首歌的API成本极高（约$500–$2000），存在版权和可重复性问题。可作为Tier 3探索性项目。

9. **AudioMAE** — 通用音频自监督模型，未针对音乐优化。与PANNs/MERT相比，音乐任务上预期无增量价值。

10. **BEATs** — 通用音频模型，主要用于语音和通用音频。在音乐特定任务上已被MuQ和MERT超越。


## 2. 当前状态审计

### 已有资产

| 类别 | 内容 | 状态 |
|---|---|---|
| 歌曲元数据 | 10,000行，32列 | ✅ 完整 |
| 音频文件 | 10,000个Opus/WebM，48kHz立体声 | ✅ 已下载 |
| VGGish | 128维，song-level | ✅ 已提取 |
| MERT v1 95M | 768维，24kHz mono，前30秒均值池化 | ✅ 已提取 |
| PANNs Cnn14 | 2,048维，32kHz mono | ✅ 已提取 |
| Mel谱统计 | 512维（128 bands × mean/std/max/min） | ✅ 已提取 |
| MiniLM lyric | 384维，前3000字符 | ✅ 已提取 |
| MPNet lyric | 768维，前3000字符，归一化 | ✅ 已提取 |
| 基础歌词统计 | 5维（词数、唯一词数等） | ✅ 已提取 |
| TextBlob情感 | 2维（polarity, subjectivity） | ✅ 已提取 |
| 歌词文本 | 9,797/10,000非空 | ✅ 已有 |

### 关键缺口

1. **音频层面**：缺乏时序/结构化特征（节奏变化、和声进行、段落结构）；缺乏多模态音频-文本联合嵌入；缺乏质量控制和副本检测机制。
2. **歌词层面**：当前仅使用通用句子嵌入（MiniLM/MPNet），缺乏多语言支持、歌词特定优化、结构化特征（押韵、主题）和情感轨迹。
3. **元数据层面**：未利用时间信息（发行年代）、未构建艺术家协作图、未计算音频-歌词一致性。
4. **评估层面**：无系统化的特征ablation协议；无相似性检索基准。


## 3. 候选矩阵：音频

| 候选方法 | 模型/版本 | 主要来源 | 编码属性 | 与现有特征互补性 | 6GB VRAM可行性 | 10k估算运行时 | 许可/分发 | 最佳用途 | 推荐 |
|---|---|---|---|---|---|---|---|---|---|
| **MuQ** | `OpenMuQ/MuQ-base` (95M) | Zhu et al., 2025 | Mel-RVQ自监督；音乐结构、零样本标注SOTA | 高 — 预训练目标(Mel-RVQ)与MERT的HuBERT风格互补 | ✅ 95M参数，batch size 8–16 | ~8–12h (GPU) | MIT-like | 音频相似性、预测建模 | **Tier 1** |
| **LAION-CLAP (music)** | `laion/larger_clap_music` (~200MB) | LAION-AI | 音乐-文本联合嵌入；音色语义对齐最佳 | 高 — 跨模态能力是现有纯音频模型不具备的 | ✅ 200MB，batch size 16–32 | ~6–10h (GPU) | MIT | 文本-音频检索、自然语言发现 | **Tier 1** |
| **MuQ-MuLan** | `OpenMuQ/MuQ-MuLan-large` (~700M) | Zhu et al., 2025 | CLIP-like音乐-文本联合；中英文支持；零样本标注SOTA | 高 — 跨模态，且为音乐专门优化 | ⚠️ 700M，需梯度检查点，batch size 2–4 | ~20–30h (GPU) | MIT-like | 文本-音频检索、零样本分类 | **Tier 2**(pilot) |
| **MusicFM** | ICASSP 2024 | 腾讯/ASLP | 30秒上下文；音乐结构理解强 | 中 — 类似MERT但更长上下文 | ✅ ~100M | ~10–15h (GPU) | 需确认 | 音乐结构分析 | **Tier 2**(pilot) |
| **MERT-v1-330M** | `m-a-p/MERT-v1-330M` | Li et al. | 同v1但更大 | 低 — 与现有MERT v1 95M高度冗余 | ⚠️ 330M，batch size 4–8 | ~15–20h | CC-BY-NC | 可能略优于95M | **推迟** |
| **BEATs** | `BEATs` (300M) | CMU-AIST | 通用音频自监督 | 低 — 通用音频，未针对音乐 | ⚠️ 300M | ~15h | 需确认 | 不推荐 | **拒绝** |
| **AudioMAE++** | AudioMAE++ | 2025 MLSP | 掩码自编码器音频 | 低 — 通用音频 | ⚠️ 需确认 | 未知 | 需确认 | 不推荐 | **拒绝** |
| **Jukebox** | OpenAI Jukebox | OpenAI | 4800维生成式编码 | 中 — 但规模过大 | ❌ 5B参数 | 不可行 | OpenAI非商业 | 不推荐 | **拒绝** |
| **结构化MIR** | librosa/essentia | 标准库 | 节奏、和声、音色、动态、结构 | 高 — 可解释、与黑箱互补 | ✅ CPU | ~5–10h (CPU) | MIT/BSD | 用户解释、质量控制 | **Tier 0** |
| **源分离特征** | Demucs v4 | Meta | 人声/鼓/贝斯/其他分离 | 中 — 分离质量受限 | ⚠️ 需~2GB VRAM | ~200+h | MIT | 人声专用分析 | **Tier 2**(pilot) |
| **时域池化改进** | 注意力/VLAD | 自定义 | 时序聚合 | 高 — 改进现有嵌入 | ✅ 轻量 | 低 | N/A | 所有任务 | **Tier 0** |


## 4. 候选矩阵：歌词

| 候选方法 | 模型/版本 | 主要来源 | 编码属性 | 多语言 | 6GB VRAM | 输出维度 | 许可/分发 | 最佳用途 | 推荐 |
|---|---|---|---|---|---|---|---|---|---|
| **BGE-M3** | `BAAI/bge-m3` | BAAI | 多语言、8192 token、稠密检索 | ✅ 100+语言 | ✅ ~1GB | 1,024 | MIT | 歌词检索、相似性 | **Tier 1** |
| **GTE-Qwen2-1.5B** | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 阿里 | 指令微调、MTEB领先 | ✅ 多语言 | ⚠️ 1.5B，需量化 | 未知 | 需确认 | 歌词检索 | **Tier 2**(pilot) |
| **XLM-RoBERTa** | `xlm-roberta-base` | Facebook | 多语言句子编码 | ✅ 100+语言 | ✅ ~500MB | 768 | CC-BY-NC | 多语言歌词 | **Tier 0**(baseline) |
| **E5-mistral** | `intfloat/e5-mistral-7b-instruct` | Microsoft | 指令微调、MTEB SOTA | ✅ 多语言 | ❌ 7B | 4,096 | MIT | 高质量检索 | **Tier 3** |
| **ALBERTI**(歌词领域适应) | ALBERTI | 领域适应 | 诗歌/歌词领域优化 | ⚠️ 西班牙语为主 | ✅ | 768 | 需确认 | 西班牙语歌词 | **推迟** |
| **MPNet**(已有) | `all-mpnet-base-v2` | 现有 | 通用句子嵌入 | ⚠️ 主要英语 | ✅ 已提取 | 768 | Apache 2.0 | 基线对比 | **保留** |
| **MiniLM**(已有) | `all-MiniLM-L6-v2` | 现有 | 通用句子嵌入 | ⚠️ 主要英语 | ✅ 已提取 | 384 | Apache 2.0 | 基线对比 | **保留** |
| **歌词清洗/语言ID** | `fasttext/langid` | 标准库 | 语言检测、清洗 | ✅ | ✅ CPU | N/A | MIT/BSD | 所有歌词任务前置 | **Tier 0** |
| **歌词韵律/押韵** | `pronouncing`/自定义 | 标准库 | 押韵密度、韵律特征 | ⚠️ 英语为主 | ✅ CPU | ~20维 | MIT | 歌词风格分析 | **Tier 0** |
| **歌词情感轨迹** | 基于BGE-M3分句 | 自定义 | 逐句情感、情绪变化 | ✅ | ✅ | 可变 | N/A | 歌词相似性、解释 | **Tier 1** |
| **LLM主题标注** | GPT-4o mini / Llama-3 | OpenAI/Meta | 主题、叙事、情绪 | ✅ | N/A (API) | 结构化标签 | 依赖API | 解释性特征 | **Tier 2**(pilot) |


## 5. 歌词基准答案

### 5.1 是否存在直接选择"最佳"歌词表示模型的基准？

**结论：不存在单一的、被广泛接受的、可直接用于选择歌词表示模型的基准。**

原因如下：

1. **任务多样性** — 歌词相似性、情感分类、主题建模、流派分类、歌词-音频对齐等任务对表示的要求不同。一个模型在检索任务上最优，在分类任务上可能不是。

2. **语言限制** — 大多数现有基准以英语为主。多语言歌词评估仍处于早期阶段。

3. **版权限制** — 完整歌词数据集难以公开分发，限制了基准的可重复性。

4. **评估维度不统一** — 有的评估检索（Recall@k），有的评估分类（F1），有的评估回归（RMSE），难以直接比较。

### 5.2 现有相关基准/数据集

| 基准/数据集 | 任务 | 语言 | 规模 | 可用性 | 相关性 |
|---|---|---|---|---|---|
| **LyricSIM** | 歌词语义相似性 | 西班牙语 | 2,775对 | 学术研究 | 高（相似性任务） |
| **MoodyLyrics** | 歌词情感分类 | 英语 | 四分类（Russell模型） | 研究可用 | 中（情感任务） |
| **DEAM** | 音乐情感（VA维度） | — | 744首 | 研究可用 | 中（多模态） |
| **CMI-Bench** | 音乐指令跟随 | 多语言 | 综合 | 2025年9月发布 | 中（综合MIR） |
| **WEALY** | 音频歌词匹配 | 多语言 | — | 2025年10月 | 中（音频-歌词对齐） |
| **MTEB/MMTEB** | 通用文本嵌入 | 多语言 | 综合 | 公开 | 中（间接参考） |
| **PoetryMTEB** | 诗歌/歌词检索 | 多语言 | 综合 | 2025年11–12月 | 高（诗歌/歌词领域） |

### 5.3 推荐的域内评估设计

鉴于缺乏直接适用的基准，建议构建专门的小规模评估集：

**A. 人工标注的歌词相似性（核心）**

- **采样策略**：从10k首歌中随机抽取200首作为查询。对每首查询，从语料库中采样50首候选（25首同流派+25首随机），确保包含艺术家重复控制。
- **标注任务**：3–5名标注者对每对(查询, 候选)从1（完全不相似）到5（非常相似）评分，基于"主题/叙事相似性"和"情感/氛围相似性"两个维度。
- **标注规模**：200 × 50 = 10,000对 × 2维度 × 3标注者 = 60,000个标注点。
- **可行性**：使用众包平台（如Prolific），预计成本$500–$1,000，2–3周完成。
- **评估指标**：标注者间一致性（Krippendorff's α），模型预测与平均人工评分的Spearman相关系数。

**B. 自动代理标签（辅助）**

- **流派一致性**：同一`main_genres`内的歌曲对作为"相似"正例，不同大类的作为负例。评估模型是否能检索同流派歌曲。
- **艺术家一致性**：同一艺术家的歌曲对（确保不用于训练）。
- **时间邻近性**：同年发行的歌曲对。

**C. 下游任务验证**

- 在100–200首的子集上标注歌词情感（4分类），比较不同嵌入模型+简单分类器的F1。
- 使用已有的Spotify音频特征作为辅助验证。

**D. 防泄漏措施**

- 确保同一艺术家的歌曲不同时出现在训练和评估中。
- 所有评估使用留出集，不用于任何模型选择或超参数调优。


## 6. 网站/产品概念（按优先级排序）

| 优先级 | 功能 | 用户需求 | 推荐表示 | 索引策略 | 10k可行性 | 1M扩展性 |
|---|---|---|---|---|---|---|
| **1** | **"找相似歌曲"多模态融合** | 发现新音乐 | MuQ + BGE-M3加权融合 | FAISS L2索引 | ✅ 可行 | ✅ 需量化 |
| **2** | **可调节相似性滑块** | 控制音频/歌词/节奏权重 | 结构化MIR + 嵌入 | 多FAISS索引 | ✅ 可行 | ⚠️ 需优化 |
| **3** | **"为什么相似"解释** | 理解推荐理由 | 结构化MIR特征 | 特征差异向量 | ✅ 可行 | ✅ 可行 |
| **4** | **音频-歌词情绪不一致浏览器** | 发现"反差感"歌曲 | 歌词情感 + 音频valence | 预计算差异 | ✅ 可行 | ✅ 可行 |
| **5** | **歌曲进化/时代地图** | 探索音乐演变 | MuQ + 发行年份 | UMAP + 时间轴 | ✅ 可行 | ✅ 可行 |
| **6** | **歌词主题地图** | 主题探索 | BGE-M3 + BERTopic | UMAP + 主题标签 | ✅ 可行 | ⚠️ 需优化 |
| **7** | **封面/翻唱/相似旋律检测** | 发现版本 | MuQ + fingerprint | 相似性搜索 | ✅ 可行 | ⚠️ 需优化 |
| **8** | **播放列表连贯性诊断** |  playlist优化 | 嵌入+结构化 | 集合统计 | ✅ 可行 | ✅ 可行 |

### 索引策略

- **当前（10k）**：全量FAISS索引（`IndexFlatIP`），内存~100MB，毫秒级查询。
- **未来（100k–1M）**：FAISS `IndexIVFPQ` + 量化，或hnswlib。UMAP仅用于可视化，不作为相似性引擎。


## 7. Kaggle就绪数据架构

```
data/
├── processed/
│   ├── songs.csv                    # 主元数据（不含原始歌词）
│   ├── songs_lyrics_cleaned.csv     # 清洗后歌词（仅本地，不发布）
│   └── songs_metadata_public.csv    # 公开发布版本（无歌词）
├── features/
│   ├── audio/
│   │   ├── muq_embeddings.npy       # (10000, 768) float32
│   │   ├── clap_embeddings.npy      # (10000, 512) float32
│   │   ├── mir_structured.npy       # (10000, 150) float32
│   │   └── manifests/
│   │       └── audio_features_manifest.json
│   ├── lyric/
│   │   ├── bge_m3_embeddings.npy    # (10000, 1024) float32
│   │   ├── lyric_linguistic.npy     # (10000, 50) float32
│   │   └── manifests/
│   └── metadata/
│       ├── derived_features.npy     # (10000, 30) float32
│       └── feature_names.json       # 所有特征列名
├── indexes/
│   ├── faiss_audio.index
│   ├── faiss_lyric.index
│   └── faiss_multimodal.index
├── visualizations/
│   ├── umap_audio.npy
│   ├── umap_lyric.npy
│   └── umap_metadata.csv
├── kaggle_public/                   # 可发布内容
│   ├── songs_metadata_public.csv    # 无歌词、无ISRC
│   ├── audio_embeddings_public.npy  # 仅允许分发的模型
│   ├── lyric_embeddings_public.npy  # 仅允许分发的模型
│   ├── feature_names.json
│   ├── data_dictionary.md
│   └── LICENSE.txt
└── logs/
    ├── extraction_timing.csv
    ├── extraction_failures.csv
    └── checksums.txt
```

### 不可分发内容
- 原始歌词文本（版权风险）
- 原始音频文件（版权风险）
- ISRC（隐私/商业敏感）
- 任何可直接重建音频或歌词的信息

### 版本控制
- 所有特征文件包含版本号（`v1`, `v2`）
- manifest记录：模型名称、版本、预处理参数、提取时间、checksum


## 8. 分阶段10k首歌路线图

### Tier 0：数据审计 + 几乎免费的特征（第1–2周）

| 项目 | 描述 | 估算时间 | 输出 |
|---|---|---|---|
| 歌词语言识别 | `fasttext`/`langid` | 1h CPU | 语言标签列 |
| 歌词清洗 | 括号去除、空白规范化 | 1h CPU | 清洗后歌词 |
| 歌词语言分布统计 | 多语言占比 | 0.5h | 报告 |
| 音频时长验证 | 与metadata对比 | 1h CPU | 质量报告 |
| 结构化MIR特征 | librosa: chroma, MFCC, onset, tempogram | 5–10h CPU | (10000, ~150) |
| 现有特征完整性检查 | NaN/Inf/形状验证 | 0.5h | 完整性报告 |

### Tier 1：最高价值、低风险（第2–4周）

| 项目 | 描述 | 估算时间 | 输出 | 决策标准 |
|---|---|---|---|---|
| **MuQ嵌入** | `OpenMuQ/MuQ-base` | 8–12h GPU | (10000, 768) | 与MERT相关性<0.85则保留 |
| **LAION-CLAP music** | `laion/larger_clap_music` | 6–10h GPU | (10000, 512) | 跨模态检索Recall@10 > 0.5 |
| **BGE-M3歌词嵌入** | 全歌词，分句池化 | 2–4h GPU | (10000, 1024) | 相似性检索优于MPNet |
| **歌词情感轨迹** | 基于BGE-M3逐句 | 2h GPU | (10000, 可变) | 情感方差>阈值 |
| **BGE-M3 vs MPNet pilot** | 100首歌人工评估 | 1周 | 对比报告 | BGE-M3胜出则全量 |

### Tier 2：试点后决定扩展（第5–8周）

| 项目 | 试点规模 | 试点时间 | 全量时间 | 成功标准 |
|---|---|---|---|---|
| **MuQ-MuLan-large** | 500首 | 3h GPU | 20–30h GPU | 零样本标注F1>0.7 |
| **MusicFM** | 500首 | 2h GPU | 10–15h GPU | 与MERT互补性>0.3 |
| **源分离特征** | 200首 | 4h GPU | 200h+ GPU | 分离质量MOS>3.5 |
| **LLM主题标注** | 200首 | $10–20 API | $500–2000 | 人工评估>0.7一致 |
| **时域注意力池化** | 全量 | 2h GPU | 2h GPU | 检索指标提升>5% |

### Tier 3：未来/成本过高（暂不执行）

- Jukebox（5B参数，不可行）
- E5-mistral-7B（7B参数，需量化或云端）
- 全词干波形存储（存储~500GB+）
- 大规模人工标注（>1,000首）


## 9. Ablation/评估协议

### 9.1 基线定义

**音频基线**：VGGish (128) + MERT (768) + PANNs (2048) + mel统计 (512) = 3,456维

**歌词基线**：MPNet (768) + MiniLM (384) + 5统计 + 2情感 = 1,159维

**元数据基线**：Spotify音频特征 (danceability, energy, etc.) = ~15维

### 9.2 评估任务

| 任务 | 指标 | 数据 | 说明 |
|---|---|---|---|
| **音频相似性检索** | Recall@10, nDCG@10 | 人工标注200查询 | 核心指标 |
| **歌词相似性检索** | Recall@10, nDCG@10 | 人工标注200查询 | 核心指标 |
| **多模态相似性** | Recall@10, nDCG@10 | 人工标注 | 融合策略对比 |
| **流派分类** | Macro F1 | `main_genres` | 监督任务代理 |
| **年份回归** | RMSE, R² | `release_date` | 时间信息编码 |
| **流行度预测** | RMSE, R² | `popularity` | ⚠️ 仅用于理解，不用于产品 |

### 9.3 拆分策略

- **分层随机拆分**：按`main_genres`分层，80/10/10
- **艺术家去重**：同一艺术家的所有歌曲在同一折内
- **时间拆分**：额外按年份拆分验证时间泛化
- **重复3次**：不同随机种子，报告均值±标准差

### 9.4 特征添加协议

每次添加一个特征族，记录增量提升：

1. 仅音频基线
2. 音频基线 + 新音频特征(X)
3. 仅歌词基线
4. 歌词基线 + 新歌词特征(Y)
5. 音频 + 歌词基线
6. 音频 + 歌词 + 新特征

### 9.5 统计显著性

- 使用paired bootstrap（10,000次重采样）计算95% CI
- 仅当p<0.05且效应量>阈值时视为显著改进


## 10. 风险登记

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| **歌词版权** | 高 — 无法公开分发 | 仅发布嵌入，不发布原文；使用清洗后非版权特征 |
| **模型许可** | 中 — CC-BY-NC禁止商业使用 | 优先MIT/Apache模型；标注许可状态 |
| **YouTube音频来源** | 高 — 版权问题 | 不发布音频；仅发布特征；添加来源说明 |
| **多语言偏差** | 中 — 英语模型对非英语歌词失效 | 使用BGE-M3等多语言模型；报告语言分层结果 |
| **流行度泄漏** | 高 — popularity不可用于预测 | 明确标记；从特征中排除用于预测 |
| **艺术家泄漏** | 中 — 同艺术家跨split | 艺术家级拆分；报告去重结果 |
| **提取失败** | 低 — 部分文件损坏 | 日志记录；重试机制；缺失值标记 |
| **嵌入版本漂移** | 低 — 模型更新后不一致 | 固定模型版本；保存模型hash |
| **GPU OOM** | 中 — 6GB VRAM限制 | batch size=1；梯度检查点；CPU fallback |
| **存储增长** | 低 — 当前<10GB | 监控；清理中间文件 |


## 11. 参考文献

### 音频模型

1. **MERT**: Li et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," 2023. [HuggingFace](https://huggingface.co/m-a-p/MERT-v1-95M) . License: CC-BY-NC.

2. **MuQ**: Zhu et al., "MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization," 2025. [Paper](https://ieeexplore.ieee.org/document/10687389). [GitHub](https://github.com/tencent-ailab/MuQ).

3. **MuQ-MuLan**: Joint music-text embedding, SOTA on MagnaTagATune zero-shot tagging. [HuggingFace](https://huggingface.co/OpenMuQ/MuQ-MuLan-large).

4. **LAION-CLAP**: [GitHub](https://github.com/LAION-AI/CLAP). Music-specialized variant `music_audioset_epoch_15_esc_90.14`. License: MIT.

5. **MusicFM**: "A Foundation Model for Music Informatics," ICASSP 2024.

6. **BEATs**: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers," ICML 2023.

7. **AudioMAE++**: Yadav et al., "AudioMAE++: Learning Better Masked Audio Representations with Swiglu FFNS," MLSP 2025.

8. **Jukebox**: Dhariwal et al., "Jukebox: A Generative Model for Music," OpenAI, 2020.

### 歌词/文本模型

9. **BGE-M3**: Chen et al., "BGE-M3: Retrieval, Reranking, and Multi-lingual," BAAI, 2024. License: MIT.

10. **GTE-Qwen2-1.5B**: Alibaba, [HuggingFace](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct).

11. **XLM-RoBERTa**: Conneau et al., "Unsupervised Cross-lingual Representation Learning," 2019.

12. **MTEB/MMTEB**: Muennighoff et al., "MTEB: Massive Text Embedding Benchmark," 2022; Enevoldsen et al., "MMTEB," ICLR 2025.

### 歌词基准/数据集

13. **LyricSIM**: Benito-Santos et al., "LyricSIM: A Dataset and Benchmark for Similarity Detection in Song Lyrics," 2025.

14. **MoodyLyrics**: "MoodyLyrics: A Benchmark for Lyrics-based Music Emotion Recognition".

15. **CMI-Bench**: "CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following," ISMIR 2025.

16. **WEALY**: "Leveraging Whisper Embeddings for Audio-based Lyrics Matching," 2025.

17. **PoetryMTEB**: Poetry-focused MTEB extension.

### 工具

18. **librosa**: McFee et al., "librosa: Audio and Music Signal Analysis in Python," 2015.

19. **essentia**: Bogdanov et al., "Essentia: Audio Analysis Library for Music Information Retrieval," 2013.

20. **Demucs**: Défossez et al., "Music Source Separation in the Waveform Domain," 2019. [GitHub](https://github.com/facebookresearch/demucs).

21. **FAISS**: Johnson et al., "Billion-scale Similarity Search with GPUs," 2019.