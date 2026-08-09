# AI (Model & Authentication)

本資料夾包含針對 Force-Matrix 感測資料的資料載入、前處理、分類訓練，以及基於距離的生物特徵驗證實驗程式。

重要原則：本 README 的內容完全基於資料夾內的實際程式碼實作；未在程式碼中確認的細節已標註為「需要確認」。

## Directory Structure
- `data_loader.py` : 讀取單一 CSV 與整個資料集的工具。
- `preprocess.py` : 資料正規化、展平與 train/test 切分邏輯。
- `prepare_dataset.py` : 與 `data_loader`/`preprocess` 類似的資料準備腳本（範例流程）。
- `model.py` : 簡單的檢視腳本（列印欄位值）。
- `cross_validation_knn.py` : 以 KNN 在整個 dataset 做 k 值交叉驗證的腳本。
- `train_knn.py` : KNN 分類模型的訓練與評估腳本（使用 flatten features）。
- `train_random_forest.py` : RandomForest 分類訓練與評估腳本。
- `authentication/` : 生物辨識（registration、template、threshold、驗證、評估與實驗）相關模組與實驗腳本。
  - `authentication/feature_extractor.py` : 工程化特徵與 hybrid feature 的產生。
  - `authentication/knn_auth.py` : 基於 pairwise distance 的 KNNAuthenticator（回傳距離，不做 threshold 判斷）。
  - `authentication/preprocess_auth.py` : 將 dataset 轉為 authentication 所需資料流（user / impostor split、flatten、normalization）。
  - `authentication/registration.py` : 註冊流程；建立 centroid template 與 per-user threshold，並可儲存成檔案。
  - `authentication/authentication.py` : 使用 template + threshold 做單一 sample 的驗證（accept/reject）。
  - `authentication/template.py` : `UserTemplate` 與 `TemplateManager`，template 的序列化/反序列化（JSON）。
  - `authentication/threshold.py` : `UserThreshold`、`ThresholdManager` 與 `compute_threshold_from_distances`（mean + k*std）。
  - 其他實驗/評估腳本：`evaluate_auth.py`, `evaluate_threshold.py`, `analyze_distances.py`, `validation_suite.py`, `demo_registration_auth.py` 等。

## Pipeline (資料從輸入到輸出)
1. 資料載入：`data_loader.load_csv()` 讀取單一 CSV（見 Input 格式），`load_dataset()` 遞迴讀取資料夾內所有 CSV。
2. 前處理：`preprocess.preprocess_dataset()` 或 `authentication.preprocess_auth.prepare_authentication_data()` 負責 label encoding、train/test 切分、以及依訓練資料計算 `sensor_min` / `sensor_max` 進行 normalization。
3. 特徵擷取：`authentication.feature_extractor` 提供：
   - `extract_frame_features`（7 維工程化特徵）
   - `extract_feature_combination`（可把原始 16 維與工程化特徵合併）
   - `prepare_feature_vectors`（回傳 normalized 並展平的向量）
4. 註冊（建立 template 與 threshold）：`authentication.registration.RegistrationSystem.register_user()` 以 registration samples 建立 centroid template、計算每筆至 centroid 的距離，並以 `mean + k * std` 決定 threshold。
5. 驗證：`authentication.authentication.AuthenticationSystem.authenticate()` 使用 template（含 `sensor_min`/`sensor_max`）將 sample normalize、展平後計算 Euclidean distance，並以 threshold 判定 accept/reject。
6. 實驗輸出：訓練腳本與評估腳本會以文字輸出（print），而 template / threshold 可由 `TemplateManager` / `ThresholdManager` 保存在 `templates/` 與 `thresholds/`（JSON）。

## Modules（主要模組說明）
- `data_loader.py`
  - `load_csv(file_path)` : 讀取單一 CSV，回傳 `(data, user_id)`，其中 `data` 為 ndarray，預期為 `(50,16)`。
  - `load_dataset(data_dir)` : 遞迴讀取 `*.csv`，回傳 `(X, raw_y)`，X 形狀為 `(N,50,16)`。

- `preprocess.py`
  - `flatten_samples(X)` : 將 (N,50,16) 轉成 (N,800)。
  - `normalize_with_train(X_train, X_test)` : 以 train 計算 `sensor_min`/`sensor_max`，並套用 normalization。
  - `preprocess_dataset(X, raw_y, ...)` : 完整 pipeline（label encode、split、normalize），回傳 train/test 與 encoder 與 sensor min/max。

- `feature_extractor.py` (authentication)
  - `extract_frame_features(frame, contact_threshold=1000.0)` : 由 16 維 single frame 產生 7 維工程化特徵（pressure_sum, max_pressure, contact_area, cop_x, cop_y, left_right_ratio, top_bottom_ratio）。
  - `extract_feature_combination(...)`, `extract_hybrid_features(...)` : 合成原始與工程化特徵。
  - `prepare_feature_vectors(...)` : 回傳 `(flattened_vectors, normalized_features, sensor_min, sensor_max)`。

- `knn_auth.py` (authentication)
  - `KNNAuthenticator` : `fit(X_user)` 設定 template（多筆訓練向量），`compute_distance(X_query)` 回傳每筆 query 的最小 Euclidean 距離（使用 sklearn.metrics.pairwise_distances(metric='euclidean')）。

- `registration.py` (authentication)
  - `RegistrationSystem.register_user(registration_samples, user_id)` : 建立 centroid template、可回傳 distances 並儲存 template / threshold（使用 `compute_threshold_from_distances`，預設 k=2.0）。

- `template.py` / `threshold.py` (authentication)
  - `UserTemplate` / `UserThreshold` 與對應的 Manager 類別提供 JSON 存取 (`templates/*.json`, `thresholds/*.json`)。

- 其他實驗腳本（可直接執行）：
  - `cross_validation_knn.py`：KNN cross validation（scikit-learn）。
  - `train_knn.py`, `train_random_forest.py`：模型訓練與評估（打印 accuracy / confusion matrix / classification report）。
  - `evaluate_auth.py`, `evaluate_threshold.py`, `analyze_distances.py`, `validation_suite.py`：驗證 / 門檻掃描 / 距離統計 / few-shot 實驗等。

## Input / Output
- Input CSV 格式（從程式碼可確認的部分）：
  - 必須包含 `label` 欄位，以及 `value_0` .. `value_15` 共 16 個 sensor 欄位。
  - 單一 CSV 被當成一個樣本，程式碼期望該檔案讀取後 `data.shape == (50, 16)`（需要確認：是否每個 CSV 都確實為 50 列）。
  - `data_loader.load_dataset()` 會跳過 shape 不符的 CSV 並打印提示。

- Output / artifacts：
  - 訓練/評估腳本會在 stdout 印出 accuracy / confusion matrix / statistics。
  - `TemplateManager.save_template()` 會在 `templates/`（或指定 storage_dir）下產生 `{user_id}.json`。
  - `ThresholdManager.save_threshold()` 會在 `thresholds/`（或指定 storage_dir）下產生 `{user_id}.json`。

## Usage (可直接執行的 command)
注意：這些 script 大多以 project workspace root 為相對路徑呼叫 `../dataset`，請在 repository root（含 `dataset/` 的上層）執行以下命令。

範例（在 repository root）：
```
python AI/cross_validation_knn.py
python AI/train_knn.py
python AI/train_random_forest.py

python AI/authentication/evaluate_auth.py
python AI/authentication/evaluate_threshold.py
python AI/authentication/analyze_distances.py
python AI/authentication/validation_suite.py
python AI/authentication/demo_registration_auth.py
```

備註：某些腳本會在檔頭插入 `AI` 到 `sys.path`，以確保相對 import 正常；若碰到 ModuleNotFoundError，請確認當前工作目錄與 Python path 設定。

## Configuration
- Normalization
  - 由 `preprocess.normalize_with_train()` 與 `feature_extractor.prepare_feature_vectors()` 決定：`sensor_min` 與 `sensor_max` 會從訓練資料（或 registration samples）計算而來，並用於後續 samples 的 scale。程式碼會避免 range=0 的 division-by-zero（將 range==0 設為 1）。

- Distance metric
  - authentication 與 KNN 計算均使用 Euclidean distance：
    - `KNNAuthenticator.compute_distance()` 使用 `pairwise_distances(..., metric='euclidean')`。
    - `AuthenticationSystem.authenticate()` 使用 `np.linalg.norm`（L2 範數）計算 sample 與 template 之間距離。

- Threshold 設定
  - 門檻由 `compute_threshold_from_distances(distances, k_value)` 計算：`mean + k * std`。
  - `k_value` 預設為 `2.0`（在 `RegistrationSystem` 與 `UserThreshold` 中預設）。

- Feature / Contact threshold
  - 工程化特徵 `contact_threshold` 預設為 `1000.0`（`feature_extractor.DEFAULT_CONTACT_THRESHOLD`），用於計算 contact_area 等特徵。

## Experiments (目前實作)
- KNN cross-validation for classification: `cross_validation_knn.py`（vary k, report accuracy）。
- Supervised classifiers: `train_knn.py` (KNN), `train_random_forest.py` (RandomForest)，輸出 accuracy / confusion matrix / classification report。
- Authentication / Verification evaluations:
  - `evaluate_auth.py`：列印 genuine / impostor 距離摘要。
  - `evaluate_threshold.py`：在一系列 threshold 下計算 FAR/FRR，並尋找 FAR+FRR 最小的 threshold。
  - `analyze_distances.py`：註冊 template 並輸出 genuine / impostor 距離統計 (min/mean/std/quantiles)，以及接受率。
  - `validation_suite.py`：包含 per-user threshold 評估、registration few-shot 實驗、以及模板對其他使用者的驗證結果列印。
  - `demo_registration_auth.py`：示範註冊並用單一 sample 做一次驗證（示範用）。

  ### Experiments Files (詳細清單)
  下面列出 `authentication/experiments/` 及其子目錄中實際存在的檔案，並根據程式碼說明其用途（內容皆可於對應檔案確認）：

  - `authentication/experiments/__init__.py` : package 初始化。
  - `authentication/experiments/common.py` : 共用 helper（資料載入、向量準備、距離計算、FAR/FRR 與結果格式化）。
  - `authentication/experiments/benchmark.py` : 組合不同實驗（distance/template/threshold）並輸出彙總結果。
  - `authentication/experiments/experiment_distance.py` : 比較 distance metric（Euclidean vs Cosine）並計算 FAR/FRR。
  - `authentication/experiments/experiment_template.py` : 比較 template 策略（Centroid / Robust / Median）並評估其效能。
  - `authentication/experiments/experiment_threshold.py` : 比較 threshold 策略（Mean + k*Std 與 threshold sweep）。
  - `authentication/experiments/experiment_zscore_cosine.py` : 實作 Z-score 正規化 + Cosine 距離的複合 pipeline，並提供專屬 template manager（`ZScoreCosineTemplateManager`）。
  - `authentication/experiments/run_benchmark.py` : CLI 入口，執行完整 benchmark suite（有 argparse 支援 `--users`, `--registration-count`, `--k-value`）。
  - `authentication/experiments/run_template_benchmark.py` : 針對 template 策略的 runner（可呼叫 `experiment_template`）。
  - `authentication/experiments/run_normalization_benchmark.py` : 針對 normalization 策略的 runner（對 `normalization_benchmark` 子目錄內容進行實驗）。
  - `authentication/experiments/run_zscore_cosine_benchmark.py` : 執行 Z-score + Cosine pipeline 的 runner。

  #### `authentication/experiments/normalization_benchmark/` 子目錄
  - `authentication/experiments/normalization_benchmark/__init__.py` : 子 package 初始化。
  - `authentication/experiments/normalization_benchmark/common.py` : normalization 實驗的共用工具（minmax / zscore / robust normalization、ROC 計算、zero-range 分析）。
  - `authentication/experiments/normalization_benchmark/normalization_roc_euclidean.png` : 實驗產生的示意圖（已存在的 artifact）。
  - `authentication/experiments/normalization_benchmark/normalization_roc_cosine.png` : 實驗產生的示意圖（已存在的 artifact）。
  - `authentication/experiments/normalization_benchmark/report.md` : 實驗報告（現存檔案，內容請參閱該檔）。

  #### `authentication/experiments/template_benchmark/` 子目錄
  - `authentication/experiments/template_benchmark/__init__.py` : 子 package 初始化。
  - `authentication/experiments/template_benchmark/common.py` : template benchmark 共用函數（若有，請參閱檔案）。
  - `authentication/experiments/template_benchmark/benchmark.py` : 產生 template-specific 圖表/報表的程式（有輸出圖片 artifact）。
  - `authentication/experiments/template_benchmark/template_roc.png` : 實驗產生的 ROC 圖片（artifact）。
  - `authentication/experiments/template_benchmark/template_distance_distribution.png` : 實驗產生的距離分布圖（artifact）。


## Notes / Limitations / 需要確認的事項
- 資料格式假設：程式碼中明確檢查 `data.shape == (50, 16)`；但是否所有 CSV 一律包含 50 列需由資料來源確認（標註為「需要確認」）。
- CLI 參數支援：多數腳本以硬編碼常數 (e.g., `authorized_user='amber'`, DATA_DIR = parent/"dataset") 運行，並未實作 argparse CLI 參數；若要靈活指定 user 或 data path，需改寫腳本（標註為「需要確認/擴充」）。
- 實驗 reproducibility：隨機相關的 `random_state` 在部分函式有使用（如 `train_test_split`），但部分流程（例如某些腳本的 sample 選取）仍需確認是否一致。
- I/O 路徑：`TemplateManager`、`ThresholdManager` 預設儲存在相對路徑 `templates/` 與 `thresholds/`。某些 demo/測試會將 storage_dir 指定在 `AI/authentication/templates` 等位置。
- 支援的距離/度量：目前僅實作 Euclidean；若需其他距離（cosine、mahalanobis），需要擴充。
