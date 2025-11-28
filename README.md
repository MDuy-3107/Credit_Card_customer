# Credit Card Customer Churn Prediction

Dự án phân tích và dự đoán churn (khách hàng rời bỏ dịch vụ) cho ngân hàng thẻ tín dụng. Toàn bộ quy trình xử lý dữ liệu và mô hình Logistic Regression được **triển khai hoàn toàn bằng NumPy** (không sử dụng scikit-learn cho phần core), nhằm hiểu sâu về thuật toán Machine Learning từ cơ bản.

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

## Giới thiệu

### Mô tả bài toán

Dự đoán khả năng một khách hàng sẽ rời bỏ (churn) dịch vụ thẻ tín dụng dựa trên các đặc trưng hành vi giao dịch và thông tin nhân khẩu học. Bài toán được mô hình hóa như bài toán **phân loại nhị phân**:

```math
y \in \{0, 1\}
```

- `y = 0`: Existing Customer (khách hàng còn sử dụng dịch vụ)
- `y = 1`: Attrited Customer (khách hàng đã rời bỏ dịch vụ)

### Động lực và ứng dụng thực tế

- **Giảm chi phí**: Chi phí giữ chân khách hàng cũ thấp hơn nhiều so với tìm kiếm khách hàng mới.
- **Tăng doanh thu**: Khách hàng trung thành mang lại giá trị lâu dài và ổn định.
- **Ứng dụng thực tế**:
  - Hệ thống cảnh báo sớm để xác định khách hàng có nguy cơ cao rời bỏ dịch vụ
  - Thiết kế chương trình chăm sóc, ưu đãi nhắm mục tiêu
  - Phân tích nguyên nhân churn để cải thiện sản phẩm/dịch vụ

### Mục tiêu cụ thể

1. **Xây dựng pipeline xử lý dữ liệu hoàn chỉnh bằng NumPy**:
   - Xử lý missing values
   - Feature engineering
   - Chuẩn hóa dữ liệu (Standardization)
   - One-hot encoding cho categorical features
   - Xử lý imbalanced data bằng undersampling

2. **Triển khai Logistic Regression từ đầu bằng NumPy**:
   - Implement sigmoid function
   - Binary cross-entropy loss
   - Gradient descent optimization
   - Tính toán metrics (Accuracy, Precision, Recall, F1-score)

3. **Phân tích và đánh giá mô hình**:
   - So sánh metrics với các threshold khác nhau
   - Vẽ confusion matrix
   - Phân tích loss curve

---

## Dataset

### Nguồn dữ liệu

- **File gốc**: `data/raw/BankChurners.csv`
- **Nguồn**: [Credit Card Customers Dataset from Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Kích thước**: 10,127 samples × 23 columns
- **Phiên bản đã xử lý**: Lưu trong `data/processed/` dưới dạng `.npy` files:
  - `X_train.npy`, `y_train.npy`: Training set (balanced bằng undersampling)
  - `X_test.npy`, `y_test.npy`: Test set (giữ nguyên phân bố gốc)
  - `X_train_full.npy`, `y_train_full.npy`: Training set đầy đủ (chưa balance)

### Mô tả các features

Dữ liệu gồm 19 features (cột 2-20 trong CSV) và 1 target (Attrition_Flag):

**Categorical Features (5 cột)**:
- `Gender`: M/F
- `Education_Level`: Doctorate, Graduate, High School, etc.
- `Marital_Status`: Married, Single, Divorced, Unknown
- `Income_Category`: Less than $40K, $40K - $60K, $60K - $80K, $80K - $120K, $120K +, Unknown
- `Card_Category`: Blue, Silver, Gold, Platinum

**Numeric Features (14 cột)**:
- `Customer_Age`: Tuổi khách hàng
- `Dependent_count`: Số người phụ thuộc
- `Months_on_book`: Thời gian là khách hàng (tháng)
- `Total_Relationship_Count`: Tổng số sản phẩm đang sử dụng
- `Months_Inactive_12_mon`: Số tháng không hoạt động trong 12 tháng gần nhất
- `Contacts_Count_12_mon`: Số lần liên hệ trong 12 tháng
- `Credit_Limit`: Hạn mức tín dụng
- `Total_Revolving_Bal`: Tổng số dư xoay vòng
- `Avg_Open_To_Buy`: Hạn mức còn lại trung bình
- `Total_Amt_Chng_Q4_Q1`: Tỷ lệ thay đổi số tiền giao dịch (Q4 vs Q1)
- `Total_Trans_Amt`: Tổng số tiền giao dịch trong 12 tháng
- `Total_Trans_Ct`: Tổng số giao dịch trong 12 tháng
- `Total_Ct_Chng_Q4_Q1`: Tỷ lệ thay đổi số lượng giao dịch (Q4 vs Q1)
- `Avg_Utilization_Ratio`: Tỷ lệ sử dụng tín dụng trung bình

**Target**:
- `Attrition_Flag`: Existing Customer (0) / Attrited Customer (1)

### Kích thước và đặc điểm dữ liệu

- **Tổng số mẫu**: 10,127 records
- **Class imbalance**: 
  - Existing Customer: ~83.93% (8,500 samples)
  - Attrited Customer: ~16.07% (1,627 samples)
- **Missing values**: Một số giá trị "Unknown" trong categorical features
- **Đặc điểm**: Dữ liệu hỗn hợp (mixed) gồm numeric và categorical, cần preprocessing kỹ lưỡng

### Phát hiện quan trọng từ EDA

Từ notebook `01_data_exploration.ipynb`, kiểm định t-test cho thấy:
- **Mean Total_Trans_Amt (Churn)**: 3,095.03
- **Mean Total_Trans_Amt (Non-churn)**: 4,654.66
- **t-statistic**: -22.69
- **Kết luận**: Khách hàng churn có tổng số tiền giao dịch thấp hơn **có ý nghĩa thống kê** (p < 0.05)

---

## Method

### Quy trình xử lý dữ liệu (Preprocessing Pipeline)

Toàn bộ pipeline được triển khai trong `src/data_processing.py` và `notebooks/02_preprocessing.ipynb`:

#### 1. Load dữ liệu bằng NumPy
```python
def load_data(csv_path: str):
    # Đọc header
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    
    # Load dữ liệu bằng np.genfromtxt
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
    return header, raw
```

#### 2. Xử lý Missing Values
- **Categorical features**: Thay thế "Unknown" và "" bằng **mode** của cột
```python
def fill_unknown_with_mode(col: np.ndarray) -> np.ndarray:
    mask_unknown = (col == "") | (col == "Unknown")
    vals, counts = np.unique(col[~mask_unknown], return_counts=True)
    mode_val = vals[np.argmax(counts)]
    col_filled[mask_unknown] = mode_val
    return col_filled
```

#### 3. Feature Engineering
Tạo thêm 3 features mới từ features có sẵn:
- `Amt_per_Trans = Total_Trans_Amt / Total_Trans_Ct`: Số tiền trung bình mỗi giao dịch
- `Inactive_Ratio = Months_Inactive_12_mon / Months_on_book`: Tỷ lệ thời gian không hoạt động
- `Rel_per_Month = Total_Relationship_Count / Months_on_book`: Số sản phẩm trung bình mỗi tháng

#### 4. Chuẩn hóa (Standardization)
Áp dụng **Z-score standardization** cho tất cả numeric features:

```math
x' = \frac{x - \mu}{\sigma}
```

```python
def standardize(x: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    x_std = (x - mean) / std
    return x_std, mean, std
```

#### 5. One-Hot Encoding
Chuyển đổi 5 categorical features thành one-hot vectors bằng NumPy:
```python
def one_hot_encode(col: np.ndarray):
    classes, inv = np.unique(col, return_inverse=True)
    N = col.shape[0]
    K = classes.shape[0]
    one_hot = np.zeros((N, K), dtype=float)
    one_hot[np.arange(N), inv] = 1.0
    return one_hot, classes
```

#### 6. Train/Test Split
- Split: 80% train, 20% test (random seed = 42)
- Sử dụng `np.random.permutation` để shuffle

#### 7. Xử lý Imbalanced Data
Áp dụng **Undersampling** cho class 0 (Existing Customer) với ratio = 1.0:
```python
def undersample(X, y, ratio=1.0):
    # Giữ toàn bộ class 1 (churn)
    # Random sample class 0 sao cho n0 = n1 * ratio
    X_bal = np.vstack([X1, X0_selected])
    y_bal = np.concatenate([y1, y0_selected])
    return shuffle(X_bal, y_bal)
```

**Kết quả**: Train set balanced có số lượng class 0 ≈ class 1

---

### Thuật toán: Logistic Regression (NumPy Implementation)

#### Công thức toán học

**1. Sigmoid Function**
```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

**2. Model Prediction**
```math
\hat{y} = P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)
```

**3. Binary Cross-Entropy Loss**
```math
L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
```

**4. Gradient Descent Update**

Gradients:
```math
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})
```

```math
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
```

Update rules:
```math
\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}
```

```math
b \leftarrow b - \eta \frac{\partial L}{\partial b}
```

---

### Cách implement bằng NumPy

**Vectorized Implementation** (từ `src/models.py`):

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)  # Tránh overflow
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegressionNumpy:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.loss_history = []
    
    def fit(self, X, y, verbose=True):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for i in range(self.n_iter):
            # Forward pass: logits = X @ w + b
            logits = np.einsum('ij,j->i', X, self.w) + self.b
            y_pred = sigmoid(logits)
            
            # Compute gradients
            error = y_pred - y
            dw = np.einsum('ij,i->j', X, error) / n_samples
            db = np.mean(error)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Track loss
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
    
    def predict(self, X, threshold=0.5):
        logits = np.einsum('ij,j->i', X, self.w) + self.b
        proba = sigmoid(logits)
        return (proba >= threshold).astype(int)
```

**Ưu điểm của implementation này**:
- Sử dụng `np.einsum` để tính toán hiệu quả
- Clip logits để tránh numerical overflow
- Vectorized hoàn toàn, không có Python loops trong tính gradient
- Track loss history để visualize quá trình training

**Metrics được tính bằng NumPy**:
```python
def classification_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    
    return accuracy, precision, recall, f1, (tp, tn, fp, fn)
```

---

## Installation & Setup

### Yêu cầu môi trường

- **Python**: 3.8+ (khuyến nghị 3.9 hoặc 3.10)
- **Dependencies**: 
  - `numpy`: Core library cho toàn bộ xử lý dữ liệu và model
  - `matplotlib`: Visualization
  - `seaborn`: Statistical data visualization
  - `scikit-learn`: (Chỉ dùng cho so sánh, không dùng trong core implementation)

### Cài đặt

**1. Clone repository**
```powershell
git clone <repository-url>
cd "Credit Card customer"
```

**2. Tạo virtual environment (khuyến nghị)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Cài đặt dependencies**
```powershell
pip install -r requirements.txt
```

Nội dung `requirements.txt`:
```
numpy
matplotlib
seaborn
scikit-learn
```

**4. (Tùy chọn) Cài đặt Jupyter Lab để chạy notebooks**
```powershell
pip install jupyterlab
```

---

## Usage

### Hướng dẫn chạy từng phần

Project được tổ chức thành 3 notebooks chính, chạy theo thứ tự:

#### **1. Data Exploration** (`notebooks/01_data_exploration.ipynb`)

Khám phá và phân tích dữ liệu:
- Load và kiểm tra cấu trúc dữ liệu
- Phân tích phân bố target (class imbalance)
- Thống kê mô tả các features numeric và categorical
- Visualization: histograms, pie charts
- **Hypothesis Testing**: Kiểm định t-test cho `Total_Trans_Amt` giữa churn vs non-churn

**Chạy**:
```powershell
jupyter lab
# Mở notebooks/01_data_exploration.ipynb và chạy các cells
```

#### **2. Preprocessing** (`notebooks/02_preprocessing.ipynb`)

Tiền xử lý dữ liệu hoàn chỉnh bằng NumPy:
- Load raw data
- Xử lý missing values (fill Unknown với mode)
- Feature engineering (3 features mới)
- Standardization (Z-score)
- One-hot encoding cho categorical features
- Train/test split (80/20)
- Undersampling để balance training set
- Lưu processed data vào `data/processed/*.npy`

**Output**: 
- `X_train.npy`, `y_train.npy` (balanced)
- `X_test.npy`, `y_test.npy`
- `X_train_full.npy`, `y_train_full.npy` (unbalanced)

#### **3. Modeling & Analysis** (`notebooks/03_modeling.ipynb`)

Phân tích theo phương pháp **Research Question-Driven**:

**Phần 1: Đặt câu hỏi nghiên cứu**
- Câu hỏi 1: Những yếu tố nào quan trọng nhất trong việc dự đoán khách hàng churn?
- Câu hỏi 2: Mô hình có thể dự đoán chính xác bao nhiêu % khách hàng churn?
- Câu hỏi 3: Threshold nào cân bằng tốt nhất giữa phát hiện churn và giảm false alarms?
- Câu hỏi 4: Với chi phí mất khách hàng là $500 và chi phí chăm sóc là $50, threshold nào tiết kiệm chi phí nhất?

**Phần 2: Thực hiện Modeling**
- Load processed data
- Train model với `lr=0.01`, `n_iter=1500`
- Visualize loss curve
- Đánh giá với nhiều thresholds (0.3, 0.4, 0.5)
- Vẽ confusion matrix

**Phần 3: Trả lời câu hỏi nghiên cứu**
- Phân tích mức độ ảnh hưởng của đặc trưng từ model weights
- So sánh metrics kết quả với threshold = 0.5
- Tối ưu threshold theo metrics và chi phí kinh doanh
- Đưa ra những lời khuyên tốt cho việc kinh doanh của ngân hàng


**Chạy**:
```powershell
jupyter lab
# Mở notebooks/03_modeling.ipynb
```

### Sử dụng modules trong code

**Import và sử dụng data processing functions**:
```python
import sys
sys.path.append("..")  # Nếu chạy từ notebooks/

import numpy as np
from src.data_processing import (
    load_data,
    split_features_target,
    fill_unknowns_in_categorical,
    add_engineered_features,
    standardize,
    one_hot_encode_all,
    train_test_split_np,
    undersample
)

# Load data
header, raw = load_data("../data/raw/BankChurners.csv")
feature_cols, X_raw, y = split_features_target(header, raw)
```

**Import và sử dụng model**:
```python
from src.models import LogisticRegressionNumpy, classification_metrics

# Load processed data
X_train = np.load("../data/processed/X_train.npy")
y_train = np.load("../data/processed/y_train.npy")
X_test = np.load("../data/processed/X_test.npy")
y_test = np.load("../data/processed/y_test.npy")

# Train model
model = LogisticRegressionNumpy(lr=0.01, n_iter=1500)
model.fit(X_train, y_train, verbose=True)

# Predict
y_pred = model.predict(X_test, threshold=0.5)

# Evaluate
acc, prec, rec, f1, _ = classification_metrics(y_test, y_pred)
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
```

**Visualization**:
```python
from src.visualization import (
    plot_class_distribution,
    plot_histogram,
    plot_confusion_matrix
)

# Plot class distribution
plot_class_distribution(y, labels=["Existing", "Attrited"])

# Plot confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
plot_confusion_matrix(cm, title="Confusion Matrix")
```

---

## Results

### Kết quả Training

**Hyperparameters**:
- Learning rate: 0.01
- Number of iterations: 1,500
- Optimization: Batch Gradient Descent
- Training set: Balanced với undersampling (ratio = 1.0)

**Loss Curve**: 
- Binary cross-entropy loss giảm đều qua các iterations
- Model converge tốt, không có dấu hiệu overfitting trong quá trình training
- Có thể xem loss curve trong notebook `03_modeling.ipynb`

### Metrics đạt được

Kết quả trên **test set** (2,025 samples, 327 churn cases) với các threshold khác nhau:

| Threshold | Accuracy | Precision | Recall | F1-Score | Đánh giá |
|-----------|----------|-----------|--------|----------|----------|
| 0.3       | 69.53%   | 33.85%    | 92.97% | 49.63%   | Aggressive - Recall cao nhất (304/327) |
| 0.4       | 78.07%   | 41.68%    | 89.60% | 56.89%   | Balanced - Cân bằng tốt (293/327) |
| **0.5**   | **85.09%** | **52.39%** | **83.79%** | **64.47%** | **Conservative - F1 cao nhất (274/327)**  |

**Giải thích metrics**:
- **Accuracy**: Tỷ lệ dự đoán đúng tổng thể (85.09%)
- **Precision**: Trong số dự đoán là churn, bao nhiêu % đúng (52.39%)
- **Recall**: Trong số khách churn thật, phát hiện được bao nhiêu % (83.79%)
- **F1-score**: Điểm trung bình điều hòa của Precision và Recall (64.47%)

**Lưu ý**: Tất cả mục tiêu đề ra đều đạt được (Accuracy > 80%, Recall > 70%, F1 > 60%)

### Confusion Matrix (Threshold = 0.5)

```
                Predicted
                0       1
Actual  0      [TN]    [FP]
        1      [FN]    [TP]
```

- **True Negatives (TN)**: Existing customers được dự đoán đúng
- **False Positives (FP)**: Existing customers bị dự đoán nhầm là churn
- **False Negatives (FN)**: Churn customers bị bỏ sót
- **True Positives (TP)**: Churn customers được phát hiện đúng

### Hình ảnh trực quan hóa

Các biểu đồ được tạo trong notebooks:

1. **Class Distribution** (`01_data_exploration.ipynb`):
   - Pie chart thể hiện class imbalance
   - Existing: ~84%, Attrited: ~16%

2. **Feature Distributions** (`01_data_exploration.ipynb`):
   - Histogram của `Customer_Age`
   - Histogram (log scale) của `Total_Trans_Amt`
   - Comparison histogram: `Total_Trans_Ct` giữa Existing vs Attrited

3. **Loss Curve** (`03_modeling.ipynb`):
   - Binary cross-entropy loss theo iterations
   - Thể hiện quá trình convergence của gradient descent

4. **Confusion Matrix** (`03_modeling.ipynb`):
   - Heatmap 2×2 với annotations
   - So sánh predicted vs actual labels

5. **Metrics vs Threshold** (`03_modeling.ipynb`):
   - Line plot của 4 metrics (Accuracy, Precision, Recall, F1) theo threshold
   - Thể hiện trade-off giữa precision và recall

### Feature Importance - Top 3 đặc trưng quan trọng nhất

Từ phân tích model weights:

1. **Total_Trans_Ct** (số giao dịch) - weight = -1.19 
   - Yếu tố QUAN TRỌNG NHẤT
   - Weight âm → Số giao dịch càng thấp, khả năng churn càng cao

2. **Amt_per_Trans** (engineered feature!) - weight = +0.65
   - Feature engineering thành công
   - Số tiền trung bình mỗi giao dịch càng cao → càng ít churn

3. **Total_Ct_Chng_Q4_Q1** (tỷ lệ thay đổi giao dịch) - weight = -0.61
   - Giảm hoạt động = tín hiệu churn

### Threshold Optimization

**Về Metrics**: Threshold = 0.5 tốt nhất với F1-score cao nhất (64.47%)

**Về Chi phí kinh doanh**: Threshold = 0.4 tối ưu nhất
- Tổng chi phí: $37,500 (tiết kiệm $1,450 so với threshold 0.5)
- Phát hiện 90% khách churn (293/327 cases)
- Chi phí FN (bỏ sót): $17,000
- Chi phí FP (chăm sóc nhầm): $20,500

### Business Recommendations

1. **Deploy với threshold = 0.4** để tối ưu chi phí
2. **Focus vào transaction behavior**: Monitor Total_Trans_Ct và Total_Trans_Amt
3. **Retention strategies**:
   - Target customers với low Total_Trans_Ct (< 50 giao dịch/năm)
   - Offer benefits cho high Amt_per_Trans
   - Focus vào customers inactive > 2.5 tháng

### Phân tích và nhận xét

**1. Trade-off giữa Precision và Recall**:
- **Threshold thấp (0.3)**: Recall cao (~0.75) nhưng Precision thấp (~0.35)
  - Phát hiện được nhiều churn customers nhưng có nhiều false alarms
  - Phù hợp khi muốn ưu tiên không bỏ sót khách hàng churn (cost of losing customer cao)
  
- **Threshold cao (0.5)**: Precision cao hơn (~0.50) nhưng Recall thấp hơn (~0.55)
  - Chính xác hơn khi dự đoán churn nhưng bỏ sót một số cases
  - Phù hợp khi muốn tập trung nguồn lực vào những cases chắc chắn hơn

**2. Model Performance**:
- Model học được pattern phân biệt giữa churn và non-churn customers
- Với balanced training set, model không bị bias quá nhiều về majority class
- F1-score khoảng 0.50-0.52 cho thấy model có thể cải thiện thêm

**3. Feature Importance (từ EDA)**:
- `Total_Trans_Amt` và `Total_Trans_Ct` có sự khác biệt rõ rệt giữa 2 classes
- Khách hàng churn có xu hướng:
  - Tổng số tiền giao dịch thấp hơn
  - Số lượng giao dịch ít hơn
  - Thời gian inactive dài hơn

**4. Impact of Imbalance Handling**:
- Undersampling giúp model không bị overwhelm bởi majority class
- Test set giữ nguyên phân bố gốc (imbalanced) để đánh giá realistic performance

---

## Project Structure

```
Credit Card customer/
│
├── data/
│   ├── raw/
│   │   └── BankChurners.csv          # Dataset gốc (10,127 rows × 23 cols)
│   │
│   └── processed/
│       ├── X_train.npy               # Training features (balanced)
│       ├── y_train.npy               # Training labels (balanced)
│       ├── X_test.npy                # Test features
│       ├── y_test.npy                # Test labels
│       ├── X_train_full.npy          # Training features (full, imbalanced)
│       └── y_train_full.npy          # Training labels (full, imbalanced)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA và hypothesis testing
│   ├── 02_preprocessing.ipynb        # Preprocessing pipeline bằng NumPy
│   └── 03_modeling.ipynb             # Training và evaluation
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_processing.py            # Data processing functions (301 lines)
│   ├── models.py                     # LogisticRegressionNumpy class (81 lines)
│   └── visualization.py              # Plotting functions (89 lines)
│
├── requirements.txt                   # Dependencies (numpy, matplotlib, seaborn, scikit-learn)
└── README.md                         # Documentation (file này)
```

### Mô tả chi tiết từng file/folder

#### **`data/`**
- **`raw/BankChurners.csv`**: 
  - Dataset gốc từ Kaggle
  - 10,127 customers × 23 columns
  - Chứa thông tin nhân khẩu học, giao dịch và target (Attrition_Flag)

- **`processed/`**: 
  - Dữ liệu đã qua preprocessing, lưu dưới dạng NumPy arrays (`.npy`)
  - Mỗi sample có 17 numeric features (14 gốc + 3 engineered) + 19 one-hot features = 36 features
  - Train set có 2 versions: balanced (để train) và full (backup)

#### **`notebooks/`**

**`01_data_exploration.ipynb`** (440 lines):
- Load data bằng NumPy (`np.genfromtxt`)
- Thống kê mô tả: min, max, mean, std, quartiles
- Phân tích class distribution (pie chart)
- Visualize distributions: histograms, comparison plots
- **Hypothesis Testing**: t-test cho `Total_Trans_Amt` giữa churn vs non-churn
  - Kết quả: có sự khác biệt có ý nghĩa thống kê (t = -22.69, p < 0.05)

**`02_preprocessing.ipynb`** (159 lines):
- Load raw data
- Fill missing categorical values với mode
- Feature engineering: 3 features mới
- Standardization: Z-score cho numeric features
- One-hot encoding: 5 categorical features → 19 dimensions
- Train/test split: 80/20
- Undersampling: balance training set
- Save processed data

**`03_modeling.ipynb`**:
- Load processed data từ `.npy` files
- Train LogisticRegressionNumpy với lr=0.01, n_iter=1500
- Plot loss curve
- Evaluate với threshold = 0.5
- Compare metrics across thresholds (0.3, 0.4, 0.5)
- Plot confusion matrix và metrics vs threshold

#### **`src/`**

**`data_processing.py`** (301 lines) - Core preprocessing module:
```python
# Main functions:
- load_data(csv_path)                    # Load CSV bằng NumPy
- split_features_target(header, raw)     # Tách X, y
- get_default_feature_groups()           # Phân loại cat/num features
- fill_unknowns_in_categorical(X_cat)    # Xử lý missing với mode
- add_engineered_features(X_num)         # Tạo 3 features mới
- standardize(x)                         # Z-score standardization
- one_hot_encode_all(X_cat)              # One-hot encoding
- train_test_split_np(X, y)              # Train/test split
- undersample(X, y, ratio)               # Undersampling
```

**`models.py`** (81 lines) - Logistic Regression implementation:
```python
# Main components:
- sigmoid(z)                             # Sigmoid activation
- binary_cross_entropy(y_true, y_pred)   # Loss function
- LogisticRegressionNumpy                # Main model class
  - __init__(lr, n_iter)
  - fit(X, y, verbose)                   # Training với GD
  - predict_proba(X)                     # Probability prediction
  - predict(X, threshold)                # Binary prediction
- confusion_matrix_np(y_true, y_pred)    # Compute TP, TN, FP, FN
- classification_metrics(y_true, y_pred) # Compute Acc, Prec, Rec, F1
```

**`visualization.py`** (89 lines) - Plotting utilities:
```python
# Main functions:
- plot_class_distribution(y)             # Pie chart cho class balance
- plot_histogram(x)                      # Histogram đơn giản
- plot_histogram_by_class(x, y)          # Histogram so sánh 2 classes
- plot_confusion_matrix(cm)              # Heatmap confusion matrix
```

#### **Root files**

**`requirements.txt`**:
```
numpy         # Core library cho mọi thứ
matplotlib    # Basic plotting
seaborn       # Statistical visualization
scikit-learn  # (Chỉ để so sánh, không dùng trong core)
```

**`README.md`**: 
- Documentation đầy đủ về project
- Hướng dẫn installation, usage
- Giải thích method và algorithms
- Kết quả và phân tích

---

## Challenges & Solutions

### Khó khăn gặp phải khi sử dụng NumPy

#### **1. Không có Automatic Differentiation**
**Thách thức**: 
- Phải tự tính gradient bằng tay cho mọi parameter
- Dễ mắc lỗi trong công thức toán học
- Khó mở rộng cho các model phức tạp

**Giải pháp**:
- Sử dụng công thức đạo hàm chuẩn của Binary Cross-Entropy Loss
- Kiểm tra kỹ shape của mảng trong mọi phép toán
- Vectorize hoàn toàn: dùng `np.einsum` thay vì loops
```python
# Efficient vectorized gradient computation
dw = np.einsum('ij,i->j', X, error) / n_samples
```

#### **2. Numerical Stability Issues**
**Thách thức**:
- Sigmoid overflow khi z quá lớn/nhỏ: `e^(-z)` có thể → ∞ hoặc 0
- Log của 0 hoặc số rất nhỏ → NaN hoặc -∞
- Ảnh hưởng đến loss computation và convergence

**Giải pháp**:
```python
# Clip logits trước khi đưa vào sigmoid
z = np.clip(z, -500, 500)

# Clip probabilities trước khi tính log
eps = 1e-15
y_pred = np.clip(y_pred, eps, 1 - eps)
```

#### **3. Hiệu năng với Dataset lớn**
**Thách thức**:
- NumPy chỉ chạy trên CPU (single-threaded cho nhiều operations)
- Không có GPU acceleration như PyTorch/TensorFlow
- Memory-intensive với large matrices

**Giải pháp**:
- Sử dụng `np.einsum` cho efficient matrix operations
- Avoid Python loops → pure vectorized NumPy operations
- Với dataset > 100K samples hoặc cần GPU: chuyển sang PyTorch/TensorFlow

#### **4. Xử lý Categorical Features**
**Thách thức**:
- NumPy không có built-in one-hot encoder như pandas
- Phải tự implement từ đầu
- Dễ nhầm lẫn về indexing và shape

**Giải pháp**:
```python
def one_hot_encode(col):
    classes, inv = np.unique(col, return_inverse=True)
    N, K = col.shape[0], classes.shape[0]
    one_hot = np.zeros((N, K))
    one_hot[np.arange(N), inv] = 1.0
    return one_hot, classes
```
- Sử dụng `np.unique` với `return_inverse=True` để map values → indices
- Tạo zero matrix và fill bằng advanced indexing

#### **5. Reading CSV với Mixed Data Types**
**Thách thức**:
- CSV có cả numeric và categorical columns
- `np.genfromtxt` yêu cầu single dtype

**Giải pháp**:
```python
# Load toàn bộ dưới dạng string
raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
# Convert numeric columns sau khi tách
X_num = X_num_str.astype(float)
```

#### **6. Imbalanced Data**
**Thách thức**:
- Class 1 (churn) chỉ chiếm ~16% → model bias về class 0
- Metrics như accuracy không phản ánh đúng performance

**Giải pháp**:
- **Undersampling**: Random sample class 0 để cân bằng với class 1
- Giữ nguyên test set imbalanced để đánh giá realistic
- Sử dụng F1-score thay vì chỉ accuracy
- Thử các threshold khác nhau để optimize precision/recall trade-off

---

## Future Improvements

### 1. Advanced Algorithms
- **Implement Neural Network bằng NumPy**: 
  - Multi-layer perceptron với backpropagation
  - ReLU, Dropout, Batch Normalization
- **Ensemble Methods**:
  - Implement Decision Tree và Random Forest từ đầu
  - Bagging, Boosting (AdaBoost)
- **Compare với scikit-learn/XGBoost/LightGBM**

### 2. Better Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **Cost-sensitive learning**: Assign higher weights to minority class
- **Threshold optimization**: Tìm optimal threshold bằng ROC curve

### 3. Hyperparameter Tuning
- Implement **Grid Search** bằng NumPy
- **Random Search** for faster exploration
- Learning rate scheduling: decay theo epochs
- Early stopping based on validation loss
- L2 regularization với cross-validation để tìm λ tốt nhất

### 4. Feature Engineering & Selection
- **Feature importance analysis**: 
  - Tính correlation với target
  - Permutation importance
- **Polynomial features**: `X₁²`, `X₁X₂`, etc.
- **Feature selection**: 
  - Forward/Backward selection
  - Recursive Feature Elimination (RFE)
- **Dimensionality reduction**: PCA bằng NumPy

### 5. Model Evaluation
- **K-Fold Cross-Validation** để đánh giá robust hơn
- **ROC-AUC curve** và optimal threshold selection
- **Precision-Recall curve** cho imbalanced data
- **Learning curves**: Training vs validation loss/accuracy
- **Calibration plot**: So sánh predicted probabilities vs actual frequencies

### 6. Deployment
- **Web API**: Flask/FastAPI endpoint cho prediction
- **Model serialization**: Save/load weights bằng `np.save`/`np.load`
- **Real-time prediction pipeline**:
  - Input validation
  - Preprocessing với saved statistics (mean, std, one-hot classes)
  - Inference
  - Output post-processing
- **Monitoring**: Track prediction distribution, performance metrics over time

### 7. Code Quality
- **Unit tests** cho mọi functions trong `src/`
- **Type hints** và docstrings đầy đủ
- **Logging** thay vì print statements
- **Configuration file** (YAML/JSON) cho hyperparameters
- **Reproducibility**: Fix all random seeds

### 8. Scalability
- **Mini-batch Gradient Descent** thay vì batch GD
- **Stochastic Gradient Descent** (SGD)
- **Multi-threading** cho data loading và preprocessing
- **Chuyển sang PyTorch/TensorFlow** khi cần:
  - GPU acceleration
  - Automatic differentiation
  - Built-in optimizers (Adam, RMSprop)

---

## Contributors

**Tác giả**: 
- Họ tên: Nguyễn Minh Duy
- MSSV: 22120080
- Lớp: P4CS_23/21
- Trường: Đại học Khoa học Tự nhiên - ĐHQG TP.HCM

**Contact**:
- Email: 22120080@student.hcmus.edu.vn
- GitHub: MDuy-3107

**Môn học**: Python for Computer Science  
**Học kỳ**: HK1 (2024-2025)  
**Lab**: Lab02 - Credit Card Customer Churn Prediction

---

## License

This project is created for **educational purposes** as part of coursework at VNU-HCMUS.

**Sử dụng dataset**: BankChurners dataset từ Kaggle  
**Mã nguồn**: Tự triển khai hoàn toàn bằng NumPy

---

## Acknowledgments

- **Dataset**: [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) from Kaggle
- **Course**: Python for Computer Science
- **Instructor**: Phạm Trọng Nghĩa / Lê Nhựt Nam
- **References**:
  - NumPy Documentation: https://numpy.org/doc/
  - "Pattern Recognition and Machine Learning" - Christopher Bishop

---

**Last Updated**: November 2025

