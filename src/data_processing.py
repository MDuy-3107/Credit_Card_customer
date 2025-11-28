import numpy as np

# 1. Load & tách dữ liệu
def load_data(csv_path: str):
    """
    Đọc file BankChurners.csv bằng NumPy.

    Trả về:
      - header: list tên cột
      - raw: mảng (N, D) dtype=str
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        # Bỏ các kí tự thừa ""
        header = [h.strip('"') for h in header]

    raw = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1,
        dtype=str
    )
    # bỏ các kí tự thừa ""
    raw = np.char.strip(raw, '"')
    return header, raw

# 2. Tách X, y
def split_features_target(header, raw):
    """
    Tách X_raw, y từ header + raw, giữ các cột 2..20 (Customer_Age -> Avg_Utilization_Ratio),
    target là Attrition_Flag.

    Attrited Customer -> 1
    Existing Customer -> 0
    """
    idx_attr = header.index("Attrition_Flag")
    feature_cols = header[2:21]
    idx_features = list(range(2, 21))

    y_str = raw[:, idx_attr]
    y = np.where(y_str == "Attrited Customer", 1, 0).astype(int)

    X_raw = raw[:, idx_features]
    return feature_cols, X_raw, y

# 3. Mô tả dữ liệu
def describe_numeric_column(x, name):
    print(f"=== {name} ===")
    print("Min :", np.min(x))
    print("Max :", np.max(x))
    print("Mean:", np.mean(x))
    print("Std :", np.std(x))
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    print("25% :", q25)
    print("50% :", q50)
    print("75% :", q75)
    print()

def describe_categorical_column(col: np.ndarray, name: str):
    """
    Mô tả cột categorical: in ra các giá trị và tần số.
    """
    print(f"=== {name} ===")
    vals, counts = np.unique(col, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"{v}: {c}")
    print()

def get_default_feature_groups(feature_cols):
    """
    Trả về:
      - cat_cols, num_cols
      - cat_idx, num_idx
    dựa trên danh sách feature_cols mặc định của BankChurners.
    """
    cat_cols = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    num_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]

    cat_idx = [feature_cols.index(c) for c in cat_cols]
    num_idx = [feature_cols.index(c) for c in num_cols]

    return cat_cols, num_cols, cat_idx, num_idx


# 4. Xử lý missing/unknown trong categorical

def fill_unknown_with_mode(col: np.ndarray) -> np.ndarray:
    """
    col: mảng 1D dtype=str
    Thay "" hoặc "Unknown" bằng mode của cột.
    """
    mask_unknown = (col == "") | (col == "Unknown")
    if not np.any(mask_unknown):
        return col

    vals, counts = np.unique(col[~mask_unknown], return_counts=True)
    mode_val = vals[np.argmax(counts)]

    col_filled = col.copy()
    col_filled[mask_unknown] = mode_val
    return col_filled


def fill_unknowns_in_categorical(X_cat: np.ndarray) -> np.ndarray:
    """
    Áp dụng fill_unknown_with_mode cho từng cột categorical.
    """
    X_cat_filled = X_cat.copy()
    for j in range(X_cat_filled.shape[1]):
        X_cat_filled[:, j] = fill_unknown_with_mode(X_cat_filled[:, j])
    return X_cat_filled


# 5. Normalization / Standardization

def minmax_norm(x: np.ndarray, feature_range=(0.0, 1.0)):
    """
    Min-max normalization theo từng cột.
    Trả về:
      - x_norm
      - x_min, x_max
    """
    a, b = feature_range
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    diff = x_max - x_min
    diff[diff == 0] = 1.0
    x_norm = (x - x_min) / diff
    return a + x_norm * (b - a), x_min, x_max


def decimal_scaling(x: np.ndarray):
    """
    Decimal scaling normalization:
      x' = x / 10^j sao cho max(|x'|) < 1
    """
    max_abs = np.max(np.abs(x), axis=0)
    j = np.ceil(np.log10(max_abs + 1e-15))
    scale = np.power(10.0, j)
    scale[scale == 0] = 1.0
    return x / scale, scale


def log_transform(x: np.ndarray, epsilon: float = 1e-6):
    """
    Log transform cho dữ liệu không âm.
    """
    return np.log(x + epsilon)


def standardize(x: np.ndarray):
    """
    Z-score standardization: (x - mean) / std theo từng cột.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    x_std = (x - mean) / std
    return x_std, mean, std


# 6. Feature engineering

def add_engineered_features(X_num: np.ndarray, num_cols: list):
    """
    Tạo thêm một số feature từ BankChurners numeric:

    - Amt_per_Trans    = Total_Trans_Amt / Total_Trans_Ct
    - Inactive_Ratio   = Months_Inactive_12_mon / Months_on_book
    - Rel_per_Month    = Total_Relationship_Count / Months_on_book

    Trả về:
      - X_num_ext
      - num_cols_ext
    """
    X = X_num
    cols = num_cols

    total_amt = X[:, cols.index("Total_Trans_Amt")]
    total_ct = X[:, cols.index("Total_Trans_Ct")]
    months_on_book = X[:, cols.index("Months_on_book")]
    months_inactive = X[:, cols.index("Months_Inactive_12_mon")]
    rel_count = X[:, cols.index("Total_Relationship_Count")]

    amt_per_trans = total_amt / (total_ct + 1e-6)
    inactive_ratio = months_inactive / (months_on_book + 1e-6)
    rel_per_month = rel_count / (months_on_book + 1e-6)

    X_fe = np.column_stack([amt_per_trans, inactive_ratio, rel_per_month])
    fe_names = ["Amt_per_Trans", "Inactive_Ratio", "Rel_per_Month"]

    X_num_ext = np.concatenate([X, X_fe], axis=1)
    num_cols_ext = cols + fe_names

    return X_num_ext, num_cols_ext


# 7. One-hot encoding

def one_hot_encode(col: np.ndarray):
    """
    col: (N,) dtype=str

    Trả về:
      - one_hot: (N, K)
      - classes: (K,)
    """
    classes, inv = np.unique(col, return_inverse=True)
    N = col.shape[0]
    K = classes.shape[0]
    one_hot = np.zeros((N, K), dtype=float)
    one_hot[np.arange(N), inv] = 1.0
    return one_hot, classes


def one_hot_encode_all(X_cat: np.ndarray, cat_cols: list):
    """
    One-hot cho toàn bộ X_cat.

    Trả về:
      - X_cat_oh: (N, sum(K_i))
      - classes_per_cat: dict {col_name: classes}
    """
    one_hot_list = []
    classes_per_cat = {}

    for j, name in enumerate(cat_cols):
        col = X_cat[:, j]
        oh, classes = one_hot_encode(col)
        one_hot_list.append(oh)
        classes_per_cat[name] = classes

    X_cat_oh = np.concatenate(one_hot_list, axis=1)
    return X_cat_oh, classes_per_cat


# 8. Train/test split & undersampling

def train_test_split_np(X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2,
                        random_state: int = 42):
    """
    Train/test split dùng NumPy thuần.
    """
    np.random.seed(random_state)
    N = X.shape[0]
    idx = np.random.permutation(N)
    test_N = int(N * test_size)
    test_idx = idx[:test_N]
    train_idx = idx[test_N:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def undersample(X: np.ndarray, y: np.ndarray, ratio: float = 1.0):
    """
    Undersampling class 0 (Existing) để cân bằng với class 1 (Attrited).

    ratio = số mẫu class 0 giữ lại / số mẫu class 1
    """
    mask1 = (y == 1)
    mask0 = (y == 0)

    X1, y1 = X[mask1], y[mask1]
    X0, y0 = X[mask0], y[mask0]

    n1 = X1.shape[0]
    n0_keep = int(n1 * ratio)

    idx0 = np.random.permutation(X0.shape[0])[:n0_keep]
    X0_sel, y0_sel = X0[idx0], y0[idx0]

    X_bal = np.vstack([X1, X0_sel])
    y_bal = np.concatenate([y1, y0_sel])

    idx = np.random.permutation(X_bal.shape[0])
    return X_bal[idx], y_bal[idx]


