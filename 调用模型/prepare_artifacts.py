"""只导出已有模型及预处理参数，不执行 main.py 的训练和绘图。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source = (source_dir / "main.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    preprocess = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess_data")
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TitanicModel")
    numeric_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "num_feature_cols" for t in n.targets))
    numeric_columns = ast.literal_eval(numeric_node.value)
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=[preprocess], type_ignores=[]), "<preprocess_data>", "exec"), namespace)
    data = pd.read_csv(source_dir / "train.csv")
    train, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Survived"])
    age_mean = float(train["Age"].mean())
    embarked_mode = str(train["Embarked"].mode()[0])
    processed = namespace["preprocess_data"](train, age_mean, embarked_mode)
    feature_columns = np.load(source_dir / "feature_cols.npy", allow_pickle=False).tolist()
    if feature_columns != processed.drop(columns="Survived").columns.tolist():
        raise ValueError("feature_cols.npy 与当前 main.py 的特征顺序不一致")
    scaler = StandardScaler().fit(processed[numeric_columns])
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "titanic_model.pth", assets / "titanic_model.pth")
    metadata = {
        "SchemaVersion": 1,
        "FeatureColumns": feature_columns,
        "NumericColumns": numeric_columns,
        "NumericMean": scaler.mean_.tolist(),
        "NumericScale": scaler.scale_.tolist(),
        "AgeMean": age_mean,
        "EmbarkedMode": embarked_mode,
        "EmbarkedCategories": sorted(train["Embarked"].dropna().unique().tolist()),
        "TrainingRows": len(train),
        "Source": {
            "MainSHA256": sha256(source_dir / "main.py"),
            "TrainingCSVSHA256": sha256(source_dir / "train.csv"),
            "FeatureColumnsSHA256": sha256(source_dir / "feature_cols.npy"),
            "WeightsSHA256": sha256(source_dir / "titanic_model.pth"),
            "Split": {"test_size": 0.2, "random_state": 42, "stratify": "Survived"},
            "Note": "原项目未保存 scaler；按当前 main.py 和 train.csv 的固定划分重建，未重新训练模型。",
        },
    }
    (assets / "preprocessor.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "model.py").write_text('"""从原 main.py 提取的网络定义，无训练副作用。"""\nimport torch\n\n\n' + ast.unparse(model) + '\n', encoding="utf-8")
    print("已导出原网络、现有权重与固定预处理参数；训练样本数：", len(train))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir)
