"""CSV 解析、固定预处理与现有 Titanic 神经网络推理。"""
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, field_validator
import torch

from model import TitanicModel


MAX_ROWS = 1000
REQUIRED_COLUMNS = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")


class Passenger(BaseModel):
    model_config = ConfigDict(extra="ignore")
    PassengerId: Annotated[int | None, Field(ge=1)] = None
    Pclass: Annotated[int, Field(ge=1, le=3)]
    Sex: Literal["male", "female"]
    Age: Annotated[FiniteFloat | None, Field(ge=0, le=120)] = None
    SibSp: Annotated[int, Field(ge=0, le=20)]
    Parch: Annotated[int, Field(ge=0, le=20)]
    Fare: Annotated[FiniteFloat, Field(ge=0)]
    Embarked: Literal["C", "Q", "S"] | None = None

    @field_validator("PassengerId", "Age", "Embarked", mode="before")
    @classmethod
    def blank_to_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("Sex", "Embarked", mode="before")
    @classmethod
    def normalize_category(cls, value, info):
        if isinstance(value, str):
            value = value.strip()
            if info.field_name == "Embarked":
                return value.upper() or None
            return value.lower()
        return value

    @field_validator("PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("数字字段不能使用布尔值")
        return value


class PredictionRequest(BaseModel):
    Passengers: Annotated[list[Passenger], Field(min_length=1, max_length=MAX_ROWS)]
    Threshold: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0.5


class InputError(ValueError):
    def __init__(self, detail, status_code=400):
        super().__init__(str(detail))
        self.detail = detail
        self.status_code = status_code


def validation_details(errors):
    return [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in errors]


def parse_csv_passengers(content: bytes) -> list[Passenger]:
    if not content or not content.strip():
        raise InputError("上传的 CSV 文件不能为空")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("CSV 文件请使用 UTF-8 编码") from exc
    if "\x00" in text:
        raise InputError("文件包含二进制内容，请上传 CSV 文本")
    try:
        reader = csv.reader(StringIO(text, newline=""), strict=True)
        header = next(reader, None)
        if not header:
            raise InputError("CSV 缺少表头")
        header = [name.strip() for name in header]
        if len(set(header)) != len(header):
            raise InputError("CSV 存在重复列名")
        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise InputError({"Message": "CSV 缺少必需列", "MissingColumns": missing})
        passengers = []
        for row_number, values in enumerate(reader, start=1):
            if row_number > MAX_ROWS:
                raise InputError(f"单次请求最多处理 {MAX_ROWS} 名乘客", 413)
            if len(values) != len(header):
                raise InputError(f"第 {row_number} 条数据的列数与表头不一致")
            try:
                passenger = Passenger.model_validate(dict(zip(header, values)))
            except ValidationError as exc:
                raise InputError({
                    "Message": "乘客字段验证失败", "Row": row_number,
                    "Errors": validation_details(exc.errors()),
                }, 422) from exc
            passengers.append(passenger)
    except csv.Error as exc:
        raise InputError("CSV 格式错误，请检查引号、逗号和换行") from exc
    if not passengers:
        raise InputError("CSV 只有表头，没有乘客数据")
    return passengers


class TitanicPredictor:
    def __init__(self, assets_dir: Path):
        assets_dir = Path(assets_dir)
        self.metadata = json.loads((assets_dir / "preprocessor.json").read_text(encoding="utf-8-sig"))
        weights_path = assets_dir / "titanic_model.pth"
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != self.metadata["Source"]["WeightsSHA256"]:
            raise ValueError("模型权重与预处理导出记录不匹配，请重新导出配套文件")
        self.feature_columns = self.metadata["FeatureColumns"]
        expected_features = {"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S", "FamilySize"}
        if len(self.feature_columns) != 9 or set(self.feature_columns) != expected_features:
            raise ValueError("不支持的特征列配置")
        self.numeric_columns = self.metadata["NumericColumns"]
        self.numeric_indices = [self.feature_columns.index(name) for name in self.numeric_columns]
        self.mean = np.asarray(self.metadata["NumericMean"], dtype=np.float64)
        self.scale = np.asarray(self.metadata["NumericScale"], dtype=np.float64)
        if self.mean.shape != (6,) or self.scale.shape != (6,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("标准化参数无效")
        self.model = TitanicModel(input_dim=len(self.feature_columns))
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.model.eval()
        self.weights_sha256 = digest

    def preprocess(self, passengers: list[Passenger]) -> np.ndarray:
        rows = []
        for passenger in passengers:
            embarked = passenger.Embarked or self.metadata["EmbarkedMode"]
            raw = {
                "Pclass": passenger.Pclass,
                "Sex": int(passenger.Sex == "female"),
                "Age": passenger.Age if passenger.Age is not None else self.metadata["AgeMean"],
                "SibSp": passenger.SibSp,
                "Parch": passenger.Parch,
                "Fare": passenger.Fare,
                # 固定训练时的类别；避免单行 get_dummies(drop_first=True) 丢失港口信息。
                "Embarked_Q": int(embarked == "Q"),
                "Embarked_S": int(embarked == "S"),
                "FamilySize": passenger.SibSp + passenger.Parch + 1,
            }
            rows.append([raw[name] for name in self.feature_columns])
        features = np.asarray(rows, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            features[:, self.numeric_indices] = (features[:, self.numeric_indices] - self.mean) / self.scale
            features = features.astype(np.float32)
        if not np.isfinite(features).all():
            raise InputError("预处理结果超出模型支持的数值范围", 422)
        return features

    def predict(self, passengers: list[Passenger], threshold=0.5):
        features = self.preprocess(passengers)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features)).squeeze(1)
            probabilities = torch.sigmoid(logits)
        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("模型输出包含非有限值")
        results = []
        for index, (passenger, probability) in enumerate(zip(passengers, probabilities.tolist()), start=1):
            results.append({
                "Row": index,
                "PassengerId": passenger.PassengerId,
                "Survived": int(probability > threshold),
                "SurvivalProbability": round(probability, 6),
            })
        return {"Model": "TitanicModel", "Count": len(results), "Threshold": threshold, "Result": results}
