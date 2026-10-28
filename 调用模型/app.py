"""上传乘客 CSV → 固定预处理 → 已有神经网络 → JSON。"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from predictor import InputError, PredictionRequest, TitanicPredictor, parse_csv_passengers, validation_details


logger = logging.getLogger(__name__)
MAX_FILE_BYTES = 2 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = TitanicPredictor(Path(__file__).resolve().parent / "assets")
    yield
    app.state.predictor = None


app = FastAPI(title="Titanic 神经网络预测 API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": validation_details(exc.errors())})


def run_prediction(request, response, passengers, threshold):
    try:
        result = request.app.state.predictor.predict(passengers, threshold)
    except InputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:
        logger.exception("Titanic 模型推理失败")
        raise HTTPException(status_code=500, detail="模型预测失败，请查看服务日志") from exc
    response.headers["X-Model-Name"] = "TitanicModel"
    response.headers["X-Received-Content-Type"] = request.headers.get("Content-Type", "not-provided")
    return result


@app.get("/")
def index():
    return {"Service": "TitanicModel", "Docs": "/docs", "CSVUpload": "POST /predict"}


@app.get("/health")
def health(request: Request):
    return {"Status": "ok", "ModelLoaded": request.app.state.predictor is not None}


@app.get("/model-info")
def model_info(request: Request):
    predictor = request.app.state.predictor
    return {
        "Model": "TitanicModel", "InputFeatures": predictor.feature_columns,
        "Device": "cpu", "Ready": True, "WeightsSHA256": predictor.weights_sha256,
        "RequiredCSVColumns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    }


@app.post("/predict")
def predict_csv(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="UTF-8 编码的乘客 CSV 文件")],
    threshold: Annotated[float, Form(ge=0, le=1, allow_inf_nan=False)] = 0.5,
):
    try:
        content = file.file.read(MAX_FILE_BYTES + 1)
    finally:
        file.file.close()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="CSV 文件内容不能超过 2 MiB")
    try:
        passengers = parse_csv_passengers(content)
    except InputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return run_prediction(request, response, passengers, threshold)


@app.post("/predict-json")
def predict_json(data: PredictionRequest, request: Request, response: Response):
    return run_prediction(request, response, data.Passengers, data.Threshold)


@app.get("/headers-demo")
def headers_demo(
    request: Request,
    response: Response,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    client: Annotated[str | None, Header(alias="X-Client-Name")] = None,
):
    response.headers["X-Model-Name"] = "TitanicModel"
    return {"ClientName": client, "AuthorizationProvided": bool(authorization), "ContentType": request.headers.get("Content-Type")}
