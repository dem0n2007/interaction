import logging
import math
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Response

from calculator import calculate


app = FastAPI(title="四则运算 API")
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"Status": "ok"}


@app.post("/calculate")
def calculate_api(
    response: Response,
    Operation: Annotated[
        Literal["sum", "multiply", "subtract", "divide"],
        Form(),
    ],
    Data_A: Annotated[float, Form(allow_inf_nan=False)],
    Data_B: Annotated[float, Form(allow_inf_nan=False)],
):
    # 表单解析、必填检查和数字转换由 FastAPI 完成。
    try:
        result = calculate(Operation, Data_A, Data_B)

    except ZeroDivisionError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("计算过程发生异常")
        raise HTTPException(
            status_code=500,
            detail="服务器内部错误",
        ) from exc

    # 避免数值溢出后生成无法正常返回的结果。
    if not math.isfinite(result):
        raise HTTPException(
            status_code=400,
            detail="计算结果超出支持的数值范围",
        )

    response.headers["X-Service-Name"] = "calculator"

    return {"Result": result}