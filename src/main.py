from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from api.predictions import router as predictions_router
from api.dataset import router as dataset_router
from api.model import router as model_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from service.model import load_saved_model

    load_saved_model()
    logger.info("Churn service started")
    yield
    logger.info("Churn service stopped")


app = FastAPI(lifespan=lifespan)


def _error_response(code: str, message: str, details: list | dict | None, status_code: int):
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "details": details or []},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    """Keep expected API errors in one stable JSON envelope."""
    detail = exc.detail
    logger.warning("API request failed: method=%s path=%s status=%s", request.method, request.url.path, exc.status_code)
    if isinstance(detail, dict) and {"code", "message"}.issubset(detail):
        return _error_response(
            detail["code"], detail["message"], detail.get("details"), exc.status_code
        )
    code_by_status = {
        400: "BAD_REQUEST", 404: "NOT_FOUND", 422: "VALIDATION_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    # HTTPException messages are authored by this application; never serialize
    # arbitrary exception objects or traceback details.
    return _error_response(
        code_by_status.get(exc.status_code, "HTTP_ERROR"),
        str(detail) if isinstance(detail, str) else "The request could not be completed.",
        [],
        exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: method=%s path=%s", request.method, request.url.path)
    details = [
        {"field": ".".join(str(part) for part in error.get("loc", ())[1:]) or "body",
         "issue": error.get("msg", "Invalid value")}
        for error in exc.errors()
    ]
    return _error_response(
        "VALIDATION_ERROR", "Request data does not match the churn API schema.", details, 422
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected API error: method=%s path=%s", request.method, request.url.path)
    return _error_response(
        "INTERNAL_ERROR", "The service could not complete the request.", [], 500
    )

app.include_router(predictions_router, tags=["predictions"])
app.include_router(dataset_router, tags=["dataset"])
app.include_router(model_router, tags=["model"])


@app.get("/health", tags=["health"], summary="Check service runtime state")
def health():
    from service.health import get_health_status

    return get_health_status()
