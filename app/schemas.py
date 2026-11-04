"""Pydantic request/response schemas shared by the API and the Streamlit frontend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.common import load_config

_LIMITS = load_config()["input_limits"]


def _field(name: str) -> Any:
    lim = _LIMITS[name]
    return Field(..., ge=lim["min"], le=lim["max"], description=f"{lim['label']} ({lim['unit']})")


class PredictionRequest(BaseModel):
    """The eight health inputs. Ranges come from configs/model_config.yaml."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "pregnancies": 2,
                "glucose": 135,
                "blood_pressure": 80,
                "skin_thickness": 25,
                "insulin": 90,
                "bmi": 29.5,
                "diabetes_pedigree_function": 0.45,
                "age": 35,
            }
        },
    )

    pregnancies: int = _field("pregnancies")
    glucose: float = _field("glucose")
    blood_pressure: float = _field("blood_pressure")
    skin_thickness: float = _field("skin_thickness")
    insulin: float = _field("insulin")
    bmi: float = _field("bmi")
    diabetes_pedigree_function: float = _field("diabetes_pedigree_function")
    age: int = _field("age")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="1 = risk indicated, 0 = risk not indicated")
    prediction_label: str
    probability: float | None = Field(None, description="Model-estimated probability of the positive class")
    decision_threshold: float = Field(..., description="Probability at or above which risk is indicated")
    model_version: str
    model_name: str
    interpretation: str
    disclaimer: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: Any


# ----------------------------------------------------------------------------- authentication
class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=128)


class NewUser(Credentials):
    role: str = Field("user", pattern="^(user|admin)$")


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str | None = Field(None, pattern="^(user|admin)$")
    is_active: bool | None = None
    password: str | None = Field(None, min_length=1, max_length=128)


class UserOut(BaseModel):
    username: str
    role: str
    is_active: bool
    created_at: str
    last_login: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_at: int
    username: str
    role: str
