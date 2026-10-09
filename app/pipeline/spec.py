"""PipelineSpec: the only thing an AI planner may produce. It is data, never code.

The operator catalog is an allowlist; each operator has a typed parameter model and declared
safety properties (DataPilot TDD §7.1).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnsParams(_P):
    columns: list[str] = Field(default_factory=lambda: ["*"], description='Column names, or ["*"] for all text columns')


class CaseParams(_P):
    columns: list[str]
    mode: Literal["lower", "upper", "title"] = "lower"


class PhoneParams(_P):
    columns: list[str]
    country: Literal["IN"] = "IN"


class DateParams(_P):
    columns: list[str]
    output_format: str = "%Y-%m-%d"
    dayfirst: bool = True


class FillParams(_P):
    columns: list[str]
    value: str


class MaskParams(_P):
    columns: list[str]
    method: Literal["partial", "hash", "redact"] = "partial"


class DedupeParams(_P):
    keys: list[str] = Field(min_length=1)
    keep: Literal["first", "last", "most_complete"] = "most_complete"
    conflict_columns: list[str] = Field(default_factory=list,
                                        description="If grouped rows disagree on these columns, quarantine instead of merging")
    text_match: Literal["normalized", "content"] = Field(
        "normalized", description='"normalized" ignores case and spacing; "content" also ignores links, '
                                  'retweet prefixes (RT @user:), HTML codes and punctuation (for free-text keys)')


class MergeParams(_P):
    clusters: list[list[int]] = Field(description="Reviewed duplicate groups, as original row numbers")
    keep: Literal["first", "most_complete"] = "most_complete"
    conflict_columns: list[str] = Field(default_factory=list)


class ValidateParams(_P):
    required: list[str] = Field(default_factory=list)
    formats: dict[str, Literal["email", "phone", "pan", "aadhaar", "date", "ifsc"]] = Field(default_factory=dict)
    allowed_values: dict[str, list[str]] = Field(default_factory=dict)
    unique: list[str] = Field(default_factory=list)


class AnomalyParams(_P):
    columns: list[str] = Field(default_factory=lambda: ["*"], description='Numeric columns, or ["*"] to auto-select measured quantities')
    z_threshold: float = Field(3.5, ge=2.0, le=20.0, description="Robust z-score a value must reach (on top of Isolation Forest)")


class OperatorInfo(BaseModel):
    name: str
    version: str
    title: str
    description: str
    params_model: type[BaseModel]
    lossy: bool = False
    destructive: bool = False   # removes rows/columns from the output
    column_params: tuple[str, ...] = ("columns",)

    model_config = ConfigDict(arbitrary_types_allowed=True)


CATALOG: dict[str, OperatorInfo] = {
    o.name: o
    for o in [
        OperatorInfo(name="trim_whitespace", version="1.0", title="Trim whitespace",
                     description="Strip leading/trailing spaces and collapse repeated spaces.", params_model=ColumnsParams),
        OperatorInfo(name="standardize_case", version="1.0", title="Standardize case",
                     description="Convert text to lower, UPPER or Title case.", params_model=CaseParams),
        OperatorInfo(name="normalize_email", version="1.0", title="Normalize emails",
                     description="Trim and lowercase email addresses.", params_model=ColumnsParams),
        OperatorInfo(name="normalize_phone", version="1.0", title="Normalize phones",
                     description="Format Indian mobile numbers as +91XXXXXXXXXX; invalid values are left for validation.",
                     params_model=PhoneParams),
        OperatorInfo(name="parse_dates", version="1.0", title="Standardize dates",
                     description="Parse mixed date formats and write one format (default YYYY-MM-DD).", params_model=DateParams),
        OperatorInfo(name="fill_missing", version="1.0", title="Fill missing values",
                     description="Replace empty values with an explicit constant.", params_model=FillParams, lossy=True),
        OperatorInfo(name="decode_html", version="1.0", title="Decode HTML codes",
                     description="Replace HTML codes such as &amp; &lt; &#39; with the characters they stand for.",
                     params_model=ColumnsParams),
        OperatorInfo(name="repair_text", version="1.0", title="Repair garbled text",
                     description="Fix corrupted characters, mojibake (such as Ã©) and lost characters (??????) from text-encoding mix-ups.",
                     params_model=ColumnsParams),
        OperatorInfo(name="redact_text_pii", version="1.0", title="Redact PII in text",
                     description="Find emails, phone numbers, Aadhaar, PAN and card numbers inside free text and mask them, keeping the rest of the text.",
                     params_model=ColumnsParams, lossy=True),
        OperatorInfo(name="mask_pii", version="1.0", title="Mask PII",
                     description="Mask sensitive values (partial, hash or redact).", params_model=MaskParams, lossy=True),
        OperatorInfo(name="drop_columns", version="1.0", title="Drop columns",
                     description="Remove columns from the output.", params_model=ColumnsParams, lossy=True, destructive=True),
        OperatorInfo(name="deduplicate", version="1.0", title="Deduplicate",
                     description="Group rows by keys, keep one survivor; conflicting groups are quarantined.",
                     params_model=DedupeParams, lossy=True, destructive=True, column_params=("keys", "conflict_columns")),
        OperatorInfo(name="merge_rows", version="1.0", title="Merge reviewed duplicates",
                     description="Merge human-approved duplicate groups into one record each.",
                     params_model=MergeParams, lossy=True, destructive=True, column_params=("conflict_columns",)),
        OperatorInfo(name="quarantine_anomalies", version="1.0", title="Quarantine anomalies",
                     description="Fit an Isolation Forest on numeric columns and quarantine rows with extreme values (robust z-score), for human review.",
                     params_model=AnomalyParams, destructive=True),
        OperatorInfo(name="validate_rules", version="1.0", title="Validate & quarantine",
                     description="Check required fields, formats, allowed values and uniqueness; failing rows go to quarantine.",
                     params_model=ValidateParams, destructive=True, column_params=("required", "unique")),
    ]
}


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    op: str
    params: dict = Field(default_factory=dict)
    reason: str = ""


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "1.0"
    source_asset_id: str = ""        # injected by the backend, never trusted from a model
    steps: list[Step]
    write_mode: Literal["new_file"] = "new_file"
    dry_run_required: bool = True
    approval_required_for_commit: bool = True


def catalog_for_prompt() -> list[dict]:
    return [
        {"op": o.name, "description": o.description, "params_schema": o.params_model.model_json_schema(),
         "lossy": o.lossy, "destructive": o.destructive}
        for o in CATALOG.values()
    ]
