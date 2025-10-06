"""
Report Validator: Enforce schema + business rules

Multi-layer validation for wallet activity reports.

Usage:
    validator = ReportValidator(schema_path)
    validator.validate(report)  # Raises if invalid

Features:
- JSON Schema compliance (Draft 2020-12)
- Business logic validation
- Detailed error reporting
"""

import json
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
from typing import Dict, Any, List


class ReportValidator:
    """
    Multi-layer report validation
    
    Validates:
    1. JSON Schema compliance
    2. Business logic consistency
    3. Edge cases (zero activity, etc.)
    """
    
    def __init__(self, schema_path: str = "schemas/report_v1.json"):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.validator = Draft202012Validator(self.schema)
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load and validate schema itself"""
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Check schema is valid
        Draft202012Validator.check_schema(schema)
        
        return schema
    
    def validate(self, report: Dict[str, Any]) -> bool:
        """
        Validate report (raises on error)
        
        Args:
            report: Report dict to validate
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: Schema or business logic violation
        """
        # Layer 1: Schema validation
        self.validator.validate(report)
        
        # Layer 2: Business logic
        self._validate_business_logic(report)
        
        return True
    
    def _validate_business_logic(self, report: Dict[str, Any]):
        """Validate business rules"""
        # Rule 1: Time range sanity
        from datetime import datetime
        from_ts = datetime.fromisoformat(report["time"]["from_ts"].replace("Z", "+00:00"))
        to_ts = datetime.fromisoformat(report["time"]["to_ts"].replace("Z", "+00:00"))
        
        if from_ts >= to_ts:
            raise ValidationError("from_ts must be < to_ts")
        
        # Rule 2: Consistency check (totals ≈ sum of stats)
        # Note: Allow small floating-point discrepancy
        stats_inbound = sum(s["inbound"] for s in report["transfer_stats"])
        stats_outbound = sum(s["outbound"] for s in report["transfer_stats"])
        
        totals_inbound = report["totals"]["inbound"]
        totals_outbound = report["totals"]["outbound"]
        
        tolerance = 0.01  # 1 cent tolerance
        
        if abs(stats_inbound - totals_inbound) > tolerance:
            raise ValidationError(
                f"Totals mismatch: stats inbound={stats_inbound}, "
                f"totals inbound={totals_inbound}"
            )
        
        if abs(stats_outbound - totals_outbound) > tolerance:
            raise ValidationError(
                f"Totals mismatch: stats outbound={stats_outbound}, "
                f"totals outbound={totals_outbound}"
            )
        
        # Rule 3: tx_count consistency
        if report["tx_count"] > 0 and len(report["transfer_stats"]) == 0:
            raise ValidationError("tx_count > 0 but transfer_stats is empty")
    
    def validate_file(self, report_path: str) -> bool:
        """Validate report from file"""
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return self.validate(report)
    
    def get_errors(self, report: Dict[str, Any]) -> List[Dict]:
        """Get all validation errors (non-raising)"""
        errors = []
        
        # Schema errors
        for error in self.validator.iter_errors(report):
            errors.append({
                "type": "schema",
                "path": ".".join(str(p) for p in error.path),
                "message": error.message
            })
        
        # Business logic errors
        try:
            self._validate_business_logic(report)
        except ValidationError as e:
            errors.append({
                "type": "business",
                "path": None,
                "message": str(e)
            })
        
        return errors


# CLI usage
if __name__ == "__main__":
    import sys
    
    validator = ReportValidator()
    
    if len(sys.argv) > 1:
        # Validate file
        report_path = sys.argv[1]
        try:
            validator.validate_file(report_path)
            print(f"✅ {report_path} is valid")
        except ValidationError as e:
            print(f"❌ Validation failed: {e.message}")
            sys.exit(1)
    else:
        # Validate stdin
        report = json.load(sys.stdin)
        try:
            validator.validate(report)
            print("✅ Report is valid")
        except ValidationError as e:
            print(f"❌ Validation failed: {e.message}")
            sys.exit(1)

