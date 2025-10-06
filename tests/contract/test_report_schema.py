"""
Contract tests for JSON Schema compliance

Tests that report_v1.json schema is valid and enforces
required fields, types, and constraints correctly.
"""

import pytest
from jsonschema import Draft202012Validator, ValidationError


@pytest.mark.contract
class TestReportSchemaValidity:
    """Test schema itself is valid"""
    
    def test_schema_is_valid_json_schema(self, schema_v1):
        """Test schema conforms to JSON Schema Draft 2020-12"""
        # Should not raise
        Draft202012Validator.check_schema(schema_v1)
    
    def test_schema_has_required_top_level_fields(self, schema_v1):
        """Test schema defines all required top-level fields"""
        required_fields = schema_v1.get("required", [])
        
        expected_fields = [
            "version", "wallet", "window_hours", "time",
            "totals", "tx_count", "transfer_stats",
            "top_counterparties", "meta"
        ]
        
        for field in expected_fields:
            assert field in required_fields, f"Schema missing required field: {field}"


@pytest.mark.contract
class TestValidReports:
    """Test valid reports pass validation"""
    
    def test_minimal_valid_report(self, schema_v1):
        """Test minimal valid report passes"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {
                "inbound": 0.0,
                "outbound": 0.0
            },
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        # Should not raise
        validator.validate(report)
    
    def test_report_with_transfers(self, schema_v1):
        """Test report with transfer stats passes"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {
                "inbound": 100.50,
                "outbound": 25.25
            },
            "tx_count": 3,
            "transfer_stats": [
                {
                    "token": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                    "symbol": "USDC",
                    "decimals": 6,
                    "inbound": 100.0,
                    "outbound": 25.0,
                    "tx_count": 2
                },
                {
                    "token": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                    "symbol": "USDT",
                    "decimals": 6,
                    "inbound": 0.50,
                    "outbound": 0.25,
                    "tx_count": 1
                }
            ],
            "top_counterparties": [
                {
                    "address": "0x000000000000000000000000000000000000dead",
                    "count": 2
                }
            ],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "novadev://duckdb/transfers"
            }
        }
        
        # Should not raise
        validator.validate(report)


@pytest.mark.contract
class TestInvalidReports:
    """Test invalid reports fail validation"""
    
    def test_missing_required_field_wallet(self, schema_v1):
        """Test missing 'wallet' field fails"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            # "wallet": missing!
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        with pytest.raises(ValidationError, match="'wallet' is a required property"):
            validator.validate(report)
    
    def test_invalid_wallet_format(self, schema_v1):
        """Test invalid wallet address format fails"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "not_a_valid_address",  # Invalid format
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        with pytest.raises(ValidationError, match="pattern"):
            validator.validate(report)
    
    def test_window_hours_too_small(self, schema_v1):
        """Test window_hours < 1 fails"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 0,  # Too small
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        with pytest.raises(ValidationError, match="minimum"):
            validator.validate(report)
    
    def test_window_hours_too_large(self, schema_v1):
        """Test window_hours > 720 fails"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 1000,  # Too large
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        with pytest.raises(ValidationError, match="maximum"):
            validator.validate(report)
    
    def test_additional_properties_rejected(self, schema_v1):
        """Test additional properties are rejected"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            },
            "extra_field": "should_fail"  # Additional property
        }
        
        with pytest.raises(ValidationError, match="Additional properties"):
            validator.validate(report)
    
    def test_invalid_transfer_stat_decimals(self, schema_v1):
        """Test invalid decimals range in transfer_stats"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 1,
            "transfer_stats": [
                {
                    "token": "0x" + "a" * 40,
                    "symbol": "TEST",
                    "decimals": 50,  # > 36 (invalid)
                    "inbound": 0,
                    "outbound": 0,
                    "tx_count": 1
                }
            ],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        with pytest.raises(ValidationError, match="maximum"):
            validator.validate(report)


@pytest.mark.contract
class TestFieldConstraints:
    """Test specific field constraints"""
    
    def test_wallet_address_exactly_40_hex_chars(self, schema_v1):
        """Test wallet must be 0x + 40 hex characters"""
        validator = Draft202012Validator(schema_v1)
        
        # Valid: exactly 40 hex chars
        valid_addresses = [
            "0x" + "0" * 40,
            "0x" + "f" * 40,
            "0x" + "a1b2c3d4" * 5,
        ]
        
        for addr in valid_addresses:
            report = {
                "version": "v1",
                "wallet": addr,
                "window_hours": 24,
                "time": {
                    "from_ts": "2025-10-06T00:00:00Z",
                    "to_ts": "2025-10-07T00:00:00Z"
                },
                "totals": {"inbound": 0, "outbound": 0},
                "tx_count": 0,
                "transfer_stats": [],
                "top_counterparties": [],
                "meta": {
                    "chain_id": 11155111,
                    "generated_at": "2025-10-07T00:00:00Z",
                    "source": "test"
                }
            }
            
            # Should not raise
            validator.validate(report)
    
    def test_negative_totals_invalid(self, schema_v1):
        """Test negative totals are rejected by schema"""
        validator = Draft202012Validator(schema_v1)
        
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {
                "inbound": -10.0,  # Negative (should be rejected by schema minimum: 0)
                "outbound": 0.0
            },
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        # Schema should reject negative values (minimum: 0)
        with pytest.raises(ValidationError, match="minimum"):
            validator.validate(report)

