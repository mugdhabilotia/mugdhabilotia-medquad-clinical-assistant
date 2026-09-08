import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("medquad.clinical_db")

# Mock PostgreSQL Clinical Database Records
MOCK_PATIENTS = {
    "PT-10492": {
        "demographics": {
            "patient_id": "PT-10492",
            "name": "Eleanor Vance",
            "mrn": "MRN-849201",
            "dob": "1958-04-12",
            "gender": "Female",
            "blood_type": "A+",
        },
        "conditions": [
            {
                "icd10": "E11.22",
                "name": "Type 2 Diabetes Mellitus with Diabetic Chronic Kidney Disease",
                "severity": "Moderate",
                "status": "Active",
            },
            {
                "icd10": "N18.32",
                "name": "Chronic Kidney Disease, Stage 3b (eGFR 31 mL/min)",
                "severity": "Severe",
                "status": "Active",
            },
            {
                "icd10": "I10",
                "name": "Essential (Primary) Hypertension",
                "severity": "Moderate",
                "status": "Active",
            },
        ],
        "medications": [
            {
                "drug": "Metformin HCl",
                "dosage": "500 mg",
                "frequency": "Twice Daily",
                "route": "Oral",
                "status": "Active",
            },
            {
                "drug": "Lisinopril",
                "dosage": "20 mg",
                "frequency": "Once Daily",
                "route": "Oral",
                "status": "Active",
            },
            {
                "drug": "Atorvastatin",
                "dosage": "40 mg",
                "frequency": "Once Daily",
                "route": "Oral",
                "status": "Active",
            },
        ],
        "lab_results": [
            {
                "test": "Serum Creatinine",
                "value": 1.82,
                "unit": "mg/dL",
                "reference": "0.50 - 1.10",
                "flag": "High",
            },
            {
                "test": "eGFR (CKD-EPI)",
                "value": 31.0,
                "unit": "mL/min/1.73m2",
                "reference": "> 60.0",
                "flag": "Low",
            },
            {
                "test": "Hemoglobin A1c",
                "value": 8.4,
                "unit": "%",
                "reference": "4.0 - 5.6",
                "flag": "High",
            },
            {
                "test": "Urine Albumin/Creatinine Ratio (uACR)",
                "value": 420.0,
                "unit": "mg/g",
                "reference": "< 30.0",
                "flag": "High",
            },
            {
                "test": "Serum Potassium",
                "value": 4.9,
                "unit": "mmol/L",
                "reference": "3.5 - 5.0",
                "flag": "Normal",
            },
        ],
        "vital_signs": {
            "systolic_bp": 138,
            "diastolic_bp": 84,
            "heart_rate": 76,
            "sp_o2": 98.0,
            "temperature": 98.4,
        },
        "allergies": [
            {
                "allergen": "Penicillin",
                "reaction": "Anaphylaxis, Urticaria",
                "severity": "Severe",
            }
        ],
    },
    "PT-20831": {
        "demographics": {
            "patient_id": "PT-20831",
            "name": "Marcus Holloway",
            "mrn": "MRN-391024",
            "dob": "1972-11-23",
            "gender": "Male",
            "blood_type": "O-",
        },
        "conditions": [
            {
                "icd10": "C34.90",
                "name": "Stage IV Non-Small Cell Lung Carcinoma (NSCLC)",
                "severity": "Severe",
                "status": "Active",
            },
            {
                "icd10": "Z85.118",
                "name": "Confirmed EGFR Exon 19 In-Frame Deletion",
                "severity": "Severe",
                "status": "Active",
            },
        ],
        "medications": [
            {
                "drug": "Osimertinib (Tagrisso)",
                "dosage": "80 mg",
                "frequency": "Once Daily",
                "route": "Oral",
                "status": "Active",
            },
            {
                "drug": "Dexamethasone",
                "dosage": "2 mg",
                "frequency": "PRN fatigue/nausea",
                "route": "Oral",
                "status": "Active",
            },
        ],
        "lab_results": [
            {
                "test": "EGFR Molecular Panel",
                "value": 1.0,
                "unit": "Exon 19 del detected",
                "reference": "Negative",
                "flag": "Abnormal",
            },
            {
                "test": "ALT / SGPT",
                "value": 42.0,
                "unit": "U/L",
                "reference": "7 - 56",
                "flag": "Normal",
            },
            {
                "test": "AST / SGOT",
                "value": 38.0,
                "unit": "U/L",
                "reference": "10 - 40",
                "flag": "Normal",
            },
            {
                "test": "Platelet Count",
                "value": 210.0,
                "unit": "x10^3/uL",
                "reference": "150 - 450",
                "flag": "Normal",
            },
        ],
        "vital_signs": {
            "systolic_bp": 122,
            "diastolic_bp": 78,
            "heart_rate": 72,
            "sp_o2": 99.0,
            "temperature": 98.2,
        },
        "allergies": [],
    },
    "PT-30914": {
        "demographics": {
            "patient_id": "PT-30914",
            "name": "Sophia Chen",
            "mrn": "MRN-572910",
            "dob": "1965-08-19",
            "gender": "Female",
            "blood_type": "B+",
        },
        "conditions": [
            {
                "icd10": "I48.0",
                "name": "Paroxysmal Atrial Fibrillation (CHA2DS2-VASc = 3)",
                "severity": "Moderate",
                "status": "Active",
            },
            {
                "icd10": "N18.31",
                "name": "Chronic Kidney Disease, Stage 3a",
                "severity": "Moderate",
                "status": "Active",
            },
        ],
        "medications": [
            {
                "drug": "Apixaban (Eliquis)",
                "dosage": "5 mg",
                "frequency": "Twice Daily",
                "route": "Oral",
                "status": "Active",
            },
            {
                "drug": "Metoprolol Succinate",
                "dosage": "50 mg",
                "frequency": "Once Daily",
                "route": "Oral",
                "status": "Active",
            },
        ],
        "lab_results": [
            {
                "test": "Serum Creatinine",
                "value": 1.55,
                "unit": "mg/dL",
                "reference": "0.50 - 1.10",
                "flag": "High",
            },
            {
                "test": "eGFR",
                "value": 42.0,
                "unit": "mL/min/1.73m2",
                "reference": "> 60.0",
                "flag": "Low",
            },
            {
                "test": "Hemoglobin",
                "value": 12.8,
                "unit": "g/dL",
                "reference": "12.0 - 16.0",
                "flag": "Normal",
            },
        ],
        "vital_signs": {
            "systolic_bp": 132,
            "diastolic_bp": 82,
            "heart_rate": 88,
            "sp_o2": 97.5,
            "temperature": 98.6,
        },
        "allergies": [
            {
                "allergen": "Sulfa Drugs",
                "reaction": "Maculopapular Rash",
                "severity": "Moderate",
            }
        ],
    },
}


def execute_readonly_sql_query(
    query: str,
    params: list[Any] | None = None,
) -> dict[str, Any]:
    """Executes a strictly read-only, parameterized SQL query against Cloud SQL PostgreSQL.

    Enforces mandatory security constraints:
      1. Only SELECT and WITH (Common Table Expression) queries are permitted.
      2. Destructive SQL keywords (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, etc.) are rejected.
      3. Positional parameters are bound safely ($1, $2, ...) to prevent SQL injection.
    """
    params = params or []
    cleaned_query = query.strip()
    q_upper = cleaned_query.upper()

    # 1. Guardrail: Must start with SELECT or WITH
    if not (q_upper.startswith("SELECT") or q_upper.startswith("WITH")):
        return {
            "status": "ERROR_SECURITY_VIOLATION",
            "error": "Only read-only SELECT queries are permitted on the clinical database.",
            "results_count": 0,
            "rows": [],
        }

    # 2. Guardrail: Forbid destructive DDL / DML keywords
    forbidden_keywords = {
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "GRANT", "REVOKE", "EXECUTE", "CALL", "COPY"
    }
    tokens = set(q_upper.replace(";", " ").replace("(", " ").replace(")", " ").split())
    violating = forbidden_keywords.intersection(tokens)
    if violating:
        return {
            "status": "ERROR_SECURITY_VIOLATION",
            "error": f"Security restriction: Destructive SQL statement forbidden: {', '.join(sorted(violating))}.",
            "results_count": 0,
            "rows": [],
        }

    # 3. Query Cloud SQL PostgreSQL if instance is configured
    instance_name = os.getenv("CLOUD_SQL_CONNECTION_NAME")
    if instance_name:
        try:
            from google.cloud.sql.connector import create_async_connector

            async def _execute_sql():
                connector = await create_async_connector()
                conn = await connector.connect_async(
                    instance_name,
                    "asyncpg",
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD", "MedQuAD_Pg2026!Secure"),
                    db=os.getenv("DB_NAME", "medquad_clinical"),
                )
                try:
                    clean_sql = cleaned_query.rstrip(";").strip()
                    records = await conn.fetch(clean_sql, *params)
                    results = []
                    for r in records:
                        row_dict = dict(r)
                        for k, v in row_dict.items():
                            if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                                try:
                                    row_dict[k] = json.loads(v)
                                except Exception:
                                    pass
                            elif hasattr(v, "isoformat"):
                                row_dict[k] = v.isoformat()
                        results.append(row_dict)
                    return results
                finally:
                    await conn.close()
                    await connector.close_async()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    rows = pool.submit(lambda: asyncio.run(_execute_sql())).result()
            else:
                rows = asyncio.run(_execute_sql())

            return {
                "status": "SUCCESS",
                "source": "Cloud SQL PostgreSQL (medquad_clinical)",
                "query": cleaned_query,
                "results_count": len(rows),
                "rows": rows,
            }
        except Exception as exc:
            logger.warning(
                f"Cloud SQL PostgreSQL query execution failed: {exc}. "
                "Engaging resilient in-memory simulated EHR fallback."
            )

    # 4. Fallback execution against in-memory EHR records
    return {
        "status": "FALLBACK_SUCCESS",
        "source": "Simulated In-Memory Clinical Database (Fallback)",
        "query": cleaned_query,
        "results_count": len(MOCK_PATIENTS),
        "rows": list(MOCK_PATIENTS.values()),
    }


def query_mock_clinical_db(
    patient_id_or_mrn: str | None = None,
    search_term: str | None = None,
    query_type: str = "summary",
    sql_query: str | None = None,
    params: list[Any] | None = None,
) -> dict[str, Any]:
    """Queries simulated PostgreSQL electronic health records (EHR) for patient data.

    Allows the researcher subagent to look up patient demographics, ICD-10 diagnoses,
    active pharmacotherapy, longitudinal lab results (e.g. eGFR, creatinine), vitals,
    and documented drug allergies, or execute parameterized read-only SQL queries.

    Args:
        patient_id_or_mrn: The unique Patient ID (e.g. 'PT-10492') or Medical Record Number (e.g. 'MRN-849201').
        search_term: Optional keyword to search patients by condition or medication.
        query_type: The scope of inquiry ('summary', 'labs', 'medications', 'cohort').
        sql_query: Optional parameterized read-only SELECT query against PostgreSQL.
        params: Optional query parameters for parameterized execution.

    Returns:
        A dictionary containing the clinical records or query results from PostgreSQL.
    """
    # 1. Parameterized SQL execution path
    if sql_query:
        return execute_readonly_sql_query(sql_query, params)

    # 2. Direct patient lookup by ID or MRN (attempts Cloud SQL first, then MOCK_PATIENTS)
    if patient_id_or_mrn:
        clean_id = patient_id_or_mrn.strip().upper()
        if os.getenv("CLOUD_SQL_CONNECTION_NAME"):
            try:
                sql = (
                    "SELECT patient_id, mrn, name, dob, gender, blood_type, "
                    "conditions, medications, lab_results, vital_signs, allergies "
                    "FROM mock_patient_records WHERE UPPER(patient_id) = $1 OR UPPER(mrn) = $1 LIMIT 1"
                )
                db_res = execute_readonly_sql_query(sql, [clean_id])
                if db_res["status"] == "SUCCESS" and db_res["rows"]:
                    row = db_res["rows"][0]
                    record = {
                        "demographics": {
                            "patient_id": row["patient_id"],
                            "name": row["name"],
                            "mrn": row["mrn"],
                            "dob": row.get("dob", ""),
                            "gender": row.get("gender", ""),
                            "blood_type": row.get("blood_type", ""),
                        },
                        "conditions": row.get("conditions", []),
                        "medications": row.get("medications", []),
                        "lab_results": row.get("lab_results", []),
                        "vital_signs": row.get("vital_signs", {}),
                        "allergies": row.get("allergies", []),
                    }
                    if query_type == "labs":
                        return {"status": "RECORD_FOUND", "patient_id": row["patient_id"], "lab_results": record["lab_results"]}
                    if query_type == "medications":
                        return {"status": "RECORD_FOUND", "patient_id": row["patient_id"], "medications": record["medications"]}
                    return {"status": "RECORD_FOUND", "patient_id": row["patient_id"], "record": record}
            except Exception as exc:
                logger.warning(f"Cloud SQL patient lookup encountered: {exc}. Falling back to in-memory records.")

        # In-memory lookup fallback
        for pid, record in MOCK_PATIENTS.items():
            if pid == clean_id or record["demographics"]["mrn"].upper() == clean_id:
                if query_type == "labs":
                    return {"status": "RECORD_FOUND", "patient_id": pid, "lab_results": record["lab_results"]}
                if query_type == "medications":
                    return {"status": "RECORD_FOUND", "patient_id": pid, "medications": record["medications"]}
                return {"status": "RECORD_FOUND", "patient_id": pid, "record": record}
        return {
            "status": "NOT_FOUND",
            "message": f"No patient matching '{patient_id_or_mrn}' in PostgreSQL clinical DB.",
        }

    # 3. Search term matching across cohort
    if search_term:
        term_clean = search_term.lower()
        matched = []
        for pid, record in MOCK_PATIENTS.items():
            name = record["demographics"]["name"].lower()
            conditions = [c["name"].lower() for c in record["conditions"]]
            meds = [m["drug"].lower() for m in record["medications"]]
            if (
                term_clean in name
                or any(term_clean in c for c in conditions)
                or any(term_clean in m for m in meds)
            ):
                matched.append(
                    {
                        "patient_id": pid,
                        "name": record["demographics"]["name"],
                        "mrn": record["demographics"]["mrn"],
                        "conditions": [c["name"] for c in record["conditions"]],
                        "active_meds": [m["drug"] for m in record["medications"]],
                    }
                )
        return {
            "status": "COHORT_SEARCH",
            "query": search_term,
            "results_count": len(matched),
            "matched_patients": matched,
        }

    # 4. Default cohort overview
    cohort = [
        {
            "patient_id": pid,
            "name": r["demographics"]["name"],
            "mrn": r["demographics"]["mrn"],
            "primary_conditions": [c["name"] for c in r["conditions"][:2]],
        }
        for pid, r in MOCK_PATIENTS.items()
    ]
    return {"status": "COHORT_OVERVIEW", "total_records": len(cohort), "cohort": cohort}
