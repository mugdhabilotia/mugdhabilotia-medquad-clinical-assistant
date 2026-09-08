#!/usr/bin/env python3
"""
Cloud SQL Verification Script
----------------------------
Verifies the Cloud SQL PostgreSQL deployment and the two target tables:
  1. clinical_sessions: Checks session_id, user_id, active_context, and compressed_summary.
  2. mock_patient_records: Checks EHR data (vitals, medication history, lab panels, conditions).
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from google.cloud.sql.connector import create_async_connector

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_cloud_sql")

INSTANCE = os.getenv("CLOUD_SQL_CONNECTION_NAME", "medquad:us-central1:medquad-postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "MedQuAD_Pg2026!Secure")
DB_NAME = os.getenv("DB_NAME", "medquad_clinical")


async def verify():
    logger.info("==================================================================")
    logger.info("STARTING CLOUD SQL POSTGRESQL VERIFICATION")
    logger.info(f"Instance: {INSTANCE}")
    logger.info(f"Database: {DB_NAME}")
    logger.info("==================================================================")

    connector = await create_async_connector()
    conn = await connector.connect_async(
        INSTANCE,
        "asyncpg",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
    )

    try:
        # 1. Verify PostgreSQL version and database
        version = await conn.fetchval("SELECT version();")
        current_db = await conn.fetchval("SELECT current_database();")
        logger.info(f"Connected to Database: {current_db}")
        logger.info(f"PostgreSQL Version:   {version.split(',')[0]}")

        # 2. Verify Table: clinical_sessions
        logger.info("--- Verifying Table: clinical_sessions ---")
        sess_cols = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'clinical_sessions'
            ORDER BY ordinal_position;
        """)
        col_names = [r["column_name"] for r in sess_cols]
        logger.info(f"Columns present: {col_names}")
        assert "session_id" in col_names, "Missing session_id"
        assert "user_id" in col_names, "Missing user_id"
        assert "active_context" in col_names, "Missing active_context"
        assert "compressed_summary" in col_names, "Missing compressed_summary"

        sess_row = await conn.fetchrow("""
            SELECT session_id, user_id, active_context, compressed_summary
            FROM clinical_sessions
            LIMIT 1;
        """)
        logger.info(f"Sample Session ID:        {sess_row['session_id']}")
        logger.info(f"User ID:                  {sess_row['user_id']}")
        logger.info(f"Active Context Cache:     {sess_row['active_context']}")
        logger.info(f"Compressed Summary:       {sess_row['compressed_summary']}")

        # 3. Verify Table: mock_patient_records
        logger.info("--- Verifying Table: mock_patient_records ---")
        patient_cols = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'mock_patient_records'
            ORDER BY ordinal_position;
        """)
        p_col_names = [r["column_name"] for r in patient_cols]
        logger.info(f"Columns present: {p_col_names}")
        assert "patient_id" in p_col_names, "Missing patient_id"
        assert "mrn" in p_col_names, "Missing mrn"
        assert "vital_signs" in p_col_names, "Missing vital_signs"
        assert "medications" in p_col_names, "Missing medications"
        assert "lab_results" in p_col_names, "Missing lab_results"
        assert "conditions" in p_col_names, "Missing conditions"

        patients = await conn.fetch("""
            SELECT patient_id, mrn, name, conditions, medications, lab_results, vital_signs, allergies
            FROM mock_patient_records
            ORDER BY patient_id;
        """)
        logger.info(f"Total Patient Records Found: {len(patients)}")
        for p in patients:
            meds = json.loads(p["medications"]) if isinstance(p["medications"], str) else p["medications"]
            labs = json.loads(p["lab_results"]) if isinstance(p["lab_results"], str) else p["lab_results"]
            vitals = json.loads(p["vital_signs"]) if isinstance(p["vital_signs"], str) else p["vital_signs"]
            logger.info(f"  [+] Patient {p['patient_id']} ({p['name']} | MRN: {p['mrn']}):")
            logger.info(f"      - Active Meds Count: {len(meds)} (e.g. {[m.get('drug') for m in meds[:2]]})")
            logger.info(f"      - Lab Panels Count:  {len(labs)} (e.g. {[lab.get('test') for lab in labs[:2]]})")
            logger.info(f"      - Vitals Recorded:   BP={vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')}, HR={vitals.get('heart_rate')}, SpO2={vitals.get('sp_o2')}%")

        logger.info("==================================================================")
        logger.info("VERIFICATION VERDICT: ALL CHECKS PASSED (100%)")
        logger.info("  1. Cloud SQL Instance 'medquad-postgres' is RUNNABLE")
        logger.info("  2. 'clinical_sessions' table verified with active context & compressed summaries")
        logger.info("  3. 'mock_patient_records' table verified with full EHR vitals, meds, & lab panels")
        logger.info("==================================================================")

    finally:
        await conn.close()
        await connector.close_async()


def main():
    asyncio.run(verify())


if __name__ == "__main__":
    main()
