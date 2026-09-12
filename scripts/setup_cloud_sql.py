#!/usr/bin/env python3
"""
Cloud SQL Setup & Data Ingestion Script
---------------------------------------
Provisions/validates tables in Cloud SQL PostgreSQL instance (medquad-postgres):
  1. clinical_sessions: Persists session_id, user_id, active context cache,
     and compressed conversational summaries.
  2. mock_patient_records: Simulated EHR data (vitals, medication history,
     lab panels, conditions, demographics) for patient brief generation.
"""

import asyncio
import datetime
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
logger = logging.getLogger("setup_cloud_sql")

INSTANCE_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "medquad:us-central1:medquad-postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "MedQuAD")
DB_NAME = os.getenv("DB_NAME", "medquad_clinical")


async def init_cloud_sql():
    logger.info(f"Connecting to Cloud SQL instance: {INSTANCE_CONNECTION_NAME}, db={DB_NAME}")
    connector = await create_async_connector()

    conn = await connector.connect_async(
        INSTANCE_CONNECTION_NAME,
        "asyncpg",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
    )

    try:
        # 1. Create clinical_sessions table
        logger.info("Ensuring table 'clinical_sessions' exists...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS clinical_sessions (
            session_id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            app_name VARCHAR(128) NOT NULL DEFAULT 'medquad-agent',
            active_context JSONB DEFAULT '{}'::jsonb,
            compressed_summary TEXT DEFAULT '',
            turn_count INT DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 2. Create mock_patient_records table
        logger.info("Ensuring table 'mock_patient_records' exists...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS mock_patient_records (
            patient_id VARCHAR(64) PRIMARY KEY,
            mrn VARCHAR(64) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            dob DATE,
            gender VARCHAR(32),
            blood_type VARCHAR(16),
            conditions JSONB DEFAULT '[]'::jsonb,
            medications JSONB DEFAULT '[]'::jsonb,
            lab_results JSONB DEFAULT '[]'::jsonb,
            vital_signs JSONB DEFAULT '{}'::jsonb,
            allergies JSONB DEFAULT '[]'::jsonb,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 3. Seed mock_patient_records
        from app.tools.clinical_db import MOCK_PATIENTS

        logger.info(f"Seeding {len(MOCK_PATIENTS)} patient records into 'mock_patient_records'...")
        for pid, rec in MOCK_PATIENTS.items():
            demo = rec["demographics"]
            dob_str = demo.get("dob", "1970-01-01")
            dob_val = datetime.date.fromisoformat(dob_str)

            await conn.execute(
                """
                INSERT INTO mock_patient_records (
                    patient_id, mrn, name, dob, gender, blood_type,
                    conditions, medications, lab_results, vital_signs, allergies, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CURRENT_TIMESTAMP)
                ON CONFLICT (patient_id) DO UPDATE SET
                    mrn = EXCLUDED.mrn,
                    name = EXCLUDED.name,
                    dob = EXCLUDED.dob,
                    gender = EXCLUDED.gender,
                    blood_type = EXCLUDED.blood_type,
                    conditions = EXCLUDED.conditions,
                    medications = EXCLUDED.medications,
                    lab_results = EXCLUDED.lab_results,
                    vital_signs = EXCLUDED.vital_signs,
                    allergies = EXCLUDED.allergies,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                pid,
                demo["mrn"],
                demo["name"],
                dob_val,
                demo.get("gender", "Unknown"),
                demo.get("blood_type", "Unknown"),
                json.dumps(rec.get("conditions", [])),
                json.dumps(rec.get("medications", [])),
                json.dumps(rec.get("lab_results", [])),
                json.dumps(rec.get("vital_signs", {})),
                json.dumps(rec.get("allergies", [])),
            )
            logger.info(f"  --> Seeded Patient: {pid} ({demo['name']} | MRN: {demo['mrn']})")

        # 4. Insert an initial verification session in clinical_sessions
        test_session_id = "sess_verify_init_001"
        await conn.execute(
            """
            INSERT INTO clinical_sessions (
                session_id, user_id, app_name, active_context, compressed_summary, turn_count, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id) DO UPDATE SET
                active_context = EXCLUDED.active_context,
                compressed_summary = EXCLUDED.compressed_summary,
                turn_count = EXCLUDED.turn_count,
                updated_at = CURRENT_TIMESTAMP;
            """,
            test_session_id,
            "clinician_user_default",
            "medquad-agent",
            json.dumps({"current_patient_id": "PT-10492", "active_diagnosis": "Type 2 Diabetes / CKD"}),
            "Initial session verification: Eleanor Vance record loaded with eGFR 31 mL/min.",
            1,
        )
        logger.info(f"  --> Seeded Verification Session: {test_session_id}")

        # 5. Query verification
        patient_count = await conn.fetchval("SELECT COUNT(*) FROM mock_patient_records;")
        session_count = await conn.fetchval("SELECT COUNT(*) FROM clinical_sessions;")

        logger.info("==================================================================")
        logger.info("CLOUD SQL SETUP COMPLETE")
        logger.info(f"Table 'mock_patient_records' row count: {patient_count}")
        logger.info(f"Table 'clinical_sessions' row count:     {session_count}")
        logger.info("==================================================================")

    finally:
        await conn.close()
        await connector.close_async()


def main():
    asyncio.run(init_cloud_sql())


if __name__ == "__main__":
    main()
