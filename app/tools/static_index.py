import re
from typing import Any

# Curated High-Yield Static MedQuAD Medical Index
STATIC_MEDQUAD_RECORDS: list[dict[str, Any]] = [
    {
        "id": "MQ-ONC-001",
        "focus": "Non-Small Cell Lung Cancer (NSCLC) - EGFR Sensitizing Mutations",
        "question_type": "treatment_guideline",
        "question": "What is the recommended frontline targeted therapy for advanced NSCLC patients harboring sensitizing EGFR mutations (exon 19 deletion or L858R)?",
        "answer": "For advanced or metastatic non-small cell lung cancer (NSCLC) harboring common sensitizing EGFR mutations (exon 19 deletion or exon 21 L858R), third-generation EGFR tyrosine kinase inhibitors (Osimertinib) are recommended as first-line standard of care based on the FLAURA Phase III clinical trial. Osimertinib demonstrates superior progression-free survival (median 18.9 months vs 10.2 months) and overall survival compared to first-generation TKIs (gefitinib, erlotinib), along with superior central nervous system (CNS) penetration. Osimertinib combined with platinum-pemetrexed chemotherapy (FLAURA2 regimen) is also approved for patients with high tumor burden or CNS metastases.",
        "source": "National Cancer Institute (NCI) / FLAURA Guidelines",
        "mesh_terms": [
            "Carcinoma, Non-Small-Cell Lung",
            "EGFR inhibitor",
            "Osimertinib",
            "Receptor, Epidermal Growth Factor",
        ],
        "evidence_level": "Phase III RCT / Practice Guideline",
    },
    {
        "id": "MQ-ONC-002",
        "focus": "Non-Small Cell Lung Cancer (NSCLC) - EGFR Resistance Pathways",
        "question_type": "resistance_mechanisms",
        "question": "What resistance mechanisms typically develop following disease progression on frontline Osimertinib, and what are the recommended diagnostic next steps?",
        "answer": "Acquired resistance to frontline osimertinib is heterogeneous. On-target mutations include tertiary EGFR mutations such as C797S in exon 20. Major off-target resistance mechanisms include MET gene amplification (15-20%), HER2 amplification, BRAF V600E mutations, RET/ALK fusions, and histological transformation to small cell lung cancer (SCLC). Clinical standard of care upon progression requires repeat liquid biopsy (cfDNA) or tissue biopsy with comprehensive genomic profiling. If MET amplification is identified, combination EGFR-MET inhibition is indicated. If no targetable genomic alteration is discovered, platinum-doublet chemotherapy (carboplatin/cisplatin + pemetrexed) with or without amivantamab is standard.",
        "source": "NCI / MedQuAD Lung Oncology",
        "mesh_terms": [
            "EGFR C797S",
            "MET Amplification",
            "Osimertinib Resistance",
            "Genomic Profiling",
        ],
        "evidence_level": "Systematic Review / Expert Consensus",
    },
    {
        "id": "MQ-CARD-001",
        "focus": "Atrial Fibrillation - DOAC Anticoagulation in Chronic Kidney Disease",
        "question_type": "dosage_and_safety",
        "question": "How should direct oral anticoagulants (DOACs) like Apixaban be dosed and monitored in patients with atrial fibrillation and renal impairment?",
        "answer": "Direct Oral Anticoagulants (DOACs) are preferred over warfarin for stroke prevention in non-valvular atrial fibrillation. For Apixaban, the standard dose is 5 mg twice daily. Dose reduction to 2.5 mg twice daily is indicated if the patient meets at least two of the following dose-reduction criteria: age >= 80 years, body weight <= 60 kg, or serum creatinine >= 1.5 mg/dL (133 micromol/L). In patients with severe renal disease (eGFR 15-29 mL/min), apixaban can be maintained with caution, but renal function (eGFR) and complete blood counts must be monitored at least every 3 months. If acute kidney injury supervenes with eGFR dropping below 15 mL/min, temporary cessation or conversion to unfractionated heparin should be considered under specialist supervision.",
        "source": "American College of Cardiology (ACC) / MedlinePlus",
        "mesh_terms": [
            "Atrial Fibrillation",
            "Apixaban",
            "Renal Insufficiency, Chronic",
            "Anticoagulants",
        ],
        "evidence_level": "Phase III RCT / ACC/AHA Guideline",
    },
    {
        "id": "MQ-CARD-002",
        "focus": "Heart Failure with Preserved Ejection Fraction (HFpEF)",
        "question_type": "treatment_guideline",
        "question": "What guideline-directed medical therapies reduce cardiovascular death and hospitalizations in patients with HFpEF (LVEF >= 50%)?",
        "answer": "Sodium-glucose cotransporter 2 (SGLT2) inhibitors (empagliflozin and dapagliflozin) are Class 1 recommended therapies for HFpEF based on the EMPEROR-Preserved and DELIVER trials, showing significant reductions in composite cardiovascular mortality and heart failure hospitalizations. Mineralocorticoid receptor antagonists (spironolactone) have a Class 2a recommendation for selected patients with elevated natriuretic peptides and eGFR > 30 mL/min/1.73m2. ARBs and ARNi (sacubitril/valsartan) hold Class 2b recommendations, particularly for patients with LVEF on the lower end of the preserved spectrum (40-57%). Blood pressure control and loop diuretic optimization for congestion remain essential.",
        "source": "NHLBI / MedlinePlus Cardiology",
        "mesh_terms": [
            "Heart Failure",
            "SGLT2 Inhibitors",
            "Empagliflozin",
            "Spironolactone",
        ],
        "evidence_level": "Phase III RCT / AHA/ACC/HFSA Guideline",
    },
    {
        "id": "MQ-ENDO-001",
        "focus": "Type 2 Diabetes Mellitus - SGLT2 Inhibitors and Renal Function Thresholds",
        "question_type": "contraindications_and_criteria",
        "question": "What are the eGFR thresholds and safety monitoring parameters for prescribing SGLT2 inhibitors (Dapagliflozin, Empagliflozin) in Type 2 Diabetes with Chronic Kidney Disease?",
        "answer": "SGLT2 inhibitors (Dapagliflozin, Empagliflozin) are strongly recommended to slow chronic kidney disease (CKD) progression and reduce cardiovascular events in patients with T2D. Initiation is supported down to an eGFR of 20 mL/min/1.73m2 (Dapagliflozin based on DAPA-CKD; Empagliflozin based on EMPA-KIDNEY). Once initiated, therapy may be continued until initiation of renal replacement therapy (dialysis). Glycemic efficacy diminishes when eGFR < 45 mL/min, but cardiorenal protection persists. Essential safety considerations include withholding during acute dehydrating illness or prior to major surgery (risk of euglycemic DKA), monitoring volume status in elderly patients on diuretics, and screening for mycotic genital infections and Fournier gangrene.",
        "source": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) / ADA",
        "mesh_terms": [
            "Diabetes Mellitus, Type 2",
            "Diabetic Nephropathies",
            "Sodium-Glucose Transporter 2 Inhibitors",
            "Glomerular Filtration Rate",
        ],
        "evidence_level": "Phase III RCT / ADA-KDIGO Consensus",
    },
    {
        "id": "MQ-DRUG-001",
        "focus": "Drug Interactions - Warfarin vs Fluconazole and Amiodarone",
        "question_type": "pharmacology_safety",
        "question": "What are the clinical consequences and management strategies for concomitant administration of Warfarin with potent CYP2C9 inhibitors such as Fluconazole or Amiodarone?",
        "answer": "Warfarin is metabolized primarily by CYP2C9 (active S-enantiomer) and CYP3A4/CYP1A2 (less potent R-enantiomer). Concomitant administration of potent CYP2C9 inhibitors such as fluconazole, amiodarone, or metronidazole profoundly impairs S-warfarin clearance, causing dramatic elevations in INR (International Normalized Ratio) and life-threatening hemorrhage risks within 3-7 days. Management requires preemptive warfarin dose reduction of 30% to 50% upon initiating the interacting agent, followed by frequent INR monitoring (every 48-72 hours) until a new steady state is achieved. In patients with high bleeding risk, alternative antifungal or antiarrhythmic therapies with lower CYP2C9 inhibition profiles should be considered.",
        "source": "NLM / DailyMed / MedQuAD Pharmacology",
        "mesh_terms": [
            "Warfarin",
            "Fluconazole",
            "Amiodarone",
            "Cytochrome P-450 CYP2C9",
            "Drug Interactions",
        ],
        "evidence_level": "Pharmacological Surveillance / Clinical Guideline",
    },
    {
        "id": "MQ-INF-001",
        "focus": "Sepsis and Septic Shock - 1-Hour Bundle",
        "question_type": "emergency_management",
        "question": "What are the core elements of the Surviving Sepsis Campaign 1-Hour Bundle for suspected sepsis and septic shock?",
        "answer": "The Surviving Sepsis Campaign 1-Hour Bundle requires rapid initial resuscitation within 60 minutes of sepsis recognition: 1) Measure blood lactate level (remeasure within 2-4 hours if initial lactate > 2 mmol/L); 2) Obtain blood cultures prior to administration of antimicrobial therapy; 3) Administer broad-spectrum intravenous antimicrobials; 4) Rapidly administer 30 mL/kg crystalloid fluid for hypotension or lactate >= 4 mmol/L; 5) Apply vasopressors (norepinephrine as first-choice) during or after fluid resuscitation to maintain mean arterial pressure (MAP) >= 65 mmHg.",
        "source": "CDC / Surviving Sepsis Campaign Guidelines",
        "mesh_terms": [
            "Sepsis",
            "Shock, Septic",
            "Resuscitation",
            "Norepinephrine",
            "Lactic Acid",
        ],
        "evidence_level": "Clinical Practice Guideline",
    },
    {
        "id": "MQ-NEUR-001",
        "focus": "Acute Ischemic Stroke - Thrombolysis and Thrombectomy Windows",
        "question_type": "treatment_emergency",
        "question": "What are the therapeutic time windows and eligibility criteria for IV thrombolysis and mechanical thrombectomy in acute ischemic stroke?",
        "answer": "Intravenous thrombolysis (alteplase 0.9 mg/kg or tenecteplase 0.25 mg/kg) is indicated for eligible acute ischemic stroke patients within 4.5 hours of symptom onset (or last known normal) after non-contrast CT excludes intracranial hemorrhage. Mechanical thrombectomy (EVT) is indicated for large vessel occlusion (LVO) of the internal carotid artery or middle cerebral artery (MCA-M1 segment) within 6 hours of onset (Class 1a) with prestroke mRS 0-1 and ASPECTS >= 6. In extended time windows (6 to 24 hours), EVT is indicated if patients satisfy DAWN or DEFUSE-3 clinical-core mismatch perfusion imaging criteria.",
        "source": "NINDS / AHA/ASA Stroke Guidelines",
        "url": "https://www.ninds.nih.gov/health-information/disorders/stroke",
        "mesh_terms": [
            "Stroke",
            "Ischemic Stroke",
            "Thrombectomy",
            "Tissue Plasminogen Activator",
            "Tenecteplase",
        ],
        "evidence_level": "Phase III RCT / AHA/ASA Guideline",
    },
    {
        "id": "MQ-PAXLOVID-001",
        "focus": "Paxlovid (Nirmatrelvir/Ritonavir) Contraindications and Drug Interactions",
        "question_type": "contraindications",
        "question": "What are the contraindications and significant drug-drug interactions for Paxlovid (nirmatrelvir and ritonavir)?",
        "answer": "Paxlovid (nirmatrelvir co-packaged with ritonavir) is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min) or severe hepatic impairment (Child-Pugh Class C), and in patients with hypersensitivity to nirmatrelvir or ritonavir. Ritonavir is a strong CYP3A inhibitor and significantly increases plasma concentrations of drugs highly dependent on CYP3A for clearance, potentially causing life-threatening toxicities. Concomitant administration is contraindicated with: antiarrhythmics (amiodarone, dronedarone, flecainide, propafenone, quinidine), statins (simvastatin, lovastatin), sedatives/hypnotics (triazolam, oral midazolam), ergot derivatives, PDE5 inhibitors (revatio/sildenafil for PAH), and CYP3A inducers (rifampin, St. John's wort, carbamazepine, phenytoin).",
        "source": "FDA / NIH COVID-19 Treatment Guidelines",
        "url": "https://www.covid19treatmentguidelines.nih.gov/therapies/antivirals-including-antibody-products/ritonavir-boosted-nirmatrelvir-paxlovid/",
        "mesh_terms": [
            "Paxlovid",
            "Nirmatrelvir",
            "Ritonavir",
            "Contraindications",
            "CYP3A4",
            "Drug Interactions",
        ],
        "evidence_level": "FDA EUA / NIH Practice Guideline",
    },
]


class StaticMedicalIndex:
    """
    In-memory static medical index for MedQuAD literature.
    Used as an immediate search engine or resilient fallback when Vertex AI Search is unavailable.
    """

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.records = records or STATIC_MEDQUAD_RECORDS

    def search(
        self, query: str, top_k: int = 3, category_filter: str | None = None
    ) -> list[dict[str, Any]]:
        clean_query = query.lower()
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "at",
            "for",
            "with",
            "about",
            "what",
            "how",
            "is",
            "are",
            "patient",
            "clinical",
        }
        query_tokens = [
            w
            for w in re.findall(r"\w+", clean_query)
            if len(w) > 2 and w not in stop_words
        ]

        scored: list[tuple[float, dict[str, Any]]] = []

        for doc in self.records:
            score = 0.0
            focus = doc.get("focus", "").lower()
            question = doc.get("question", "").lower()
            answer = doc.get("answer", "").lower()
            mesh_terms = [m.lower() for m in doc.get("mesh_terms", [])]

            if (
                category_filter
                and category_filter.lower() not in focus
                and category_filter.lower() not in doc.get("question_type", "").lower()
            ):
                continue

            if clean_query in question or clean_query in answer or clean_query in focus:
                score += 12.0

            for token in query_tokens:
                if token in focus:
                    score += 4.5
                if token in question:
                    score += 3.5
                if any(token in m for m in mesh_terms):
                    score += 4.0
                if token in answer:
                    score += 1.2

            if score > 0:
                snippet = (
                    doc["answer"][:280] + "..."
                    if len(doc["answer"]) > 280
                    else doc["answer"]
                )
                url = doc.get("url") or f"https://medlineplus.gov/search?query={doc['focus'].replace(' ', '+')}"
                scored.append(
                    (
                        score,
                        {
                            "id": doc["id"],
                            "focus": doc["focus"],
                            "question": doc["question"],
                            "answer_snippet": snippet,
                            "full_answer": doc["answer"],
                            "source": doc["source"],
                            "url": url,
                            "mesh_terms": doc.get("mesh_terms", []),
                            "evidence_level": doc.get(
                                "evidence_level", "Clinical Evidence"
                            ),
                            "relevance_score": round(min(score / 15.0, 0.99), 3),
                        },
                    )
                )

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = [item[1] for item in scored[:top_k]]

        # Fallback to top 2 if no token matched
        if not top_matches and self.records:
            for doc in self.records[:2]:
                url = doc.get("url") or f"https://medlineplus.gov/search?query={doc['focus'].replace(' ', '+')}"
                top_matches.append(
                    {
                        "id": doc["id"],
                        "focus": doc["focus"],
                        "question": doc["question"],
                        "answer_snippet": doc["answer"][:280] + "...",
                        "full_answer": doc["answer"],
                        "source": doc["source"],
                        "url": url,
                        "mesh_terms": doc.get("mesh_terms", []),
                        "evidence_level": doc.get(
                            "evidence_level", "Clinical Guidance"
                        ),
                        "relevance_score": 0.50,
                    }
                )

        return top_matches


static_medical_index = StaticMedicalIndex()
