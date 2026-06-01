"""
rag_engine.py  –  Core RAG logic
  • Web scraping    : requests + BeautifulSoup
  • Embeddings      : sentence-transformers (all-MiniLM-L6-v2, runs locally, free)
  • Vector store    : FAISS (in-memory)
  • LLM             : Groq API (llama / mixtral / gemma)
  • Live web search : Tavily Search API (FREE tier = 1000 searches/month)
                      → fallback when FAISS index has no relevant docs
"""

from __future__ import annotations
import re
import time
import textwrap
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Medical Knowledge Base
# Auto-indexed into FAISS on RAGEngine init — always available, no API needed
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_KNOWLEDGE_BASE = """
MEDICAL KNOWLEDGE BASE — MEDIBOT
Comprehensive General Medicine Reference

SECTION 1: COMMON DISEASES & CONDITIONS

--- DIABETES MELLITUS ---
Diabetes mellitus is a chronic metabolic disorder characterized by high blood sugar levels over a prolonged period. There are three main types: Type 1, Type 2, and Gestational Diabetes.
Type 1 Diabetes is an autoimmune condition where the pancreas produces little or no insulin. It is usually diagnosed in children and young adults. Patients require daily insulin injections or use of an insulin pump. Symptoms include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurred vision, and slow-healing sores.
Type 2 Diabetes is the most common form, accounting for 90-95% of all diabetes cases. The body either does not produce enough insulin or does not use it effectively (insulin resistance). It is strongly associated with obesity, physical inactivity, poor diet, and family history. Management includes lifestyle changes, oral medications such as Metformin, and sometimes insulin therapy.
Gestational Diabetes occurs during pregnancy and usually resolves after birth. However, it increases the risk of developing Type 2 diabetes later in life for both mother and child.
Complications of diabetes include diabetic retinopathy (eye damage), nephropathy (kidney damage), neuropathy (nerve damage), cardiovascular disease, and poor wound healing. HbA1c test measures average blood sugar over 3 months. Normal is below 5.7%, prediabetes is 5.7-6.4%, and diabetes is 6.5% or higher.

--- HYPERTENSION (HIGH BLOOD PRESSURE) ---
Hypertension is a condition in which blood pressure in the arteries is persistently elevated. Normal blood pressure is below 120/80 mmHg. Stage 1 hypertension is 130-139/80-89 mmHg. Stage 2 hypertension is 140/90 mmHg or higher. Hypertensive crisis is above 180/120 mmHg and requires emergency care.
Primary (essential) hypertension has no identifiable cause and develops gradually over years. Secondary hypertension is caused by an underlying condition such as kidney disease, thyroid disorders, or sleep apnea.
Risk factors include age, family history, obesity, sedentary lifestyle, high sodium intake, smoking, excessive alcohol consumption, and chronic stress. Hypertension is known as the "silent killer" because it often has no symptoms until serious complications arise, including heart attack, stroke, heart failure, kidney damage, and vision loss.
Treatment includes lifestyle modifications such as the DASH diet, regular exercise, weight loss, limiting alcohol and sodium, quitting smoking, and stress management. Medications include ACE inhibitors (lisinopril), ARBs (losartan), beta-blockers (metoprolol), calcium channel blockers (amlodipine), and diuretics (hydrochlorothiazide).

--- ASTHMA ---
Asthma is a chronic inflammatory disease of the airways characterized by recurring episodes of wheezing, breathlessness, chest tightness, and coughing, particularly at night or early morning. The airways narrow, swell, and may produce extra mucus.
Triggers include allergens (dust mites, pollen, pet dander, mold), respiratory infections, physical activity, cold air, air pollutants, smoke, certain medications like aspirin and NSAIDs, and strong emotions causing rapid breathing.
Diagnosis involves spirometry, peak flow measurement, allergy testing, and bronchoprovocation tests. Treatment uses a step-up approach. Relievers (rescue inhalers) include short-acting beta-agonists (SABA) like salbutamol/albuterol. Controllers include inhaled corticosteroids (ICS) like fluticasone and budesonide, long-acting beta-agonists (LABA), leukotriene modifiers, and biological agents for severe asthma.
An asthma attack requires immediate use of rescue inhaler. If not resolved, emergency care is necessary.

--- PNEUMONIA ---
Pneumonia is an infection that inflames air sacs (alveoli) in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.
Types include bacterial pneumonia (most common, often Streptococcus pneumoniae), viral pneumonia (influenza, COVID-19), fungal pneumonia, and aspiration pneumonia.
Community-acquired pneumonia (CAP) is contracted outside hospitals. Hospital-acquired pneumonia (HAP) is more serious and often caused by drug-resistant bacteria.
Diagnosis uses chest X-ray, blood tests, sputum culture, and pulse oximetry. Treatment depends on cause: bacterial pneumonia uses antibiotics such as amoxicillin, azithromycin, or levofloxacin. The CURB-65 score helps assess severity.

--- TUBERCULOSIS (TB) ---
Tuberculosis is a contagious bacterial infection caused by Mycobacterium tuberculosis, primarily affecting the lungs but can spread to other organs.
Latent TB infection (LTBI) means the bacteria are present but the immune system keeps them dormant. Active TB disease is when the bacteria multiply and cause symptoms.
Symptoms of active pulmonary TB include persistent cough lasting more than 3 weeks, coughing up blood, chest pain, unintentional weight loss, fatigue, fever, night sweats, and loss of appetite.
TB spreads through the air when an infected person coughs, sneezes, or speaks. It is not spread through touch, sharing food, or kissing.
Treatment for active TB uses DOTS strategy with combination therapy for 6 months: isoniazid, rifampicin, pyrazinamide, and ethambutol for 2 months, followed by isoniazid and rifampicin for 4 months.

--- MALARIA ---
Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through bites of infected female Anopheles mosquitoes. Five parasite species infect humans: P. falciparum (most dangerous), P. vivax, P. malariae, P. ovale, and P. knowlesi.
Symptoms begin 10-15 days after being bitten. Classic symptoms include fever with cyclical pattern, chills, sweating, headache, nausea, vomiting, muscle pain, and fatigue. Severe malaria can cause cerebral malaria, severe anemia, respiratory distress, multi-organ failure, and death.
Treatment: Uncomplicated P. falciparum malaria uses artemisinin-based combination therapy (ACT) such as artemether-lumefantrine. Severe malaria uses intravenous artesunate. P. vivax and P. ovale need primaquine to eliminate liver-stage parasites.
Prevention: insecticide-treated mosquito nets (ITNs), indoor residual spraying (IRS), chemoprophylaxis for travelers, and the RTS,S/AS01 (Mosquirix) vaccine.

--- DENGUE FEVER ---
Dengue is a viral infection transmitted by Aedes aegypti mosquitoes, caused by four serotypes (DENV-1 to DENV-4). Second infection with a different serotype increases the risk of severe dengue (dengue hemorrhagic fever / dengue shock syndrome).
Symptoms: sudden high fever (up to 40 degrees C), severe headache, pain behind the eyes, muscle and joint pains (breakbone fever), nausea, vomiting, swollen glands, and rash. Warning signs of severe dengue include abdominal pain, persistent vomiting, rapid breathing, bleeding gums, fatigue, and restlessness.
Diagnosis: NS1 antigen test (first 1-5 days), IgM/IgG serology, PCR, CBC showing thrombocytopenia and leukopenia.
Treatment is supportive: rest, oral rehydration, paracetamol for fever. Avoid aspirin and NSAIDs (increase bleeding risk).

--- TYPHOID FEVER ---
Typhoid fever is caused by Salmonella typhi bacteria, spread through contaminated food and water.
Symptoms develop 1-3 weeks after exposure: sustained high fever, weakness, stomach pain, headache, loss of appetite, and sometimes a flat rose-colored rash. If untreated, can cause intestinal perforation and hemorrhage.
Treatment: antibiotics such as ciprofloxacin, azithromycin, or ceftriaxone.

--- URINARY TRACT INFECTION (UTI) ---
UTI is an infection in any part of the urinary system. Most infections involve the lower urinary tract (bladder and urethra). Women are at greater risk due to shorter urethra.
Types: Cystitis (bladder) — burning urination, frequent urge, cloudy urine, pelvic discomfort. Urethritis — burning with urination. Pyelonephritis (kidney) — upper back and side pain, high fever, chills, nausea, vomiting.
Common causative organisms: Escherichia coli (80% of cases), Staphylococcus saprophyticus, Klebsiella, Proteus, Enterococcus.
Treatment: antibiotics — nitrofurantoin, TMP-SMX, fosfomycin for uncomplicated cystitis. Fluoroquinolones (ciprofloxacin) for pyelonephritis.

--- IRON DEFICIENCY ANEMIA ---
Iron deficiency anemia is the most common type of anemia. Causes include inadequate iron intake, poor absorption, increased demand (pregnancy), or blood loss (menstruation, GI bleeding).
Symptoms: fatigue, weakness, pale skin and conjunctiva, shortness of breath, dizziness, cold hands and feet, brittle nails, pica (craving non-food items), headache.
Diagnosis: low hemoglobin, low MCV (microcytic), low MCH (hypochromic). Low serum iron, low ferritin, high TIBC.
Treatment: oral iron supplements (ferrous sulfate, ferrous gluconate) taken with vitamin C to enhance absorption. Dietary sources: red meat, spinach, beans, lentils, fortified cereals.

SECTION 2: VITAL SIGNS & NORMAL VALUES

Normal Body Temperature: 36.1 to 37.2 degrees C (97 to 99 degrees F). Fever is defined as temperature above 38 degrees C (100.4 degrees F). Hyperthermia is above 40 degrees C and is an emergency. Hypothermia is below 35 degrees C and is also an emergency.
Normal Heart Rate (Pulse): Adults 60 to 100 beats per minute (bpm). Children (1-12 years) 70 to 120 bpm. Newborns 100 to 160 bpm. Tachycardia: above 100 bpm in adults. Bradycardia: below 60 bpm in adults.
Normal Respiratory Rate: Adults 12 to 20 breaths per minute. Children (1-5 years) 20 to 30 breaths/min. Newborns 30 to 60 breaths/min.
Normal Blood Pressure: Systolic 90-119 mmHg / Diastolic 60-79 mmHg. Hypotension: below 90/60 mmHg. Stage 1 hypertension: 130-139/80-89. Stage 2: 140/90 or higher.
Normal Blood Oxygen Saturation (SpO2): 95% to 100%. Below 95% is concerning. Below 90% is a medical emergency requiring supplemental oxygen.
Normal Fasting Blood Glucose: 70 to 99 mg/dL. Prediabetes: 100-125 mg/dL. Diabetes: 126 mg/dL or higher on two occasions.
Normal HbA1c: Below 5.7%. Prediabetes: 5.7-6.4%. Diabetes: 6.5% or higher.
Normal CBC: Hemoglobin Men 13.5-17.5 g/dL, Women 12.0-15.5 g/dL. WBC 4,500 to 11,000 cells/mcL. Platelets 150,000 to 400,000/mcL.
Normal Liver Function: ALT 7-56 U/L, AST 10-40 U/L, ALP 44-147 U/L, Total Bilirubin 0.1-1.2 mg/dL, Albumin 3.4-5.4 g/dL.
Normal Kidney Function: Serum Creatinine Men 0.74-1.35 mg/dL, Women 0.59-1.04 mg/dL. BUN 7-20 mg/dL. eGFR 60 mL/min/1.73m2 or above is normal.
Normal Thyroid: TSH 0.4-4.0 mIU/L, Free T4 0.8-1.8 ng/dL, Free T3 2.3-4.2 pg/mL.
Normal Lipid Profile: Total Cholesterol below 200 mg/dL. LDL below 100 mg/dL (optimal). HDL Men above 40 mg/dL, Women above 50 mg/dL. Triglycerides below 150 mg/dL.
Normal Electrolytes: Sodium 136-145 mEq/L. Potassium 3.5-5.1 mEq/L. Chloride 98-107 mEq/L. Bicarbonate 22-29 mEq/L. Calcium 8.5-10.5 mg/dL. Magnesium 1.7-2.2 mg/dL.

SECTION 3: COMMON MEDICATIONS

--- PARACETAMOL (ACETAMINOPHEN) ---
Class: Analgesic and antipyretic. Used for mild to moderate pain and fever. Standard adult dose: 500-1000 mg every 4-6 hours, maximum 4000 mg/day. Overdose causes severe liver damage. Antidote: N-acetylcysteine (NAC). Safe in pregnancy.

--- IBUPROFEN ---
Class: NSAID. Used for pain, fever, and inflammation. Adult dose: 200-400 mg every 4-6 hours, maximum 1200-2400 mg/day. Contraindicated in peptic ulcer disease, renal impairment, last trimester of pregnancy. Take with food. Side effects include GI bleeding, kidney damage.

--- AMOXICILLIN ---
Class: Penicillin antibiotic. Used for bacterial infections including ear infections, throat infections, respiratory tract infections, UTIs, skin infections. Adult dose: 250-500 mg three times daily or 875 mg twice daily. Side effects: diarrhea, rash, nausea. Contraindicated in penicillin allergy. Complete the full course.

--- METFORMIN ---
Class: Biguanide antidiabetic. First-line treatment for Type 2 diabetes. Reduces hepatic glucose production and improves insulin sensitivity. Dose: start 500 mg once or twice daily with meals, up to 2000-2500 mg/day. Side effects: GI upset, metallic taste. Rare but serious: lactic acidosis. Contraindicated in severe kidney disease (eGFR below 30), liver failure. Does not cause hypoglycemia when used alone.

--- ATORVASTATIN ---
Class: Statin (HMG-CoA reductase inhibitor). Used to lower LDL cholesterol. Dose: 10-80 mg once daily, usually at night. Side effects: muscle pain and weakness (myopathy), elevated liver enzymes, rarely rhabdomyolysis. Contraindicated in pregnancy and active liver disease.

--- OMEPRAZOLE ---
Class: Proton Pump Inhibitor (PPI). Reduces stomach acid production. Used for GERD, peptic ulcer disease, H. pylori eradication. Adult dose: 20-40 mg once daily before meals. Long-term use risks: hypomagnesemia, vitamin B12 deficiency.

--- SALBUTAMOL (ALBUTEROL) ---
Class: Short-acting beta-2 agonist (SABA) bronchodilator. Rescue inhaler for asthma and COPD bronchospasm. Onset: 5 minutes, duration 4-6 hours. Dose: 2 puffs (100 mcg each) every 4-6 hours as needed. Side effects: tremor, palpitations, tachycardia, headache, hypokalemia with high doses.

--- ASPIRIN ---
Class: NSAID / antiplatelet. Low-dose aspirin (75-100 mg daily) used to prevent heart attacks and strokes in high-risk patients. Contraindicated in children under 16 (Reye's syndrome risk), active peptic ulcer, bleeding disorders. Side effects: GI bleeding, tinnitus at high doses.

--- LISINOPRIL ---
Class: ACE Inhibitor. First-line for hypertension, heart failure, and diabetic nephropathy. Dose: 5-40 mg once daily. Side effects: dry persistent cough (up to 20% of patients), hyperkalemia, angioedema (rare but serious). Contraindicated in bilateral renal artery stenosis and pregnancy.

--- AMLODIPINE ---
Class: Calcium Channel Blocker (dihydropyridine). Used for hypertension and angina. Dose: 5-10 mg once daily. Side effects: peripheral edema (ankle swelling), flushing, headache, palpitations.

--- AZITHROMYCIN ---
Class: Macrolide antibiotic. Used for community-acquired pneumonia, bronchitis, sinusitis, STIs, skin infections. Commonly prescribed as Z-pack: 500 mg on day 1, then 250 mg on days 2-5. Side effects: GI upset, diarrhea, QT prolongation.

--- CETIRIZINE ---
Class: Second-generation antihistamine. Used for allergic rhinitis, urticaria, and other allergic conditions. Adult dose: 10 mg once daily. Less sedating than first-generation antihistamines. Side effects: mild drowsiness, dry mouth.

--- ORAL REHYDRATION SALTS (ORS) ---
Used for management of dehydration due to diarrhea and vomiting. WHO ORS formula dissolved in 1 liter of clean water. Sodium 75 mEq/L, Glucose 75 mmol/L, Potassium 20 mEq/L. Osmolarity 245 mOsm/L. Continue breastfeeding. Offer small frequent sips.

SECTION 4: FIRST AID & EMERGENCIES

--- CARDIOPULMONARY RESUSCITATION (CPR) ---
Adult CPR (CAB sequence): Check responsiveness. Call emergency services. Check for breathing (no more than 10 seconds). If no pulse and no normal breathing, begin chest compressions. Compress at least 2 inches (5 cm) deep. Rate: 100-120 compressions per minute. After 30 compressions, give 2 rescue breaths. Ratio 30:2. Continue until AED arrives or person shows signs of life.
AED: Turn on, follow voice prompts. Apply pads. Analyze rhythm. Deliver shock if advised. Immediately resume CPR.
Infant CPR: Use 2 fingers. Compress 1.5 inches. Ratio 30:2 for one rescuer.

--- CHOKING (HEIMLICH MANEUVER) ---
For conscious adult or child over 1 year: Ask if choking. Give 5 back blows with heel of hand between shoulder blades. If unsuccessful, give 5 abdominal thrusts: place fist just above navel, grasp with other hand, thrust sharply inward and upward. Alternate 5 back blows and 5 abdominal thrusts until object expelled or person becomes unconscious.
For infant under 1 year: 5 back blows face-down on forearm plus 5 chest thrusts (not abdominal). Never do blind finger sweeps.

--- ANAPHYLAXIS ---
Anaphylaxis is a severe, life-threatening allergic reaction. Triggers: food (peanuts, shellfish), medications (penicillin, NSAIDs), insect stings, latex.
Symptoms: skin hives, flushing, itching, swelling, throat tightening, wheezing, shortness of breath, drop in blood pressure, rapid heart rate, dizziness, fainting, confusion.
Treatment: Epinephrine (adrenaline) IM injection is first-line. Dose: 0.3-0.5 mg in outer thigh. Call emergency services. Lay patient flat with legs elevated. Second epinephrine dose after 5-15 minutes if no improvement. Oxygen, IV fluids, antihistamines, and corticosteroids are secondary treatments.

--- BURNS ---
First degree (superficial): Affects epidermis only. Red, dry, painful. Heals in 3-5 days.
Second degree (partial thickness): Affects epidermis and dermis. Red, blistered, wet, very painful. Heals in 2-3 weeks.
Third degree (full thickness): Destroys all skin layers. White, brown, or black, leathery, painless (nerve damage). Requires skin grafting.
First aid for minor burns: Cool the burn under cool (not cold) running water for 10-20 minutes. Do NOT use ice, butter, or toothpaste. Remove jewelry and loose clothing. Cover with sterile non-stick dressing. Do NOT pop blisters.

--- STROKE ---
FAST acronym: F-Face drooping, A-Arm weakness, S-Speech difficulty, T-Time to call emergency services immediately.
Types: Ischemic stroke (87% of cases) — clot blocks blood vessel to brain. Hemorrhagic stroke — blood vessel ruptures. TIA (Transient Ischemic Attack) — temporary block, warning sign.
First aid: Do not give anything by mouth. Lay patient on side if unconscious. Call emergency services. Note time symptoms started. Ischemic stroke must be treated within 4.5 hours with tPA (thrombolysis).

--- HYPOGLYCEMIA ---
Low blood sugar below 70 mg/dL. Symptoms: shakiness, sweating, hunger, dizziness, rapid heartbeat, irritability, confusion, blurred vision, seizures.
Rule of 15: If conscious, give 15 grams of fast-acting carbohydrate. Wait 15 minutes, recheck blood sugar. If still below 70, repeat. If unconscious: do NOT give anything by mouth. Glucagon injection 1 mg IM/SC. Call emergency services.

--- SEIZURES ---
First aid: Protect from injury, clear hard objects, cushion head. Time the seizure. Turn to recovery position after convulsions stop. Do NOT restrain movements. Do NOT put anything in mouth. Call emergency services if: first seizure, lasts more than 5 minutes, does not regain consciousness.

--- FRACTURES ---
Signs: pain, swelling, deformity, bruising, inability to use limb, crepitus.
First aid: Immobilize the injured area in the position found. Do not attempt to straighten. Apply splint if possible. Elevate if possible. Apply ice wrapped in cloth for 20 minutes. Control bleeding if open fracture. Monitor circulation, sensation, movement below fracture.

SECTION 5: NUTRITION & PREVENTIVE HEALTH

Carbohydrates: Primary energy source. 4 kcal/gram. Recommended 45-65% of daily calories. Fiber: 25-38 g/day for adults.
Proteins: Building blocks of body tissues. 4 kcal/gram. Recommended: 0.8 g/kg body weight for sedentary adults.
Fats: 9 kcal/gram. Omega-3 fatty acids (EPA, DHA) anti-inflammatory; found in fatty fish, flaxseed, walnuts. Recommended: 20-35% of daily calories.
Vitamin A: Vision, immune function. Deficiency: night blindness, xerophthalmia.
Vitamin B12: Nerve function, red blood cell formation. Deficiency: megaloblastic anemia, neuropathy. Common in vegans.
Vitamin C: Antioxidant, immune function, collagen synthesis. Deficiency: scurvy.
Vitamin D: Bone health, immune function. Deficiency: rickets (children), osteomalacia (adults). Recommended: 600-800 IU/day adults.
Folic Acid (Folate/B9): Critical in early pregnancy to prevent neural tube defects (spina bifida). Recommended: 400 mcg/day for adults, 600 mcg/day during pregnancy.
Iron: Vitamin C enhances absorption of non-heme iron.
Calcium: Recommended 1000-1200 mg/day for adults. Deficiency: osteoporosis.
Iodine: Thyroid hormone synthesis. Deficiency: goiter, hypothyroidism, cretinism.
BMI = Weight (kg) / Height (m2). Underweight below 18.5. Normal 18.5-24.9. Overweight 25-29.9. Obese Class I 30-34.9. Obese Class II 35-39.9. Morbid Obesity 40 or above.

SECTION 6: MENTAL HEALTH

--- DEPRESSION ---
Major Depressive Disorder (MDD): persistently depressed mood and/or loss of interest in activities (anhedonia) lasting at least 2 weeks.
Treatment: CBT (Cognitive Behavioral Therapy) is first-line. Medications: SSRIs (fluoxetine, sertraline, escitalopram) are first-line. SNRIs (venlafaxine, duloxetine). Screening tool: PHQ-9.

--- ANXIETY DISORDERS ---
Generalized Anxiety Disorder (GAD): Excessive, uncontrollable worry for 6 months or more. Symptoms: restlessness, fatigue, concentration difficulty, irritability, muscle tension, sleep disturbance.
Panic Disorder: Recurrent unexpected panic attacks with palpitations, sweating, shortness of breath, chest pain, dizziness, fear of dying.
Treatment: CBT (exposure therapy), SSRIs/SNRIs. Screening: GAD-7 scale.

--- SUICIDAL IDEATION ---
Always take seriously. Assess risk: ask directly about suicidal thoughts, plan, means, and intent. Risk factors: previous attempts, depression, substance use, social isolation, recent major loss.
Immediate action: Do not leave person alone. Remove access to means. Call crisis services.
Crisis resources: National Suicide Prevention Lifeline: 988 (USA).

--- POST-TRAUMATIC STRESS DISORDER (PTSD) ---
Develops after exposure to traumatic event. Symptoms: intrusive memories, flashbacks, nightmares, avoidance, negative thoughts, hyperarousal.
Treatment: Trauma-focused CBT, EMDR, SSRIs (sertraline, paroxetine are FDA-approved for PTSD).

SECTION 7: WOMEN'S HEALTH

Antenatal care: WHO recommends minimum 8 antenatal care contacts. Key tests: blood group, Rh factor, CBC, blood glucose, syphilis, HIV, hepatitis B. Danger signs in pregnancy: vaginal bleeding, severe headache, visual disturbances, sudden swelling, reduced fetal movement. Antenatal supplements: Folic acid 400 mcg daily, Iron 60 mg elemental iron daily, Calcium 1.5-2 g/day.
Preeclampsia: Hypertension (140/90 mmHg or higher) after 20 weeks of pregnancy with proteinuria. Management: Antihypertensive therapy, magnesium sulfate for seizure prevention, delivery is definitive treatment.
Breast Cancer: Risk factors include age, family history (BRCA1/BRCA2 genes). Warning signs: new breast lump, change in size/shape, skin dimpling, nipple retraction or discharge. Screening: mammography every 1-2 years for women 40-74.
Cervical Cancer: Caused by persistent infection with high-risk HPV (primarily HPV 16 and 18). Prevention: HPV vaccination (9-14 years), regular Pap smear every 3 years.

SECTION 8: PEDIATRIC HEALTH

Growth milestones: 12 months: Walks with support, first words (1-2 words), pincer grasp. 18 months: Walks independently, 10-20 words. 24 months: Runs, 2-word phrases, 50+ words. 3 years: 3-word sentences, toilet trained.
Red flags requiring evaluation: No babbling by 12 months, no single words by 16 months, no 2-word phrases by 24 months, any loss of previously acquired skills.
Acute Diarrhea in Children: Most common cause rotavirus. Assess dehydration severity. IMCI treatment: ORS, zinc supplementation 10-20 mg/day for 10-14 days, continue feeding.
Pneumonia classification (IMCI) fast breathing thresholds: 0-2 months 60/min or above. 2-12 months 50/min or above. 1-5 years 40/min or above.
Severe Acute Malnutrition (SAM): MUAC below 115 mm OR WHZ below -3 SD OR bilateral pitting edema (kwashiorkor). MUAC: Green 125 mm or above (normal), Yellow 115-124 mm (moderate malnutrition), Red below 115 mm (severe malnutrition).

SECTION 9: COMMON SYMPTOMS & DIFFERENTIAL DIAGNOSIS

Fever definition: Core temperature 38 degrees C or above (100.4 degrees F). Fever patterns: Continuous (typhoid, pneumonia), Intermittent (malaria), Remittent (bacterial infections).
Chest Pain: Cardiac — ACS (crushing/pressure pain radiating to arm/jaw). Pulmonary — Pulmonary embolism (pleuritic pain, dyspnea). GI — GERD (burning, worse lying down). Red flags: radiation to arm or jaw, sweating, associated dyspnea, hypotension.
Headache: Tension headache — bilateral pressure/tightening. Migraine — unilateral, throbbing, nausea/vomiting, photophobia. Cluster headache — severe unilateral periorbital pain. Red flags (SNOOP): Systemic symptoms, Neurological deficits, Onset sudden/thunderclap, Older age above 50.
Shortness of Breath (Dyspnea) — Acute: Asthma exacerbation, COPD exacerbation, pulmonary embolism, pneumothorax, pulmonary edema, anaphylaxis. Treat cause. Oxygen if SpO2 below 94%.

SECTION 10: INFECTION PREVENTION & CONTROL

Standard precautions: hand hygiene, PPE when exposure to body fluids anticipated, safe injection practices, respiratory hygiene, safe handling of sharps.
Transmission-based precautions: Airborne (TB, measles, chickenpox) — N95 mask, negative pressure room. Droplet (influenza, COVID-19, meningitis, pertussis) — surgical mask. Contact (MRSA, C. difficile, scabies) — gloves and gown.
Antibiotic stewardship: Use antibiotics only when bacterial infection confirmed or strongly suspected. Choose narrowest spectrum effective antibiotic. Use appropriate dose, route, and duration. Review after 48-72 hours.
Sterilization: autoclave (steam 121 degrees C, 15 min) destroys all microorganisms including spores. High-level disinfection: 2% glutaraldehyde, hydrogen peroxide. Low-level disinfection: 70% alcohol, chlorhexidine.
"""

# Trusted medical sources for web scraper fallback (when FAISS has no relevant chunks)
MEDICAL_FALLBACK_URLS = [
    "https://www.mayoclinic.org/search/search-results?q={query}",
    "https://medlineplus.gov/search/?query={query}",
    "https://www.who.int/search?query={query}",
]

# ─────────────────────────────────────────────────────────────────────────────
# Live Web Search  (Tavily — free tier)
# ─────────────────────────────────────────────────────────────────────────────

class LiveSearch:
    """Wraps Tavily Search API for real-time web results."""

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Return list of {title, url, content} dicts."""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
        }
        try:
            resp = requests.post(self.ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            # Tavily returns a top-level "answer" and "results" list
            if data.get("answer"):
                results.append({
                    "title": "Tavily Direct Answer",
                    "url": "tavily://answer",
                    "content": data["answer"],
                })
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                })
            return results
        except Exception as e:
            return [{"title": "Search Error", "url": "", "content": str(e)}]


# ─────────────────────────────────────────────────────────────────────────────
# Web Scraper
# ─────────────────────────────────────────────────────────────────────────────

class WebScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    def scrape_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        visited: set[str] = set()
        all_pages: List[Dict] = []
        self._crawl(url, url, depth, visited, all_pages)
        return {"pages": all_pages, "total": len(all_pages)}

    def _crawl(self, base: str, url: str, depth: int, visited: set, pages: list):
        if url in visited or depth < 0:
            return
        visited.add(url)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception:
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        body = soup.get_text(separator=" ")
        text = clean_text(body)

        if len(text) > 100:
            pages.append({"url": url, "title": title, "text": text})

        if depth > 1:
            for a in soup.find_all("a", href=True):
                href = urljoin(base, a["href"])
                if urlparse(href).netloc == urlparse(base).netloc:
                    self._crawl(base, href, depth - 1, visited, pages)
                    time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Web Scraper Fallback
# Triggered automatically when FAISS has no relevant chunks AND no Tavily key.
# Scrapes trusted medical sites directly for the query topic.
# ─────────────────────────────────────────────────────────────────────────────

class WebScraperFallback:
    """
    Scrapes trusted medical websites when the built-in knowledge base
    and user-indexed documents cannot answer a query.
    Designed to complete within ~30-50 seconds (well within 1 minute).
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    # Trusted medical search endpoints — format with query keyword(s)
    SEARCH_ENDPOINTS = [
        "https://www.mayoclinic.org/search/search-results?q={q}",
        "https://medlineplus.gov/search/?query={q}",
    ]

    def search_and_scrape(self, query: str, max_pages: int = 3) -> List[Dict]:
        """
        1. Hit each search endpoint to get result links.
        2. Scrape the first 1-2 result pages per endpoint.
        3. Return list of {url, title, content} dicts.
        Aborts after max_pages total pages to stay within time budget.
        """
        collected: List[Dict] = []
        seen_urls: set[str] = set()
        pages_scraped = 0

        for endpoint_tpl in self.SEARCH_ENDPOINTS:
            if pages_scraped >= max_pages:
                break

            search_url = endpoint_tpl.format(q=requests.utils.quote(query))
            try:
                resp = requests.get(search_url, headers=self.HEADERS, timeout=12)
                resp.raise_for_status()
            except Exception:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract all unique links from search results page
            candidate_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # Make absolute
                href = urljoin(search_url, href)
                parsed = urlparse(href)
                # Only follow same-domain links that look like article pages
                if (
                    parsed.netloc == urlparse(search_url).netloc
                    and len(parsed.path) > 5
                    and href not in seen_urls
                    and not any(x in href for x in ["search", "login", "account", "javascript", "#"])
                ):
                    candidate_links.append(href)
                    seen_urls.add(href)

            # Scrape top candidate links from this endpoint
            for link in candidate_links[:2]:
                if pages_scraped >= max_pages:
                    break
                try:
                    page_resp = requests.get(link, headers=self.HEADERS, timeout=12)
                    page_resp.raise_for_status()
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")

                    # Remove clutter
                    for tag in page_soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        tag.decompose()

                    title = (
                        page_soup.title.string.strip()
                        if page_soup.title and page_soup.title.string
                        else link
                    )
                    text = clean_text(page_soup.get_text(separator=" "))

                    if len(text) > 200:
                        collected.append({
                            "url": link,
                            "title": title,
                            "content": text[:4000],  # cap per page
                        })
                        pages_scraped += 1
                        time.sleep(0.5)   # be polite

                except Exception:
                    continue

        return collected


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[str], sources: List[str]):
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        self.metadata.extend([{"source": s} for s in sources])

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        if self.index.ntotal == 0:
            return [], []
        q = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results, srcs = [], []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                # Only include chunks with reasonable similarity (L2 distance < 2.0)
                if distances[0][i] < 2.0:
                    results.append(self.chunks[idx])
                    srcs.append(self.metadata[idx]["source"])
        return results, srcs

    def clear(self):
        self.index.reset()
        self.chunks.clear()
        self.metadata.clear()

    @property
    def total(self):
        return self.index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        top_k: int = 4,
        temperature: float = 0.3,
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.tavily_api_key = tavily_api_key
        self.live_search: Optional[LiveSearch] = (
            LiveSearch(tavily_api_key) if tavily_api_key else None
        )

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = VectorStore(dim=384)
        self.scraper = WebScraper()
        self.scraper_fallback = WebScraperFallback()   # ← new: web scraper fallback
        self._source_set: set[str] = set()

        # ── Auto-load the built-in medical knowledge base ──────────────────────
        self._load_builtin_kb()

    def set_tavily_key(self, key: str):
        self.tavily_api_key = key
        self.live_search = LiveSearch(key)

    # ── Built-in KB loader ────────────────────────────────────────────────────

    def _load_builtin_kb(self):
        """
        Embeds BUILTIN_KNOWLEDGE_BASE into FAISS at startup.
        Labeled as 'MediBot Built-in KB' so it appears in source chips.
        Skips if already loaded (prevents double-indexing on Streamlit reruns).
        """
        label = "MediBot Built-in KB"
        if label in self._source_set:
            return  # already loaded
        text = clean_text(BUILTIN_KNOWLEDGE_BASE)
        chunks = chunk_text(text, chunk_size=400, overlap=60)
        if not chunks:
            return
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        sources = [label] * len(chunks)
        self.vector_store.add(np.array(embeddings), chunks, sources)
        self._source_set.add(label)
        self._builtin_chunk_count = len(chunks)


    def add_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        result = self.scraper.scrape_url(url, depth=depth)
        pages = result["pages"]
        if not pages:
            raise ValueError(f"No content extracted from {url}")

        total_chunks = 0
        for page in pages:
            chunks = chunk_text(page["text"])
            if not chunks:
                continue
            embeddings = self.embedder.encode(chunks, show_progress_bar=False)
            sources = [page["url"]] * len(chunks)
            self.vector_store.add(np.array(embeddings), chunks, sources)
            self._source_set.add(page["url"])
            total_chunks += len(chunks)

        return {"chunks": total_chunks, "pages": len(pages)}

    def add_text(self, text: str, label: str = "manual") -> Dict[str, Any]:
        text = clean_text(text)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No usable text found.")
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        sources = [label] * len(chunks)
        self.vector_store.add(np.array(embeddings), chunks, sources)
        self._source_set.add(label)
        return {"chunks": len(chunks)}

    # ── Query pipeline ────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        # ── Tier 1: FAISS retrieval (built-in KB + user-indexed docs) ─────────
        q_emb = self.embedder.encode([question], show_progress_bar=False)[0]
        chunks, sources = self.vector_store.search(q_emb, top_k=self.top_k)

        used_web_search   = False
        used_web_scraper  = False   # ← new flag for scraper fallback

        if chunks:
            # ── FAISS path ────────────────────────────────────────────────────
            context = "\n\n---\n\n".join(
                f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources)
            )
            system_prompt = textwrap.dedent(f"""
                You are MediAssist AI, a helpful medical assistant with access to a
                comprehensive built-in medical knowledge base plus any documents
                the user has indexed. Answer the user's question using the context below.
                Be concise, accurate, and cite sources when relevant.
                Always include a brief disclaimer to consult a healthcare professional.

                CONTEXT:
                {context}
            """).strip()

        elif self.live_search:
            # ── Tier 2: Tavily live web search ────────────────────────────────
            used_web_search = True
            web_results = self.live_search.search(question, max_results=5)
            context_parts = []
            for r in web_results:
                if r["content"]:
                    context_parts.append(
                        f"[Source: {r['url']}]\nTitle: {r['title']}\n{r['content']}"
                    )
                    sources.append(r["url"])
            context = "\n\n---\n\n".join(context_parts) if context_parts else "No results found."

            system_prompt = textwrap.dedent(f"""
                You are MediAssist AI with access to LIVE web search results.
                Use the web search results below to answer the question accurately.
                You have up-to-date information — do NOT say your knowledge is limited.
                Be concise, accurate, and cite the sources provided.
                Always include a brief disclaimer to consult a healthcare professional.

                WEB SEARCH RESULTS (fetched live):
                {context}
            """).strip()

        else:
            # ── Tier 3: Web Scraper Fallback ──────────────────────────────────
            # No Tavily key? Automatically scrape trusted medical sites.
            scraped = self.scraper_fallback.search_and_scrape(question, max_pages=3)

            if scraped:
                used_web_scraper = True
                context_parts = []
                for r in scraped:
                    context_parts.append(
                        f"[Source: {r['url']}]\nTitle: {r['title']}\n{r['content']}"
                    )
                    sources.append(r["url"])
                context = "\n\n---\n\n".join(context_parts)

                system_prompt = textwrap.dedent(f"""
                    You are MediAssist AI. The user's question was not found in the
                    local knowledge base, so trusted medical websites were scraped
                    in real-time to answer it. Use the scraped content below.
                    Be concise, accurate, and cite the source URLs provided.
                    Always include a brief disclaimer to consult a healthcare professional.

                    SCRAPED WEB CONTENT (from Mayo Clinic / MedlinePlus):
                    {context}
                """).strip()

            else:
                # ── Tier 4: LLM general knowledge (last resort) ───────────────
                system_prompt = textwrap.dedent("""
                    You are MediAssist AI, a helpful and knowledgeable assistant.
                    Answer the user's question directly and confidently using your knowledge.
                    NEVER say things like "I'm not aware", "my knowledge cutoff", "I don't have
                    real-time information", or suggest the user check other websites.
                    Just answer the question naturally and helpfully as if you know the answer.
                    If it's a current events or recent news question, provide whatever
                    relevant context and background you can without any disclaimers about
                    knowledge limits.
                    Always include a brief disclaimer to consult a healthcare professional
                    for medical topics.
                """).strip()

        # ── LLM call ──────────────────────────────────────────────────────────
        chat = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )

        answer = chat.choices[0].message.content
        unique_sources = [
            s for s in list(dict.fromkeys(sources))
            if s and s != "tavily://answer"
        ]

        return {
            "answer": answer,
            "sources": unique_sources,
            "chunks_used": len(chunks),
            "used_web_search": used_web_search,
            "used_web_scraper": used_web_scraper,   # ← returned to app.py
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        builtin = getattr(self, "_builtin_chunk_count", 0)
        return {
            "chunks": self.vector_store.total,
            "sources": len(self._source_set),
            "builtin_chunks": builtin,
            "user_chunks": max(0, self.vector_store.total - builtin),
        }

    def clear(self):
        """Clear user-added content but keep the built-in KB intact."""
        self.vector_store.clear()
        self._source_set.clear()
        # Reload built-in KB so it's always available
        self._load_builtin_kb()
