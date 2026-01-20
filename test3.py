import streamlit as st
import pandas as pd
import random
import re
import time
import json
import difflib
import os  # <--- ADDED for file handling
from datetime import datetime

# ============ CONFIGURATION & CONSTANTS ============
DATASET_FILE = "Newdata 1.csv"
IMAGE_FOLDER = "exercise_images" # <--- CONFIGURE YOUR FOLDER NAME HERE

# CONSTANTS 
PRIMARY_GOALS = ["Weight Loss", "Weight Gain", "Weight Maintenance"]
SECONDARY_GOALS = ["Stress Reduction", "Sleep Improvement", "Athletic Performance", "Posture Correction"]
MEDICAL_CONDITIONS_OPTIONS = [
    "None",
    "Acanthosis Nigricans",
    "Addison’s Disease",
    "Ankylosing Spondylitis",
    "Anxiety Disorders",
    "Arrhythmias",
    "Asthma",
    "Bipolar Disorder",
    "Bladder Cancer",
    "Brain Tumors",
    "Breast Cancer",
    "Bronchitis",
    "Celiac Disease",
    "Cervical Cancer",
    "Cervical Spondylosis",
    "Chickenpox",
    "Chikungunya",
    "COPD",
    "Cirrhosis",
    "Colorectal Cancer",
    "Constipation",
    "Coronary Artery Disease (CAD)",
    "COVID-19 (Post-Infection)",
    "Cushing’s Syndrome",
    "Deep Vein Thrombosis (DVT)",
    "Dengue",
    "Depression",
    "DKA Recovery",
    "Diarrheal Diseases",
    "Disc Herniation",
    "Eating Disorder Recovery",
    "Encephalitis",
    "Epilepsy (Ketogenic)",
    "Fibromyalgia",
    "Fractures",
    "G6PD Deficiency",
    "Gallstones",
    "GERD",
    "Gastritis",
    "Glomerulonephritis",
    "Gout",
    "Heart Failure",
    "Hepatitis",
    "Hepatitis E",
    "HIV/AIDS",
    "Hyperthyroidism",
    "Hypertension",
    "Hypoglycemia",
    "Hypothyroidism",
    "IBD Flare",
    "IBD Remission",
    "Influenza",
    "Insomnia",
    "Interstitial Lung Disease (ILD)",
    "Irritable Bowel Syndrome (IBS)",
    "Lactose Intolerance",
    "Leukemia",
    "Low Back Pain",
    "Lung Cancer",
    "Lymphoma",
    "Malaria",
    "Measles",
    "Meningitis",
    "Menopause",
    "Metabolic Syndrome",
    "Migraine",
    "Multiple Sclerosis",
    "Myocardial Infarction (MI) Recovery",
    "Neuropathy",
    "Obesity",
    "Obsessive Compulsive Disorder (OCD)",
    "Osteoarthritis",
    "Osteoporosis",
    "Ovarian Cancer",
    "Pancreatic Cancer",
    "Parkinson’s Disease",
    "Peptic Ulcer Disease (PUD)",
    "Perimenopause",
    "Peripheral Artery Disease (PAD)",
    "Pneumonia",
    "Post-Traumatic Stress Disorder (PTSD)",
    "Prostate Cancer",
    "Benign Prostatic Hyperplasia (BPH)",
    "Pulmonary Embolism",
    "Pulmonary Hypertension",
    "Pyelonephritis",
    "Rheumatic Heart Disease",
    "Rheumatoid Arthritis",
    "Schizophrenia",
    "Sexually Transmitted Infections (STIs)",
    "Sickle Cell Disease",
    "Sleep Apnea",
    "Stevens-Johnson Syndrome (Recovery)",
    "Stomach Cancer",
    "Stroke Recovery",
    "Substance Use Recovery",
    "Thalassemia",
    "Tuberculosis (TB)",
    "Type 1 Diabetes Mellitus (T1DM)",
    "Type 2 Diabetes Mellitus (T2DM)",
    "Typhoid",
    "Urinary Tract Infection (UTI)",
    "Vitiligo",
    "Other"
]


# EXPERT ALGORITHM CONSTANTS
TRAINING_LEVELS = {
    "Beginner (0–6 months)": {"rpe_range": "2-5", "description": "Simple movements, longer rest, focus on form."},
    "Intermediate (6–24 months)": {"rpe_range": "4-7", "description": "Moderate volume, progressive overload."},
    "Advanced (2+ years)": {"rpe_range": "5-9", "description": "High volume, complex patterns."}
}

st.set_page_config(page_title="FriskaAI Fitness Coach", page_icon="💪", layout="wide")

# ============ DATA ENGINE & LOGIC ============

@st.cache_data
def load_data(filepath):
    try:
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip')
        except:
            df = pd.read_csv(filepath, error_bad_lines=False, engine='python')
        
        # 1. Clean Headers
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        
        # 2. Smart Column Renaming
        column_mapping = {
            'Physical limitations': ['physical limitations', 'Physical Limitations', 'limitations', 'Physical Limitation', 'injuries'],
            'is_not_suitable_for': ['is_not_suitable_for', 'Is Not Suitable For', 'Not Suitable For', 'Medical Contraindications'],
            'Tags': ['Tags', 'tags', 'Tag', 'tag'],
            'Equipments': ['Equipment', 'equipment', 'equipments'],
            'Exercise Name': ['Exercise name', 'exercise name', 'Name', 'name'],
            'Age Suitability': ['Age suitability', 'Age', 'age'],
            'Primary Category': ['Primary category', 'Category', 'category'],
            'Body Region': ['Body region', 'Muscle Group', 'Target'],
            'MET value': ['MET Value', 'Met Value', 'MET', 'met'],
            'Health benefit': ['Health benefit', 'Benefit', 'Benefits', 'benefit'],
            'Safety cue': ['Safety cue', 'Safety', 'safety', 'Warning', 'Cue'],
            # Explicitly added 'Rest intervals' as requested
            'Rest': ['Rest', 'rest', 'Rest intervals', 'Rest interval', 'Rest Time', 'Interval', 'rest_time', 'Rest Period'],
            # --- FIX: ADDED DIFFICULTY MAPPING ---
            'Difficulty': ['Difficulty', 'difficulty', 'Level', 'level', 'Intensity Level', 'intensity']
        }

        for standard, variations in column_mapping.items():
            if standard not in df.columns:
                for v in variations:
                    match = next((col for col in df.columns if col.lower() == v.lower()), None)
                    if match:
                        df.rename(columns={match: standard}, inplace=True)
                        break
        
        # 3. Critical Failsafe
        # --- FIX: ADDED 'Difficulty' TO REQUIRED COLUMNS ---
        required_cols = ['Physical limitations', 'is_not_suitable_for', 'Tags', 'Equipments', 'Exercise Name', 'Primary Category', 'Body Region', 'Age Suitability', 'MET value', 'Goal', 'Safety cue', 'Health benefit', 'Rest', 'Difficulty']
        for col in required_cols:
            if col not in df.columns:
                if col == 'MET value':
                    df[col] = 3.0
                elif col == 'Rest':
                    df[col] = 'None' 
                elif col == 'Difficulty':
                    df[col] = 'Beginner' # Default fallback for difficulty
                else:
                    df[col] = 'None'
        
        # 4. Data Type Conversion
        df['MET value'] = pd.to_numeric(df['MET value'], errors='coerce').fillna(3.0)
        
        text_cols = [c for c in df.columns if c != 'MET value']
        for col in text_cols:
            df[col] = df[col].fillna('').astype(str)
        
        # Clean Tags
        if 'Tags' in df.columns:
            df['Tags'] = df['Tags'].str.strip()
            
        # --- DATA SANITIZATION LAYER ---
        # 1. Fix Body Regions based on Name
        mask_upper_fix = df['Exercise Name'].str.contains('Arm|Shoulder|Neck|Wrist|Elbow', case=False)
        df.loc[mask_upper_fix, 'Body Region'] = 'Upper'
        
        # 2. Fix Categories based on Name
        mask_stretch_fix = df['Exercise Name'].str.contains('Stretch|Yoga|Fold|Butterfly|Bend|Open', case=False)
        df.loc[mask_stretch_fix, 'Primary Category'] = 'Flexibility/Stretching'
        
        # 3. Fix "Balance" disguised as Cardio
        mask_balance_fix = df['Exercise Name'].str.contains('Balance|Stork|Single Leg', case=False)
        df.loc[mask_balance_fix, 'Primary Category'] = 'Balance & Stability'
                
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

def parse_age_suitability(user_age, range_str):
    rs = str(range_str).lower()
    if 'all ages' in rs or not rs or rs == 'nan': return True
    nums = [int(x) for x in re.findall(r'\d+', rs)]
    if '+' in rs and nums: return user_age >= nums[0]
    elif len(nums) == 2: return nums[0] <= user_age <= nums[1]
    elif len(nums) == 1: return user_age <= nums[0] 
    return True

def filter_exercises(df, profile):
    if df.empty: return df
    filtered = df.copy()

    # 1. AGE FILTER
    filtered = filtered[filtered['Age Suitability'].apply(lambda x: parse_age_suitability(profile['age'], x))]

    # 2. EQUIPMENT FILTER
    user_inventory = set([str(e).lower() for e in profile['available_equipment']])
    if "full gym access" not in user_inventory:
        user_inventory.add('bodyweight')
        user_inventory.add('none')

        def is_equipment_compatible(exo_eq_str):
            if not exo_eq_str or exo_eq_str.lower() == 'nan': return True
            required = [x.strip().lower() for x in exo_eq_str.split(',')]
            for req in required:
                found = False
                for item in user_inventory:
                    if item in req or req in item: 
                        found = True
                        break
                if not found: return False
            return True
        filtered = filtered[filtered['Equipments'].apply(is_equipment_compatible)]

    # 3. MEDICAL & INJURY FILTER
    user_conditions = profile.get('medical_conditions', [])
    user_limitations_text = profile.get('physical_limitation', '').lower()
    
    avoid_terms = [c.lower() for c in user_conditions if c != "None"]
    if user_limitations_text: avoid_terms.extend(user_limitations_text.split())

    if avoid_terms:
        for term in avoid_terms:
            if len(term) < 4: continue 
            if 'is_not_suitable_for' in filtered.columns:
                filtered = filtered[~filtered['is_not_suitable_for'].str.contains(term, case=False, na=False)]
            if 'Physical limitations' in filtered.columns:
                filtered = filtered[~filtered['Physical limitations'].str.contains(term, case=False, na=False)]

    # --- FIX: ADDED DIFFICULTY FILTERING ---
    # 4. DIFFICULTY/LEVEL FILTER 
    user_level = str(profile.get('fitness_level', 'Beginner')).lower()
    
    # Define allowed levels based on user selection
    if "beginner" in user_level:
        allowed_levels = ['beginner']
    elif "intermediate" in user_level:
        allowed_levels = ['beginner', 'intermediate']
    else: 
        # Advanced users can usually do everything
        allowed_levels = ['beginner', 'intermediate', 'advanced']
        
    def is_level_suitable(ex_level):
        val = str(ex_level).lower().strip() 
        return any(lvl in val for lvl in allowed_levels)

    if 'Difficulty' in filtered.columns:
        filtered = filtered[filtered['Difficulty'].apply(is_level_suitable)]
    elif 'Tags' in filtered.columns:
        filtered = filtered[filtered['Tags'].apply(is_level_suitable)]

    return filtered

# ============ HELPER CLASSES & CALCULATORS ============

class FitnessAdvisor:
    def __init__(self, df):
        self.df = df
        
    def _get_met_value(self, exercise_name, fitness_level):
        if self.df.empty: return 3.5
        row = self.df[self.df['Exercise Name'].str.lower() == str(exercise_name).lower()]
        if not row.empty:
            return float(row.iloc[0]['MET value'])
        return 3.5

def calculate_performance_calorie_burn(exercise_index: str, day_name: str, advisor: FitnessAdvisor, weight_kg: float) -> float:
    """Calculates real-time calorie burn solving min/sec/reps conflict."""
    if 'logged_performance' not in st.session_state or day_name not in st.session_state.logged_performance:
        return 0.0

    logged_data = st.session_state.logged_performance.get(day_name, {}).get(exercise_index, {})
    actual_sets = logged_data.get('actual_sets', 0)
    actual_val_input = logged_data.get('actual_reps', 0) 
    
    if actual_sets <= 0 or actual_val_input <= 0 or weight_kg <= 0: return 0.0

    plan = st.session_state.all_json_plans.get(day_name)
    if not plan: return 0.0

    # Locate Exercise Data
    ex_data = None
    try:
        parts = exercise_index.split('_')
        section_key = parts[0]
        if section_key == 'main': section_key = 'main_workout' 
        idx = int(parts[-1]) - 1
        
        target_list = plan.get(section_key, [])
        if 0 <= idx < len(target_list):
            ex_data = target_list[idx]
    except: return 0.0

    if not ex_data: return 0.0
        
    met_value = float(ex_data.get('met_value', 3.0))
    if met_value <= 0: met_value = 3.0
    
    planned_reps_str = str(ex_data.get('reps', '')).lower()
    
    total_minutes = 0.0
    
    # 1. Check if planned target is Minutes
    if 'min' in planned_reps_str:
        total_minutes = actual_sets * actual_val_input
        
    # 2. Check if planned target is Seconds
    elif 'sec' in planned_reps_str:
        total_seconds = actual_sets * actual_val_input
        total_minutes = total_seconds / 60.0
        
    # 3. Assume Reps
    else:
        total_seconds = actual_sets * (actual_val_input * 4)
        total_minutes = total_seconds / 60.0
    
    # Formula: Calories = (MET * 3.5 * Weight / 200) * Duration_in_Minutes
    estimated_calories = (met_value * weight_kg * 3.5) / 200 * total_minutes
    return max(0.0, estimated_calories)

def generate_markdown_export(profile: dict, plans_json: dict, progression_tip: str) -> str:
    md_content = f"""# FriskaAI Fitness Plan
## Generated on {datetime.now().strftime('%B %d, %Y')}

---
## 👤 Profile Summary
**Name:** {profile.get('name', 'User')} | **Goal:** {profile.get('primary_goal', 'N/A')}
---
"""
    for day, plan in plans_json.items():
        md_content += f"\n## {day} - {plan.get('main_workout_category', 'Workout')}\n"
        
        md_content += f"\n### 🔥 Warmup\n"
        for ex in plan.get('warmup', []):
            md_content += f"- **{ex['name']}**: {ex['reps']} ({ex['sets']} set)\n"
            
        md_content += f"\n### 💪 Main Workout\n"
        for ex in plan.get('main_workout', []):
            md_content += f"- **{ex['name']}**: {ex['sets']} x {ex['reps']} (Rest: {ex['rest']})\n"
            
        md_content += f"\n### 🧘 Cooldown\n"
        for ex in plan.get('cooldown', []):
            md_content += f"- **{ex['name']}**: {ex['reps']}\n"
            
        md_content += "\n---\n"
    return md_content

# ============ EXPERT ALGORITHM IMPLEMENTATION ============

def get_weekly_split_logic(goal, num_days):
    g = str(goal).lower()
    if "loss" in g:
        if num_days == 1: return ["Cardio"]
        if num_days == 2: return ["Cardio", "Full Body Strength"]
        if num_days == 3: return ["Cardio Focus", "Cardio Focus", "Full Body Strength"]
        if num_days >= 4: return ["Cardio Focus", "Cardio Focus", "Full Body Strength", "Full Body Strength", "Cardio Focus"][:num_days]
    elif "gain" in g or "muscle" in g:
        if num_days == 1: return ["Full Body Strength"]
        if num_days == 2: return ["Upper Body Strength", "Lower Body Strength"]
        if num_days == 3: return ["Full Body Strength", "Full Body Strength", "Full Body Strength"]
        if num_days >= 4: return ["Upper Body Strength", "Lower Body Strength", "Upper Body Strength", "Lower Body Strength", "Full Body"][:num_days]
    else:
        if num_days == 1: return ["Full Body Circuit"]
        if num_days >= 2: return ["Cardio Focus", "Full Body Strength"] + ["General Fitness"]*(num_days-2)
    return ["Full Body Strength"] * num_days

def get_volume_intensity(goal, level):
    # Hardcoded volume/intensity logic
    g = str(goal).lower()
    l = str(level).lower()
    rpe = "5-7"
    sets = "3"
    reps = "10"
    rest = "60s"
    
    if "loss" in g:
        rpe = "4-7"
        reps = "12-15"
        rest = "30-45s"
        sets = "2" if "beginner" in l else "3"
    elif "gain" in g or "muscle" in g:
        rpe = "6-8"
        reps = "8-12"
        rest = "90-120s"
        sets = "3-4"
    return sets, reps, rpe, rest

def generate_workout_json(df, profile):
    schedule_output = {}
    base_pool = filter_exercises(df, profile)
    if base_pool.empty: base_pool = df.copy() 
    
    duration_str = profile.get('session_duration', '30-45 minutes')
    if "15-20" in duration_str: max_main = 2
    elif "20-30" in duration_str: max_main = 3
    elif "30-45" in duration_str: max_main = 4
    else: max_main = 5

    # --- POOL SEGMENTATION ---
    
    # Warmup Tagged
    warmup_tagged = base_pool[base_pool['Tags'].str.contains('Warmup', case=False, na=False)]
    if warmup_tagged.empty: warmup_tagged = base_pool 

    # Cooldown Tagged
    cooldown_tagged = base_pool[base_pool['Tags'].str.contains('Cooldown', case=False, na=False)]
    if cooldown_tagged.empty: cooldown_tagged = base_pool

    # [STRICT MAIN FILTER]: Include ONLY Strength/Cardio categories. Exclude "Stretch", "Balance", "Mobility" etc from Main.
    
    # Allowed Categories
    allowed_cats = 'Strength|Hypertrophy|Power|Cardio|HIIT'
    main_pool_mask = base_pool['Primary Category'].str.contains(allowed_cats, case=False, na=False)
    
    # Banned Keywords (Name Check)
    banned_keywords = 'Stretch|Yoga|Balance|Mobility|Gentle|Walk|Ankle|Wrist|Neck|Bend|Opener|Fold|Butterfly|Stance|Hold|Reach|Pose|Asana'
    main_pool_mask &= ~base_pool['Exercise Name'].str.contains(banned_keywords, case=False, na=False)
    
    # Banned Tags
    main_pool_mask &= ~base_pool['Tags'].str.contains('Cooldown|Static', case=False, na=False)
    
    main_base_pool = base_pool[main_pool_mask]
    if main_base_pool.empty: main_base_pool = base_pool # Safety fallback

    strength_pool = main_base_pool[main_base_pool['Primary Category'].str.contains('Strength|Hypertrophy|Power', case=False, na=False)]
    cardio_pool = main_base_pool[main_base_pool['Primary Category'].str.contains('Cardio|HIIT', case=False, na=False)]

    used_exercise_names = set()
    days = profile['days_per_week']
    split_types = get_weekly_split_logic(profile['primary_goal'], len(days))
    
    t_sets, t_reps, t_rpe, t_rest = get_volume_intensity(profile['primary_goal'], profile['fitness_level'])

    # Helper to resolve Rest value (Dataset > Default)
    def get_dataset_rest(row_data, default_val):
        ds_val = str(row_data.get('Rest', '')).strip()
        if ds_val and ds_val.lower() != 'nan' and ds_val.lower() != 'none':
            return ds_val
        return default_val

    # Helper: Unique Exercise with Fuzzy Logic + Fallback Control
    def get_unique_exercise(pool, fallback_pool, exclude_set):
        # 1. Try Specific Pool
        avail = pool[~pool['Exercise Name'].isin(exclude_set)]
        
        # Fuzzy Filter
        if not avail.empty:
            fuzzy_pass = []
            for _, row in avail.iterrows():
                name = row['Exercise Name']
                is_dup = False
                for used in exclude_set:
                    # >85% Similarity check
                    if difflib.SequenceMatcher(None, name.lower(), used.lower()).ratio() > 0.85:
                        is_dup = True
                        break
                if not is_dup:
                    fuzzy_pass.append(row)
            if fuzzy_pass:
                avail = pd.DataFrame(fuzzy_pass)
            else:
                avail = pd.DataFrame()

        # 2. Try Fallback if empty (BUT ONLY if fallback is provided)
        if avail.empty and fallback_pool is not None:
            avail = fallback_pool[~fallback_pool['Exercise Name'].isin(exclude_set)]
             # Fuzzy Filter Fallback
            if not avail.empty:
                fuzzy_pass = []
                for _, row in avail.iterrows():
                    name = row['Exercise Name']
                    is_dup = False
                    for used in exclude_set:
                        if difflib.SequenceMatcher(None, name.lower(), used.lower()).ratio() > 0.85:
                            is_dup = True
                            break
                    if not is_dup:
                        fuzzy_pass.append(row)
                if fuzzy_pass:
                    avail = pd.DataFrame(fuzzy_pass)
                else:
                    return None
            else:
                 return None

        if avail.empty: return None
        
        selected = avail.sample(1).iloc[0]
        exclude_set.add(selected['Exercise Name'])
        return selected
    
    # --- HELPER: Parse Reps/Time & Calculate Calories ---
    def build_exercise_dict(ex_row, sets_str, reps_str, rest_str, rpe_val):
        # 1. Parse Sets
        try:
            sets_count = int(re.search(r'\d+', str(sets_str)).group())
        except:
            sets_count = 1
            
        # 2. Parse Reps/Duration
        reps_clean = str(reps_str).lower().strip()
        # Extract number
        nums = re.findall(r'\d+', reps_clean)
        if nums:
            # Take average if range (e.g. 10-15 -> 12.5), else take value
            val_nums = [float(x) for x in nums]
            base_val = sum(val_nums) / len(val_nums)
        else:
            base_val = 10.0 # Fallback
            
        units_per_set = base_val
        total_units = sets_count * units_per_set
        
        # 3. Estimate Duration (Minutes) for Calorie Calc
        duration_mins = 0.0
        if 'min' in reps_clean:
            duration_mins = total_units
        elif 'sec' in reps_clean:
            duration_mins = total_units / 60.0
        else:
            # Assume Reps. Estimate 4 sec/rep
            duration_mins = (total_units * 4.0) / 60.0
            
        # 4. Calculate Calories
        met = float(ex_row.get('MET value', 3.0))
        weight = profile.get('weight_kg', 70.0)
        # Formula: (MET * 3.5 * Weight / 200) * Duration_mins
        cal_val = (met * 3.5 * weight / 200.0) * duration_mins
        cal_int = int(round(cal_val))
        if cal_int < 1: cal_int = 1
        
        return {
            "name": ex_row['Exercise Name'],
            "benefit": ex_row.get('Health benefit', 'Benefit'),
            "steps": str(ex_row.get('Steps to perform', '')).split('\n'),
            "sets": sets_str,
            "reps": reps_str,
            "intensity_rpe": f"RPE {rpe_val}", # Renamed key
            "rest": rest_str,
            "equipment": ex_row.get('Equipments', 'Bodyweight Only'), # Added key
            "est_calories": f"Est: {cal_int} Cal", # Added key
            "safety_cue": ex_row.get('Safety cue', 'None'),
            "planned_sets_count": sets_count, # Added key
            "planned_units_per_set": units_per_set, # Added key
            "planned_total_units": total_units, # Added key
            "planned_total_cal": cal_int, # Added key
            "met_value": met
        }

    for i, day in enumerate(days):
        day_type = split_types[i]
        day_plan = {
            "day_name": day,
            "warmup_duration": "5-7 mins",
            "main_workout_category": day_type,
            "cooldown_duration": "5 mins",
            "warmup": [], "main_workout": [], "cooldown": [],
            "safety_notes": ["Stay hydrated", "Monitor RPE"]
        }

        # --- 1. WARMUP (Order: Cardio -> Upper Mob -> Lower Mob) ---
        
        # Slot 1: Cardio
        w1_pool = base_pool[base_pool['Primary Category'].str.contains('Cardio|HIIT', case=False, na=False)]
        w1_pool = w1_pool[~w1_pool['Exercise Name'].str.contains('Stretch|Lying|Seated|Gentle|Balance|Hold|Stance', case=False, na=False)]
        
        w1 = get_unique_exercise(w1_pool, None, used_exercise_names) 
        if w1 is not None:
            day_plan['warmup'].append(build_exercise_dict(
                w1, "1", "2-3 mins", get_dataset_rest(w1, "None"), "3-4"
            ))

        # Slot 2: Upper Mobility
        w2_pool = warmup_tagged[warmup_tagged['Body Region'].str.contains('Upper', case=False, na=False)]
        fallback_upper = base_pool[(base_pool['Primary Category'] == 'Mobility') & (base_pool['Body Region'].str.contains('Upper', case=False, na=False))]
        w2 = get_unique_exercise(w2_pool, fallback_upper, used_exercise_names)
        if w2 is not None:
            day_plan['warmup'].append(build_exercise_dict(
                w2, "1", "10-15 reps", get_dataset_rest(w2, "None"), "2-3"
            ))

        # Slot 3: Lower Mobility
        w3_pool = warmup_tagged[warmup_tagged['Body Region'].str.contains('Lower', case=False, na=False)]
        w3_pool = w3_pool[~w3_pool['Exercise Name'].str.contains('Arm|Shoulder|Neck', case=False, na=False)]
        
        fallback_lower = base_pool[(base_pool['Primary Category'] == 'Mobility') & (base_pool['Body Region'].str.contains('Lower', case=False, na=False))]
        fallback_lower = fallback_lower[~fallback_lower['Exercise Name'].str.contains('Arm|Shoulder|Neck', case=False, na=False)]

        w3 = get_unique_exercise(w3_pool, fallback_lower, used_exercise_names)
        if w3 is not None:
            day_plan['warmup'].append(build_exercise_dict(
                w3, "1", "10-15 reps", get_dataset_rest(w3, "None"), "2-3"
            ))

        # [NEW FIX] Ensure exactly 3 Warmup Exercises
        fill_attempts = 0
        while len(day_plan['warmup']) < 3 and fill_attempts < 20:
            fill_attempts += 1
            # Attempt to fill with any valid Mobility or Warmup exercise
            fallback_w_pool = base_pool[base_pool['Primary Category'].str.contains('Warmup|Mobility|Cardio', case=False, na=False)]
            w_fill = get_unique_exercise(fallback_w_pool, base_pool, used_exercise_names)
            if w_fill is not None:
                 day_plan['warmup'].append(build_exercise_dict(
                    w_fill, "1", "2-3 mins", get_dataset_rest(w_fill, "None"), "2-3"
                ))

        # --- 2. MAIN WORKOUT ---
        target_pool = strength_pool
        if "Cardio" in day_type:
            target_pool = cardio_pool
        elif "Upper" in day_type:
            target_pool = strength_pool[strength_pool['Body Region'].str.contains('Upper', case=False, na=False)]
        elif "Lower" in day_type:
            target_pool = strength_pool[strength_pool['Body Region'].str.contains('Lower', case=False, na=False)]

        if target_pool.empty: target_pool = main_base_pool

        for _ in range(max_main):
            row_ex = get_unique_exercise(target_pool, main_base_pool, used_exercise_names)
            if row_ex is None: break

            is_cardio = 'cardio' in str(row_ex['Primary Category']).lower() or 'run' in row_ex['Exercise Name'].lower()
            
            if is_cardio:
                is_run = 'run' in row_ex['Exercise Name'].lower()
                sets_val = "1" if is_run else "3"
                reps_val = "20-30 mins" if is_run else "30-45 sec"
                rest_default = "None" if is_run else "30s"
            else:
                sets_val = t_sets
                reps_val = t_reps
                rest_default = t_rest
            
            # Applied logic: Fetch dataset rest, fallback to algo default
            final_rest = get_dataset_rest(row_ex, rest_default)

            day_plan['main_workout'].append(build_exercise_dict(
                row_ex, sets_val, reps_val, final_rest, t_rpe
            ))

        # --- 3. COOLDOWN ---
        c1_pool = cooldown_tagged[cooldown_tagged['Body Region'].str.contains('Full', case=False, na=False)]
        c1_pool = c1_pool[~c1_pool['Exercise Name'].str.contains('Dynamic|Active|Kick', case=False, na=False)]
        fallback_c1 = base_pool[(base_pool['Primary Category'].str.contains('Flexibility|Stretch')) & (base_pool['Body Region'].str.contains('Full', case=False, na=False))]
        
        c1 = get_unique_exercise(c1_pool, fallback_c1, used_exercise_names)
        if c1 is not None:
             day_plan['cooldown'].append(build_exercise_dict(
                c1, "1", "Hold 30-45 sec", get_dataset_rest(c1, "None"), "1-2"
            ))

        c2_pool = cooldown_tagged[cooldown_tagged['Body Region'].str.contains('Upper', case=False, na=False)]
        c2_pool = c2_pool[~c2_pool['Exercise Name'].str.contains('Dynamic|Active|Kick', case=False, na=False)]
        fallback_c2 = base_pool[(base_pool['Primary Category'].str.contains('Flexibility|Stretch')) & (base_pool['Body Region'].str.contains('Upper', case=False, na=False))]
        
        c2 = get_unique_exercise(c2_pool, fallback_c2, used_exercise_names)
        if c2 is not None:
             day_plan['cooldown'].append(build_exercise_dict(
                c2, "1", "Hold 30-45 sec", get_dataset_rest(c2, "None"), "1-2"
            ))

        c3_pool = cooldown_tagged[cooldown_tagged['Body Region'].str.contains('Lower', case=False, na=False)]
        c3_pool = c3_pool[~c3_pool['Exercise Name'].str.contains('Dynamic|Active|Kick', case=False, na=False)]
        fallback_c3 = base_pool[(base_pool['Primary Category'].str.contains('Flexibility|Stretch')) & (base_pool['Body Region'].str.contains('Lower', case=False, na=False))]
        
        c3 = get_unique_exercise(c3_pool, fallback_c3, used_exercise_names)
        if c3 is not None:
             day_plan['cooldown'].append(build_exercise_dict(
                c3, "1", "Hold 30-45 sec", get_dataset_rest(c3, "None"), "1-2"
            ))
            
        # [NEW FIX] Ensure exactly 3 Cooldown Exercises
        fill_attempts = 0
        while len(day_plan['cooldown']) < 3 and fill_attempts < 20:
            fill_attempts += 1
            # Attempt to fill with any valid Flexibility or Cooldown exercise
            fallback_c_pool = base_pool[base_pool['Primary Category'].str.contains('Cooldown|Flexibility|Stretch|Yoga', case=False, na=False)]
            c_fill = get_unique_exercise(fallback_c_pool, base_pool, used_exercise_names)
            if c_fill is not None:
                 day_plan['cooldown'].append(build_exercise_dict(
                    c_fill, "1", "Hold 30-45 sec", get_dataset_rest(c_fill, "None"), "1-2"
                ))

        schedule_output[day] = day_plan
        
    return schedule_output

# --- NEW HELPER: Fuzzy Image Matcher ---
def get_fuzzy_image_path(exercise_name):
    if not os.path.exists(IMAGE_FOLDER):
        return None
    
    try:
        # Get all valid image files
        files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Create map of "clean_name" -> "actual_filename"
        file_map = {os.path.splitext(f)[0].lower(): f for f in files}
        
        # Use existing difflib to find closest match
        matches = difflib.get_close_matches(exercise_name.lower(), file_map.keys(), n=1, cutoff=0.6)
        
        if matches:
            best_match_key = matches[0]
            return os.path.join(IMAGE_FOLDER, file_map[best_match_key])
            
    except Exception:
        return None
        
    return None

def display_interactive_workout_day(day_name: str, plan_json: dict, profile: dict, advisor: FitnessAdvisor):
    weight_kg = profile.get('weight_kg', 70.0)
    if day_name not in st.session_state.logged_performance:
        st.session_state.logged_performance[day_name] = {}

    def render_section(section_data, title, key):
        if not section_data: return
        st.markdown(f"### {title}")
        for idx, ex in enumerate(section_data):
            ex_id = f"{key}_{idx+1}"
            if ex_id not in st.session_state.logged_performance[day_name]:
                st.session_state.logged_performance[day_name][ex_id] = {'actual_sets': 0, 'actual_reps': 0}
            
            planned_sets_str = str(ex.get('sets', '1'))
            planned_reps_str = str(ex.get('reps', '10')).lower()
            try: d_sets = int(re.search(r'\d+', planned_sets_str).group())
            except: d_sets = 1
            try: d_val = int(re.search(r'\d+', planned_reps_str).group())
            except: d_val = 10
            
            if 'min' in planned_reps_str:
                label_unit = "Minutes"
                if d_val > 60: d_val = 2
            elif 'sec' in planned_reps_str:
                label_unit = "Seconds"
                if d_val < 5: d_val = 30
            else:
                label_unit = "Reps"

            # --- FIX: UPDATED LAYOUT FOR IMAGE DISPLAY ---
            with st.container():
                # Adjusted columns: [Image, Details, Input, Stats]
                c_img, c1, c2, c3 = st.columns([1, 3, 1.5, 1.5]) 
                
                # Column 0: Image
                with c_img:
                    img_path = get_fuzzy_image_path(ex['name'])
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    else:
                        st.write("🖼️ No Image")

                with c1:
                    st.markdown(f"**{ex['name']}**")
                    st.caption(f"**Benefit:** {ex.get('benefit', 'N/A')}")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.markdown(f"**Sets:** {ex.get('sets', '1')}")
                    m2.markdown(f"**Target:** {ex.get('reps', '10')}")
                    m3.markdown(f"**RPE:** {ex.get('intensity_rpe', ex.get('rpe', 'N/A'))}")
                    m4.markdown(f"**Rest:** {ex.get('rest', 'N/A')}")
                    
                    with st.expander("Instructions & Safety"):
                        st.write(ex.get('steps', []))
                        if ex.get('safety_cue') and ex.get('safety_cue') != 'None':
                            st.error(f"⚠️ **Safety:** {ex.get('safety_cue')}")
                with c2:
                    val_s = st.number_input("Sets", 0, 10, d_sets, key=f"s_{day_name}_{ex_id}")
                    val_r = st.number_input(label_unit, 0, 300, d_val, key=f"r_{day_name}_{ex_id}")
                    st.session_state.logged_performance[day_name][ex_id] = {'actual_sets': val_s, 'actual_reps': val_r}
                with c3:
                    burned = calculate_performance_calorie_burn(ex_id, day_name, advisor, weight_kg)
                    st.metric("🔥 Burned", f"{int(burned)} kcal")
                st.divider()

    render_section(plan_json.get('warmup', []), "🔥 Warmup", "warmup")
    render_section(plan_json.get('main_workout', []), "💪 Main Workout", "main")
    render_section(plan_json.get('cooldown', []), "🧘 Cooldown", "cooldown")

# ============ MAIN APP EXECUTION ============

if 'user_profile' not in st.session_state: st.session_state.user_profile = {}
if 'all_json_plans' not in st.session_state: st.session_state.all_json_plans = {}
if 'logged_performance' not in st.session_state: st.session_state.logged_performance = {}

# Replaced 'with st.form' with direct layout to allow interactivity
bmi_placeholder = st.empty()
profile = st.session_state.user_profile
cw = profile.get('weight_kg', 70.0)
ch = profile.get('height_cm', 170.0)
if cw > 0 and ch > 0: bmi_placeholder.info(f"📊 BMI: {cw/((ch/100)**2):.1f}")

st.subheader("📋 Profile")
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Name", value=profile.get('name', ''))
    age = st.number_input("Age", 13, 100, value=profile.get('age', 30))
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
with col2:
    u_sys = st.radio("Units", ["Metric (kg/cm)", "Imperial (lbs/in)"])
    if "Metric" in u_sys:
        w = st.number_input("Weight (kg)", 30.0, 300.0, value=cw)
        h = st.number_input("Height (cm)", 100.0, 250.0, value=ch)
        wk, hc = w, h
    else:
        w = st.number_input("Weight (lbs)", 66.0, 660.0, value=cw*2.2)
        h = st.number_input("Height (in)", 39.0, 98.0, value=ch/2.54)
        wk, hc = w/2.20462, h*2.54

st.subheader("🎯 Goals & Details")
p_goal = st.selectbox("Primary Goal", PRIMARY_GOALS, index=PRIMARY_GOALS.index(profile.get('primary_goal', 'Weight Loss')) if 'primary_goal' in profile else 0)
s_goal = st.selectbox("Secondary Goal", ["None"]+SECONDARY_GOALS)
bp_opts = ["Upper Body", "Lower Body", "Core", "Full Body"]
targets = st.multiselect("Target Focus", bp_opts, default=profile.get('target_body_parts', ["Full Body"]))
if not targets: targets = ["Full Body"]
level = st.selectbox("Experience Level", list(TRAINING_LEVELS.keys()), index=0)

# Updated Medical Conditions Logic
conds = st.multiselect("Medical Conditions", MEDICAL_CONDITIONS_OPTIONS, default=profile.get('medical_conditions', []))
custom_cond = ""
if "Other" in conds:
    custom_cond = st.text_input("Please specify your other medical condition:")

lims = st.text_area("Physical Limitations", value=profile.get('physical_limitation', ''))
avoid = st.text_area("Exercises to Avoid", value=profile.get('specific_avoidance', ''))

st.subheader("📅 Schedule")
days = st.multiselect("Training Days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], default=["Mon", "Wed", "Fri"])
dur = st.selectbox("Duration", ["15-20 minutes", "20-30 minutes", "30-45 minutes", "45-60 minutes"], index=2)
loc = st.selectbox("Location", ["Home", "Gym", "Outdoor"])
eq = st.multiselect("Equipment", ["Bodyweight Only", "Dumbbells", "Bands", "Kettlebells", "Full Gym Access"], default=["Bodyweight Only"])
if not eq: eq = ["Bodyweight Only"]

# Changed from form_submit_button to regular button
if st.button("🚀 Generate Plan"):
    if not name or not days:
        st.error("Please fill Name and Days.")
    else:
        # --- LOGIC START: Handle 'Other' & Custom Text ---
        final_conds = [c for c in conds if c != "Other"]
        
        # If user selected Other and typed something, add it
        if "Other" in conds and custom_cond.strip():
            final_conds.append(custom_cond.strip())
        
        # Ensure list is valid
        if not final_conds: final_conds = ["None"]
        elif "None" in final_conds and len(final_conds) > 1: 
            final_conds.remove("None")
        # --- LOGIC END ---

        df = load_data(DATASET_FILE)
        if not df.empty:
            # Updated Profile Structure to match sample
            new_prof = {
                "name": name, 
                "age": age, 
                "gender": gender, 
                "weight_kg": wk, 
                "height_cm": hc,
                "bmi": round(wk/((hc/100)**2), 1) if hc > 0 else 0.0,
                "primary_goal": p_goal,
                "secondary_goal": s_goal,
                "target_body_parts": targets,
                "fitness_level": level,
                "medical_conditions": final_conds,
                "physical_limitation": lims,
                "specific_avoidance": avoid,
                "days_per_week": days,
                "session_duration": dur,
                "available_equipment": eq,
                "unit_system": u_sys,
                "workout_location": loc
            }
            st.session_state.user_profile = new_prof
            with st.spinner("Generating..."):
                time.sleep(1)
                st.session_state.all_json_plans = generate_workout_json(df, new_prof)
                st.success("Plan Generated!")

if st.session_state.all_json_plans:
    df_loaded = load_data(DATASET_FILE)
    advisor = FitnessAdvisor(df_loaded)
    tabs = st.tabs(st.session_state.user_profile['days_per_week'])
    for i, day in enumerate(st.session_state.user_profile['days_per_week']):
        with tabs[i]:
            if day in st.session_state.all_json_plans:
                display_interactive_workout_day(day, st.session_state.all_json_plans[day], st.session_state.user_profile, advisor)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        md = generate_markdown_export(st.session_state.user_profile, st.session_state.all_json_plans, "Stay Consistent")
        st.download_button("Download MD", md, "plan.md", "text/markdown")
    with c2:
        # Initialize logged_performance skeleton
        lp_skeleton = {}
        for d in st.session_state.user_profile['days_per_week']:
            lp_skeleton[d] = {}
            if d in st.session_state.all_json_plans:
                p = st.session_state.all_json_plans[d]
                # Warmup
                for idx, _ in enumerate(p.get('warmup', [])):
                    lp_skeleton[d][f"warmup_{idx+1}"] = {"actual_sets": 0, "actual_reps": 0}
                # Main
                for idx, _ in enumerate(p.get('main_workout', [])):
                    lp_skeleton[d][f"main_{idx+1}"] = {"actual_sets": 0, "actual_reps": 0}
                # Cooldown
                for idx, _ in enumerate(p.get('cooldown', [])):
                    lp_skeleton[d][f"cooldown_{idx+1}"] = {"actual_sets": 0, "actual_reps": 0}

        final_export = {
            "profile": st.session_state.user_profile, 
            "plans_json": st.session_state.all_json_plans,
            "logged_performance": lp_skeleton
        }
        js = json.dumps(final_export, indent=4)
        st.download_button("Download JSON", js, "plan.json", "application/json")