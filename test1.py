import streamlit as st
import pandas as pd
import random
import re
import time
import json
import difflib  # Added for Fuzzy Logic
from datetime import datetime

# ============ CONFIGURATION & CONSTANTS ============
DATASET_FILE = "Newdata 1.csv"

# CONSTANTS 
PRIMARY_GOALS = ["Weight Loss", "Weight Gain", "Weight Maintenance"]
SECONDARY_GOALS = ["Stress Reduction", "Sleep Improvement", "Athletic Performance", "Posture Correction"]
MEDICAL_CONDITIONS_OPTIONS = ["None", "Diabetes", "Hypertension", "Asthma", "Arthritis", "Back Pain", "Knee Pain"]

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
            'Sets': ['Sets', 'sets', 'Set', 'set', 'Recommended Sets'],
            'Reps': ['Reps', 'reps', 'Rep', 'rep', 'Duration', 'duration', 'Recommended Reps']
        }

        for standard, variations in column_mapping.items():
            if standard not in df.columns:
                for v in variations:
                    match = next((col for col in df.columns if col.lower() == v.lower()), None)
                    if match:
                        df.rename(columns={match: standard}, inplace=True)
                        break
        
        # 3. Critical Failsafe
        required_cols = ['Physical limitations', 'is_not_suitable_for', 'Tags', 'Equipments', 'Exercise Name', 'Primary Category', 'Body Region', 'Age Suitability', 'MET value', 'Goal', 'Safety cue']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'None' if col != 'MET value' else 3.0
        
        # Ensure Sets/Reps exist
        if 'Sets' not in df.columns: df['Sets'] = ''
        if 'Reps' not in df.columns: df['Reps'] = ''

        # 4. Data Type Conversion
        df['MET value'] = pd.to_numeric(df['MET value'], errors='coerce').fillna(3.0)
        
        text_cols = ['Age Suitability', 'Goal', 'Primary Category', 'Body Region', 'Equipments', 'Fitness Level', 'Physical limitations', 'is_not_suitable_for', 'Tags', 'Safety cue', 'Exercise Name', 'Sets', 'Reps']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                
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
    """Calculates real-time calorie burn with strict Duration (min/sec) vs Reps handling."""
    if 'logged_performance' not in st.session_state or day_name not in st.session_state.logged_performance:
        return 0.0

    logged_data = st.session_state.logged_performance.get(day_name, {}).get(exercise_index, {})
    actual_sets = logged_data.get('actual_sets', 0)
    actual_val_input = logged_data.get('actual_reps', 0) # Reps OR Duration Value
    
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
    
    # Determine Type from the Plan Data
    planned_reps_str = str(ex_data.get('reps', '')).lower()
    
    total_minutes = 0.0
    
    # CASE 1: Minutes (e.g., Warmup Cardio "2-3 mins")
    if 'min' in planned_reps_str:
        # Input is assumed to be in minutes
        total_minutes = actual_sets * actual_val_input
        
    # CASE 2: Seconds (e.g., Cooldown "30s", Main Cardio "45s")
    elif 'sec' in planned_reps_str or 'hold' in str(ex_data.get('name', '')).lower():
        # Input is assumed to be in seconds
        total_seconds = actual_sets * actual_val_input
        total_minutes = total_seconds / 60.0
        
    # CASE 3: Reps (Strength)
    else:
        # Estimate: 4 seconds per rep (1s concentric, 1s eccentric, 2s pause/setup)
        total_seconds = actual_sets * (actual_val_input * 4)
        total_minutes = total_seconds / 60.0
    
    # Calorie Formula: (MET * 3.5 * Weight) / 200 * Minutes
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
    g = str(goal).lower()
    l = str(level).lower()
    rpe = "5-7"
    sets = "3"
    reps = "10"
    rest = "60s"
    
    if "loss" in g:
        rpe = "3-6"
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
    
    # Calculate Algo Volume
    duration_str = profile.get('session_duration', '30-45 minutes')
    if "15-20" in duration_str: max_main = 2
    elif "20-30" in duration_str: max_main = 3
    elif "30-45" in duration_str: max_main = 4
    else: max_main = 5

    # --- POOL SEGMENTATION (TAG BASED FOR WARMUP/COOLDOWN) ---
    # Warmup: Strictly 'Warmup' tag
    warmup_pool = base_pool[base_pool['Tags'].str.contains('Warmup', case=False, na=False)]
    if warmup_pool.empty: warmup_pool = base_pool # Fallback

    # Cooldown: Strictly 'Cooldown' tag
    cooldown_pool = base_pool[base_pool['Tags'].str.contains('Cooldown', case=False, na=False)]
    if cooldown_pool.empty: cooldown_pool = base_pool # Fallback

    # Strength & Cardio Pools (Category based)
    strength_pool = base_pool[base_pool['Primary Category'].str.contains('Strength|Hypertrophy|Power', case=False, na=False)]
    if strength_pool.empty: strength_pool = base_pool
    
    cardio_pool = base_pool[base_pool['Primary Category'].str.contains('Cardio|HIIT', case=False, na=False)]
    if cardio_pool.empty: cardio_pool = base_pool

    used_exercise_names = set()
    days = profile['days_per_week']
    split_types = get_weekly_split_logic(profile['primary_goal'], len(days))
    t_sets, t_reps, t_rpe, t_rest = get_volume_intensity(profile['primary_goal'], profile['fitness_level'])

    # Helper: Resolve Volume (Smart Fallback)
    def resolve_volume(csv_val, algo_val):
        val = str(csv_val).strip()
        if val in ['', 'nan', 'None', '0', '0.0']:
            return algo_val
        return val

    # Helper: Fuzzy Selection
    def get_fuzzy_unique(pool, count, used_set):
        selected = []
        if pool.empty: return selected
        
        # Shuffle pool to randomize
        candidates = pool.sample(frac=1)
        
        for _, row in candidates.iterrows():
            if len(selected) >= count: break
            
            name = row['Exercise Name']
            is_dup = False
            
            # Fuzzy Check
            for used in used_set:
                ratio = difflib.SequenceMatcher(None, name.lower(), used.lower()).ratio()
                if ratio > 0.85: # Threshold for similarity
                    is_dup = True
                    break
            
            if not is_dup:
                selected.append(row)
                used_set.add(name)
        
        # Fallback if we couldn't find unique ones
        if len(selected) < count:
            remaining = count - len(selected)
            others = pool[~pool['Exercise Name'].isin([r['Exercise Name'] for r in selected])]
            if not others.empty:
                extra = others.head(remaining)
                for _, row in extra.iterrows():
                    selected.append(row)
                    used_set.add(row['Exercise Name'])
                    
        return selected

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

        # --- 1. WARMUP (3 Exercises: General, Upper, Lower if possible) ---
        # We try to get variety from the Warmup Tagged pool
        w_selection = get_fuzzy_unique(warmup_pool, 3, used_exercise_names)
        
        for w_row in w_selection:
            day_plan['warmup'].append({
                "name": w_row['Exercise Name'],
                "benefit": "Warmup",
                "steps": str(w_row.get('Steps to perform', '')).split('\n'),
                "sets": resolve_volume(w_row.get('Sets'), "1"),
                "reps": resolve_volume(w_row.get('Reps'), "2-3 mins"), 
                "met_value": float(w_row.get('MET value', 4.0)),
                "safety_cue": "Start slow"
            })

        # --- 2. MAIN WORKOUT ---
        if "Cardio" in day_type:
            # Cardio Day
            m_selection = get_fuzzy_unique(cardio_pool, max_main, used_exercise_names)
            for row in m_selection:
                is_run = 'run' in row['Exercise Name'].lower() or 'jog' in row['Exercise Name'].lower()
                final_sets = resolve_volume(row.get('Sets'), "1" if is_run else "3")
                if is_run: final_reps = "20-30 mins"
                else: final_reps = resolve_volume(row.get('Reps'), "30-45 sec")

                day_plan['main_workout'].append({
                    "name": row['Exercise Name'],
                    "benefit": "Cardiovascular Endurance",
                    "steps": str(row.get('Steps to perform', '')).split('\n'),
                    "sets": final_sets,
                    "reps": final_reps,
                    "rest": "None" if is_run else "30s",
                    "met_value": float(row.get('MET value', 6.0)),
                    "safety_cue": row.get('Safety cue', 'Pace yourself')
                })
        else:
            # Strength Day
            # Filter pool by body region if needed
            pool = strength_pool
            if "Upper" in day_type: 
                pool = strength_pool[strength_pool['Body Region'].str.contains('Upper', case=False, na=False)]
            elif "Lower" in day_type: 
                pool = strength_pool[strength_pool['Body Region'].str.contains('Lower', case=False, na=False)]
            
            if pool.empty: pool = strength_pool # Fallback

            m_selection = get_fuzzy_unique(pool, max_main, used_exercise_names)
            
            for row in m_selection:
                is_cardio_move = 'cardio' in str(row['Primary Category']).lower() or 'jump' in row['Exercise Name'].lower()
                
                algo_reps = "30-45 sec" if is_cardio_move else t_reps
                algo_sets = "3" if is_cardio_move else t_sets

                final_sets = resolve_volume(row.get('Sets'), algo_sets)
                final_reps = resolve_volume(row.get('Reps'), algo_reps)

                day_plan['main_workout'].append({
                    "name": row['Exercise Name'],
                    "benefit": row.get('Health benefit', 'Strength'),
                    "steps": str(row.get('Steps to perform', '')).split('\n'),
                    "sets": final_sets,
                    "reps": final_reps,
                    "rest": t_rest,
                    "met_value": float(row.get('MET value', 3.5)),
                    "safety_cue": row.get('Safety cue', 'Maintain form')
                })

        # --- 3. COOLDOWN (Strictly Tagged Cooldown) ---
        c_selection = get_fuzzy_unique(cooldown_pool, 2, used_exercise_names)
        
        for row in c_selection:
            day_plan['cooldown'].append({
                "name": row['Exercise Name'],
                "benefit": "Recovery",
                "steps": str(row.get('Steps to perform', '')).split('\n'),
                "sets": resolve_volume(row.get('Sets'), "1"),
                "reps": resolve_volume(row.get('Reps'), "30-45 sec"),
                "met_value": float(row.get('MET value', 1.5)),
                "safety_cue": "Relax"
            })

        schedule_output[day] = day_plan
        
    return schedule_output

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
            
            # Default Values parsing
            try: d_sets = int(re.search(r'\d+', planned_sets_str).group())
            except: d_sets = 1
            try: d_val = int(re.search(r'\d+', planned_reps_str).group())
            except: d_val = 10
            
            # SMART INPUT LABELING
            if 'min' in planned_reps_str:
                label_unit = "Minutes"
                if d_val > 60: d_val = 2 # If parsing error gave huge number
            elif 'sec' in planned_reps_str:
                label_unit = "Seconds"
                if d_val < 5: d_val = 30 # Default to 30s if parsing fails
            else:
                label_unit = "Reps"

            with st.container():
                c1, c2, c3 = st.columns([3, 1.5, 1.5])
                with c1:
                    st.markdown(f"**{ex['name']}**")
                    # SEPARATED FORMATTING
                    st.markdown(f"**Sets:** {ex.get('sets', '1')}")
                    st.markdown(f"**Target:** {ex.get('reps', '10')}")
                    
                    with st.expander("Instructions"):
                        st.write(ex.get('steps', []))
                        st.warning(ex.get('safety_cue', ''))
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

with st.form("fitness_form"):
        # BMI Placeholder
        bmi_placeholder = st.empty()
        profile = st.session_state.user_profile
        
        # Initial BMI Calc
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
        
        # Body Parts
        bp_opts = ["Upper Body", "Lower Body", "Core", "Full Body"]
        targets = st.multiselect("Target Focus", bp_opts, default=profile.get('target_body_parts', ["Full Body"]))
        if not targets: targets = ["Full Body"]
        
        level = st.selectbox("Experience Level", list(TRAINING_LEVELS.keys()), index=0)
        
        conds = st.multiselect("Medical Conditions", MEDICAL_CONDITIONS_OPTIONS, default=profile.get('medical_conditions', []))
        lims = st.text_area("Physical Limitations", value=profile.get('physical_limitation', ''))
        avoid = st.text_area("Exercises to Avoid", value=profile.get('specific_avoidance', ''))
        
        st.subheader("📅 Schedule")
        days = st.multiselect("Training Days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], default=["Mon", "Wed", "Fri"])
        dur = st.selectbox("Duration", ["15-20 minutes", "20-30 minutes", "30-45 minutes", "45-60 minutes"], index=2)
        
        loc = st.selectbox("Location", ["Home", "Gym", "Outdoor"])
        eq = st.multiselect("Equipment", ["Bodyweight Only", "Dumbbells", "Bands", "Kettlebells", "Full Gym Access"], default=["Bodyweight Only"])
        if not eq: eq = ["Bodyweight Only"]
        
        if st.form_submit_button("🚀 Generate Plan"):
            if not name or not days:
                st.error("Please fill Name and Days.")
            else:
                if "None" in conds and len(conds)>1: conds.remove("None")
                if not conds: conds = ["None"]
                
                df = load_data(DATASET_FILE)
                if not df.empty:
                    new_prof = {
                        "name": name, "age": age, "gender": gender, "weight_kg": wk, "height_cm": hc,
                        "primary_goal": p_goal, "secondary_goal": s_goal, "target_body_parts": targets,
                        "fitness_level": level, "medical_conditions": conds, "physical_limitation": lims,
                        "specific_avoidance": avoid, "days_per_week": days, "session_duration": dur,
                        "workout_location": loc, "available_equipment": eq
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
        js = json.dumps({"profile": st.session_state.user_profile, "plan": st.session_state.all_json_plans}, indent=4)
        st.download_button("Download JSON", js, "plan.json", "application/json")
