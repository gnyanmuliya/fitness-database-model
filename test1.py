import streamlit as st
import pandas as pd
import random
import re
import time
import json
from datetime import datetime

# ============ CONFIGURATION & CONSTANTS ============
DATASET_FILE = "Newdata 1.csv"

# CONSTANTS 
PRIMARY_GOALS = ["Weight Loss", "Weight Gain", "Weight Maintenance"]
SECONDARY_GOALS = ["Stress Reduction", "Sleep Improvement", "Athletic Performance", "Posture Correction"]
MEDICAL_CONDITIONS_OPTIONS = ["None", "Diabetes", "Hypertension", "Asthma", "Arthritis", "Back Pain", "Knee Pain"]

TRAINING_LEVELS = {
    "Beginner (0–6 months)": {"rpe_range": "3-5", "description": "Focus on form and stability."},
    "Intermediate (6–24 months)": {"rpe_range": "6-8", "description": "Focus on progressive overload."},
    "Advanced (2+ years)": {"rpe_range": "8-10", "description": "High intensity and complex movements."}
}

st.set_page_config(page_title="FriskaAI Fitness Coach", page_icon="💪", layout="wide")

# ============ DATA ENGINE & LOGIC ============

@st.cache_data
def load_data(filepath):
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        # Normalize headers
        df.columns = [c.strip() for c in df.columns]
        
        # Numeric coercion
        df['MET value'] = pd.to_numeric(df['MET value'], errors='coerce').fillna(3.0)
        
        # String normalization
        text_cols = ['Age Suitability', 'Goal', 'Primary Category', 'Body Region', 'Equipments', 'Fitness Level', 'Physical limitations', 'Safety cue', 'Exercise Name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

def parse_age_suitability(user_age, range_str):
    """Parses '40-59', 'All Ages', '60+' from dataset."""
    rs = range_str.lower()
    if 'all ages' in rs or not rs: return True
    nums = [int(x) for x in re.findall(r'\d+', rs)]
    if '+' in rs and nums: return user_age >= nums[0]
    elif len(nums) == 2: return nums[0] <= user_age <= nums[1]
    elif len(nums) == 1: return user_age <= nums[0] 
    return True

def filter_exercises(df, profile):
    """Strict filtering engine based on User Profile."""
    filtered = df.copy()

    # 1. AGE FILTER
    filtered = filtered[filtered['Age Suitability'].apply(lambda x: parse_age_suitability(profile['age'], x))]

    # 2. EQUIPMENT FILTER
    user_inventory = set([e.lower() for e in profile['available_equipment']])
    
    if "full gym access" not in user_inventory:
        user_inventory.add('bodyweight')
        user_inventory.add('none')

        def is_equipment_compatible(exo_eq_str):
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

    # 3. INJURY FILTER
    user_conditions = profile.get('medical_conditions', [])
    user_limitations_text = profile.get('physical_limitation', '').lower()
    
    avoid_terms = [c.lower() for c in user_conditions if c != "None"]
    if user_limitations_text: avoid_terms.extend(user_limitations_text.split())

    if avoid_terms:
        for term in avoid_terms:
            if len(term) < 4: continue 
            filtered = filtered[~filtered['Physical limitations'].str.contains(term, case=False)]

    return filtered

# ============ HELPER CLASSES & CALCULATORS ============

class FitnessAdvisor:
    """Helper class for MET lookup."""
    def __init__(self, df):
        self.df = df
        
    def _get_met_value(self, exercise_name, fitness_level):
        row = self.df[self.df['Exercise Name'].str.lower() == exercise_name.lower()]
        if not row.empty:
            return float(row.iloc[0]['MET value'])
        return 3.5

def calculate_performance_calorie_burn(exercise_index: str, day_name: str, advisor: FitnessAdvisor, weight_kg: float) -> float:
    """Calculates real-time calorie burn."""
    if 'logged_performance' not in st.session_state or day_name not in st.session_state.logged_performance:
        return 0.0

    logged_data = st.session_state.logged_performance.get(day_name, {}).get(exercise_index, {})
    actual_sets = logged_data.get('actual_sets', 0)
    actual_units_per_set = logged_data.get('actual_reps', 0) 
    
    if actual_sets <= 0 or actual_units_per_set <= 0 or weight_kg <= 0: return 0.0

    plan = st.session_state.all_json_plans.get(day_name)
    if not plan: return 0.0

    # Locate Exercise Data
    ex_data = None
    section_map = {'warmup': plan.get('warmup', []), 'main': plan.get('main_workout', []), 'cooldown': plan.get('cooldown', [])}
    
    try:
        parts = exercise_index.split('_')
        section_key = parts[0]
        if section_key == 'main': section_key = 'main_workout' 
        idx = int(parts[-1]) - 1
        
        target_list = plan.get(section_key if section_key != 'main' else 'main_workout', [])
        if 0 <= idx < len(target_list):
            ex_data = target_list[idx]
    except: return 0.0

    if not ex_data: return 0.0
        
    met_value = float(ex_data.get('met_value', 3.0))
    if met_value <= 0: met_value = 3.0
    
    # Calculate Seconds
    total_seconds = 0.0
    name_lower = ex_data.get('name', '').lower()
    
    # Time-based detection
    is_time_based = False
    if section_key == 'cooldown': is_time_based = True
    elif section_key == 'warmup' and idx == 0: is_time_based = True 
    elif 'hold' in name_lower or 'plank' in name_lower: is_time_based = True

    if is_time_based:
        total_seconds = actual_sets * actual_units_per_set
    else:
        total_seconds = actual_sets * (actual_units_per_set * 5) # 5s per rep est.
        
    total_minutes = total_seconds / 60.0
    estimated_calories = (met_value * weight_kg * 3.5) / 200 * total_minutes
    
    return max(0.0, estimated_calories)

def generate_markdown_export(profile: dict, plans_json: dict, progression_tip: str) -> str:
    """Generate MD content for download."""
    md_content = f"""# FriskaAI Fitness Plan
## Generated on {datetime.now().strftime('%B %d, %Y')}

---

## 👤 Profile Summary

**Name:** {profile.get('name', 'User')}
**Age:** {profile.get('age', 'N/A')} | **Gender:** {profile.get('gender', 'N/A')} | **BMI:** {profile.get('bmi', 'N/A')}

**Primary Goal:** {profile.get('primary_goal', 'N/A')}
**Secondary Goal:** {profile.get('secondary_goal', 'None')}

**Fitness Level:** {profile.get('fitness_level', 'N/A')}
**Training Days:** {', '.join(profile.get('days_per_week', ['N/A']))}
**Session Duration:** {profile.get('session_duration', 'N/A')}

**Medical Conditions:** {', '.join(profile.get('medical_conditions', ['None']))}
**Physical Limitations:** {profile.get('physical_limitation', 'None')}
**Specific Avoidance Advice:** {profile.get('specific_avoidance', 'None')}

---

## 📈 Weekly Progression Goal
**Your Focus for Next Week:** {progression_tip}

---
"""
    for day, plan in plans_json.items():
        md_content += f"\n## {day} - {plan.get('main_workout_category', 'Workout')}\n"
        
        md_content += "\n### 🔥 Warmup\n"
        for ex in plan.get('warmup', []):
            md_content += f"- **{ex['name']}**: {ex['sets']} x {ex['reps']}\n"
            
        md_content += "\n### 💪 Main Workout\n"
        for ex in plan.get('main_workout', []):
            md_content += f"- **{ex['name']}**: {ex['sets']} sets x {ex['reps']} (Rest: {ex['rest']})\n"
            md_content += f"  *Notes: {ex['safety_cue']}*\n"
            
        md_content += "\n### 🧘 Cooldown\n"
        for ex in plan.get('cooldown', []):
            md_content += f"- **{ex['name']}**: {ex['reps']}\n"
            
        md_content += "\n---\n"
        
    md_content += "\n\n*Disclaimer: Consult a physician before starting.*"
    return md_content

# ============ GENERATOR ALGORITHM ============

def generate_workout_json(df, profile):
    """Core Algorithm: Dataset -> JSON Structure."""
    schedule_output = {}
    
    # 1. Base Filter
    base_pool = filter_exercises(df, profile)
    if base_pool.empty: base_pool = df.copy() 
    
    # 2. Global Tracking
    used_exercise_names = set()
    
    # 3. Volume Logic
    duration_str = profile.get('session_duration', '30-45 minutes')
    if "15-20" in duration_str: max_main = 3
    elif "20-30" in duration_str: max_main = 4
    elif "30-45" in duration_str: max_main = 5
    else: max_main = 6
    
    target_parts = [p.lower() for p in profile.get('target_body_parts', ['Full Body'])]
    is_split = "full body" not in target_parts[0] and len(profile['days_per_week']) > 2

    lvl = profile['fitness_level']
    if "Beginner" in lvl:
        t_sets, t_reps, t_rest, t_rpe = "2", "10-12", "60s", "3-5"
    elif "Intermediate" in lvl:
        t_sets, t_reps, t_rest, t_rpe = "3", "8-12", "45s", "6-8"
    else:
        t_sets, t_reps, t_rest, t_rpe = "4", "6-10", "90s", "8-9"

    for day_idx, day in enumerate(profile['days_per_week']):
        day_plan = {
            "day_name": day,
            "warmup_duration": "5-7 minutes",
            "main_workout_category": "General Fitness",
            "cooldown_duration": "5-7 minutes",
            "warmup": [], "main_workout": [], "cooldown": [],
            "safety_notes": ["Stay hydrated", "Focus on form"]
        }

        # --- WARMUP (Strict 3) ---
        # 1. Cardio
        cardio_pool = base_pool[base_pool['Primary Category'].str.contains('Cardio|Warmup', case=False)]
        cardio_ex = cardio_pool.sample(1).iloc[0] if not cardio_pool.empty else base_pool.sample(1).iloc[0]
        
        day_plan['warmup'].append({
            "name": cardio_ex['Exercise Name'],
            "benefit": "Increases heart rate",
            "steps": str(cardio_ex['Steps to perform']).split('\n'),
            "sets": "1",
            "reps": "90 seconds",
            "intensity_rpe": "RPE 4",
            "rest": "None",
            "equipment": "None",
            "est_calories": "Calculating...",
            "met_value": float(cardio_ex.get('MET value', 5.0)),
            "safety_cue": cardio_ex.get('Safety cue', 'Maintain pace')
        })

        # 2. Upper Dynamic
        up_pool = base_pool[
            (base_pool['Primary Category'].str.contains('Mobility|Warmup', case=False)) & 
            (base_pool['Body Region'].str.contains('Upper|Full', case=False)) &
            (~base_pool['Exercise Name'].isin(used_exercise_names))
        ]
        if up_pool.empty: up_pool = base_pool
        up_ex = up_pool.sample(1).iloc[0]
        used_exercise_names.add(up_ex['Exercise Name'])
        
        day_plan['warmup'].append({
            "name": up_ex['Exercise Name'],
            "benefit": "Mobilizes upper body",
            "steps": str(up_ex['Steps to perform']).split('\n'),
            "sets": "1",
            "reps": "10-15 reps",
            "intensity_rpe": "RPE 3",
            "rest": "None",
            "equipment": "Bodyweight",
            "met_value": float(up_ex.get('MET value', 3.0)),
            "safety_cue": up_ex.get('Safety cue', 'Focus on ROM')
        })

        # 3. Lower Dynamic
        low_pool = base_pool[
            (base_pool['Primary Category'].str.contains('Mobility|Warmup', case=False)) & 
            (base_pool['Body Region'].str.contains('Lower|Full', case=False)) &
            (~base_pool['Exercise Name'].isin(used_exercise_names))
        ]
        if low_pool.empty: low_pool = base_pool
        low_ex = low_pool.sample(1).iloc[0]
        used_exercise_names.add(low_ex['Exercise Name'])
        
        day_plan['warmup'].append({
            "name": low_ex['Exercise Name'],
            "benefit": "Mobilizes lower body",
            "steps": str(low_ex['Steps to perform']).split('\n'),
            "sets": "1",
            "reps": "10-15 reps",
            "intensity_rpe": "RPE 3",
            "rest": "15s",
            "equipment": "Bodyweight",
            "met_value": float(low_ex.get('MET value', 3.0)),
            "safety_cue": low_ex.get('Safety cue', 'Smooth movement')
        })

        # --- MAIN WORKOUT ---
        if is_split:
            focus = target_parts[day_idx % len(target_parts)]
            cat_title = f"{focus.title()} Focus"
            main_pool = base_pool[base_pool['Body Region'].str.contains(focus, case=False)]
        else:
            cat_title = "Full Body Circuit"
            main_pool = base_pool
            
        day_plan['main_workout_category'] = cat_title
        
        main_pool = main_pool[main_pool['Primary Category'].str.contains('Strength|Hypertrophy|Power', case=False)]
        main_pool = main_pool[~main_pool['Exercise Name'].isin(used_exercise_names)]
        
        count = min(max_main, len(main_pool))
        if count > 0:
            selected_main = main_pool.sample(count)
            for _, row in selected_main.iterrows():
                used_exercise_names.add(row['Exercise Name'])
                day_plan['main_workout'].append({
                    "name": row['Exercise Name'],
                    "benefit": row.get('Health benefit', 'Strength'),
                    "steps": str(row.get('Steps to perform', '')).split('\n'),
                    "sets": t_sets,
                    "reps": t_reps,
                    "intensity_rpe": f"RPE {t_rpe}",
                    "rest": t_rest,
                    "equipment": row.get('Equipments', 'Bodyweight'),
                    "met_value": float(row.get('MET value', 4.0)),
                    "safety_cue": row.get('Safety cue', 'Check form')
                })

        # --- COOLDOWN (Strict 3) ---
        cool_pool = base_pool[
            (base_pool['Primary Category'].str.contains('Stretching|Yoga|Cool', case=False)) &
            (~base_pool['Exercise Name'].isin(used_exercise_names))
        ]
        if len(cool_pool) < 3: cool_pool = base_pool[base_pool['Primary Category'].str.contains('Stretching', case=False)]
        
        selected_cool = cool_pool.sample(min(3, len(cool_pool)))
        for _, row in selected_cool.iterrows():
            day_plan['cooldown'].append({
                "name": row['Exercise Name'],
                "benefit": "Recovery",
                "steps": str(row.get('Steps to perform', '')).split('\n'),
                "sets": "1",
                "reps": "15-30 sec",
                "intensity_rpe": "RPE 1-2",
                "rest": "None",
                "equipment": "None",
                "met_value": float(row.get('MET value', 1.5)),
                "safety_cue": row.get('Safety cue', 'Relax')
            })
            used_exercise_names.add(row['Exercise Name'])
            
        schedule_output[day] = day_plan
        
    return schedule_output

def display_interactive_workout_day(day_name: str, plan_json: dict, profile: dict, advisor: FitnessAdvisor):
    """Renders interactive log."""
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
            
            planned_sets = str(ex.get('sets', '1'))
            planned_reps = str(ex.get('reps', '10'))
            
            try: d_sets = int(re.search(r'\d+', planned_sets).group())
            except: d_sets = 1
            
            try: 
                d_reps = int(re.search(r'\d+', planned_reps).group())
                u_label = "Seconds" if 'sec' in planned_reps else "Reps"
            except: 
                d_reps = 10
                u_label = "Reps"

            with st.container():
                c1, c2, c3 = st.columns([3, 1.5, 1.5])
                with c1:
                    st.markdown(f"**{ex['name']}**")
                    # CHANGED FORMATTING HERE
                    col_s, col_r = st.columns(2)
                    with col_s: st.write(f"**Sets:** {planned_sets}")
                    with col_r: st.write(f"**Reps:** {planned_reps}")
                    
                    with st.expander("Details"):
                        st.write(ex.get('steps', []))
                        st.warning(f"Safety: {ex.get('safety_cue', '')}")
                with c2:
                    val_s = st.number_input("Sets", 0, 10, d_sets, key=f"s_{day_name}_{ex_id}")
                    val_r = st.number_input(u_label, 0, 300, d_reps, key=f"r_{day_name}_{ex_id}")
                    st.session_state.logged_performance[day_name][ex_id] = {'actual_sets': val_s, 'actual_reps': val_r}
                with c3:
                    burned = calculate_performance_calorie_burn(ex_id, day_name, advisor, weight_kg)
                    st.metric("🔥 Burned", f"{int(burned)} kcal")
                st.divider()

    render_section(plan_json.get('warmup', []), "🔥 Warmup", "warmup")
    render_section(plan_json.get('main_workout', []), "💪 Main Workout", "main")
    render_section(plan_json.get('cooldown', []), "🧘 Cooldown", "cooldown")

# ============ MAIN APP EXECUTION ============

# Initialize Session State
if 'user_profile' not in st.session_state: st.session_state.user_profile = {}
if 'all_json_plans' not in st.session_state: st.session_state.all_json_plans = {}
if 'logged_performance' not in st.session_state: st.session_state.logged_performance = {}

# --- FORM START ---
with st.form("fitness_form"):
        
        # BMI Placeholder initialization
        bmi_placeholder = st.empty()
        
        # --- Default/Current Values from Session State ---
        profile = st.session_state.user_profile
        
        # Calculate initial/re-run BMI for display in the placeholder
        current_weight_kg = profile.get('weight_kg', 70.0)
        current_height_cm = profile.get('height_cm', 170.0)
        current_bmi = 0
        if current_weight_kg > 0 and current_height_cm > 0:
            current_bmi = current_weight_kg / ((current_height_cm / 100) ** 2)
            bmi_placeholder.info(f"📊 Your BMI: {current_bmi:.1f}")
        else:
            bmi_placeholder.info("📊 Your BMI: Enter height and weight.")
        
        # Basic Info
        st.subheader("📋 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name *", placeholder="Your name", key="name_input", value=profile.get('name', ''))
            age = st.number_input("Age *", min_value=13, max_value=100, value=profile.get('age', 30), key="age_input")
            gender_default_index = ["Male", "Female", "Other"].index(profile.get('gender', 'Male'))
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"], key="gender_input", index=gender_default_index)
        
        with col2:
            unit_system_default = profile.get('unit_system', 'Metric (kg, cm)')
            unit_system = st.radio("Units *", ["Metric (kg, cm)", "Imperial (lbs, in)"], key="unit_input", index=["Metric (kg, cm)", "Imperial (lbs, in)"].index(unit_system_default))
            
            weight_kg = 0.0
            height_cm = 0.0
            
            if unit_system == "Metric (kg, cm)":
                weight_kg = st.number_input("Weight (kg) *", min_value=30.0, max_value=300.0, value=profile.get('weight_kg', 70.0), key="weight_kg_input")
                height_cm = st.number_input("Height (cm) *", min_value=100.0, max_value=250.0, value=profile.get('height_cm', 170.0), key="height_cm_input")
            else:
                weight_lbs_default = profile.get('weight_kg', 70.0) / 0.453592 if profile.get('weight_kg') else 154.3
                height_in_default = profile.get('height_cm', 170.0) / 2.54 if profile.get('height_cm') else 66.9
                
                weight_lbs = st.number_input("Weight (lbs) *", min_value=66.0, max_value=660.0, value=weight_lbs_default, key="weight_lbs_input")
                height_in = st.number_input("Height (in) *", min_value=39.0, max_value=98.0, value=height_in_default, key="height_in_input")
                
                weight_kg = weight_lbs * 0.453592
                height_cm = height_in * 2.54
        
        final_bmi = 0
        if weight_kg > 0 and height_cm > 0:
            final_bmi = weight_kg / ((height_cm / 100) ** 2)
            bmi_placeholder.info(f"📊 Your BMI: {final_bmi:.1f}")

        # Goals
        st.subheader("🎯 Fitness Goals")
        col1, col2 = st.columns(2)
        
        # Primary Goal Selection - ERROR FIX: ADD SAFETY CHECK FOR DEFAULT INDEX
        primary_goal_options = PRIMARY_GOALS
        saved_goal = profile.get('primary_goal', 'Weight Maintenance')
        # Check if saved_goal is in options, if not default to index 0
        if saved_goal in primary_goal_options:
            primary_goal_default_index = primary_goal_options.index(saved_goal)
        else:
            primary_goal_default_index = 0
            
        primary_goal = st.selectbox(
            "Primary Goal *",
            primary_goal_options, key="primary_goal_input", index=primary_goal_default_index
        )
        
        # Secondary Goal Selection
        secondary_goal_options = ["None"] + SECONDARY_GOALS
        secondary_goal_default_value = profile.get('secondary_goal', 'None')
        secondary_goal_default_index = secondary_goal_options.index(secondary_goal_default_value if secondary_goal_default_value in secondary_goal_options else 'None')
        secondary_goal = st.selectbox(
            "Secondary Goal (Optional)",
            secondary_goal_options, key="secondary_goal_input", index=secondary_goal_default_index
        )
        
        # New Body Part Selection
        st.subheader("🏋️ Target Focus")
        body_part_options = ["Upper Body", "Lower Body", "Core", "Full Body"]
        body_parts_default = profile.get('target_body_parts', ["Full Body"])
        target_body_parts = st.multiselect(
            "Select Body Parts to Focus On:",
            body_part_options, 
            default=body_parts_default, 
            key="body_parts_input"
        )
        if not target_body_parts:
            target_body_parts = ["Full Body"]

        # Fitness Level (Experience)
        st.subheader("⏱️ Experience Level")
        fitness_level_options = list(TRAINING_LEVELS.keys())
        fitness_level_default_index = fitness_level_options.index(profile.get('fitness_level', 'Beginner (0–6 months)'))
        fitness_level = st.selectbox(
            "Training Experience (Level) *",
            fitness_level_options, key="fitness_level_input", index=fitness_level_default_index
        )
        
        level_info = TRAINING_LEVELS[fitness_level]
        st.info(f"**{fitness_level}** (RPE {level_info['rpe_range']}): {level_info['description']}")
        
        # Medical Conditions
        st.subheader("🏥 Health Screening")
        initial_multiselect_default = profile.get('medical_conditions', [])
        medical_conditions = st.multiselect(
            "Medical Conditions *",
            MEDICAL_CONDITIONS_OPTIONS, 
            default=initial_multiselect_default, 
            key="medical_conditions_input"
        )
        
        # Physical Limitations
        st.warning("⚠️ **Physical Limitations** - Describe ANY injuries, pain, or movement restrictions")
        physical_limitation = st.text_area( 
            "Physical Limitations (Important for Safety) *",
            placeholder="E.g., 'Previous right knee surgery - avoid deep squats'",
            height=100, key="physical_limitation_input", value=profile.get('physical_limitation', '')
        )
        
        # Specific Exercise Avoidance
        st.warning("⚠️ **Specific Exercise Restrictions**")
        initial_avoid_text = profile.get('specific_avoidance', '') 
        if initial_avoid_text == 'None': initial_avoid_text = ''
        specific_avoidance_input = st.text_area(
            "Have you been advised to avoid any specific exercises? (If yes, please list them below):",
            placeholder="E.g., 'Heavy deadlifts, overhead pressing due to shoulder issue...'",
            height=100, key="specific_avoidance_text_input", value=initial_avoid_text
        )
        
        # Training Schedule
        st.subheader("💪 Training Schedule")
        col1, col2 = st.columns(2)
        with col1:
            days_per_week = st.multiselect(
                "Training Days *",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=profile.get('days_per_week', ["Monday", "Wednesday", "Friday"]), key="days_per_week_input"
            )
        with col2:
            session_duration_options = ["15-20 minutes", "20-30 minutes", "30-45 minutes", "45-60 minutes"]
            session_duration_default_index = session_duration_options.index(profile.get('session_duration', '30-45 minutes'))
            session_duration = st.selectbox(
                "Session Duration *",
                session_duration_options, key="session_duration_input", index=session_duration_default_index
            )

        # Workout Location & Equipment
        st.subheader("🗺️ Workout Location")
        location_options = ["Home", "Gym", "Outdoor", "Any"]
        location_default_index = location_options.index(profile.get('workout_location', 'Home'))
        workout_location = st.selectbox(
            "Where will you primarily work out?",
            location_options, key="location_input", index=location_default_index
        )
        
        st.subheader("🏋️ Available Equipment")
        if workout_location == "Gym":
            st.info("✅ **Gym Selected:** We will assume access to standard gym equipment.")
            equipment = ["Full Gym Access", "Machines", "Cables", "Barbells", "Dumbbells", "Bench", "Pull-up Bar", "Kettlebells"]
        else:
            eq_options = ["Bodyweight Only", "Dumbbells", "Resistance Bands", "Kettlebells", "Barbell", "Bench", "Pull-up Bar", "Yoga Mat"]
            equipment = st.multiselect(
                "Select all available equipment:", 
                eq_options, 
                default=profile.get('available_equipment', ["Bodyweight Only"]), 
                key="equipment_input"
            )
            if not equipment: equipment = ["Bodyweight Only"]
        
        # Submit button
        st.markdown("---")
        submit_clicked = st.form_submit_button(
            "🚀 Generate My Fitness Plan",
            use_container_width=True,
            type="primary"
        )
        
        # Process ONLY when button clicked
        if submit_clicked:
            if not name or len(name.strip()) < 2:
                st.error("❌ Please enter your name.")
            elif not days_per_week:
                st.error("❌ Please select at least one training day.")
            elif final_bmi <= 0 or (weight_kg <= 0 or height_cm <= 0):
                st.error("❌ Please ensure valid weight and height inputs.")
            else:
                if "None" in medical_conditions and len(medical_conditions) > 1:
                    medical_conditions.remove("None")
                if not medical_conditions: medical_conditions = ["None"]
                
                # --- START BACKEND GENERATION LOGIC ---
                df = load_data(DATASET_FILE)
                if df.empty:
                    st.stop()
                
                updated_profile = {
                    "name": name, "age": age, "gender": gender, 
                    "weight_kg": weight_kg, "height_cm": height_cm, "bmi": f"{final_bmi:.1f}",
                    "primary_goal": primary_goal, "secondary_goal": secondary_goal,
                    "target_body_parts": target_body_parts, "fitness_level": fitness_level,
                    "medical_conditions": medical_conditions, 
                    "physical_limitation": physical_limitation,
                    "specific_avoidance": specific_avoidance_input,
                    "days_per_week": days_per_week, "session_duration": session_duration,
                    "workout_location": workout_location, "available_equipment": equipment,
                    "unit_system": unit_system
                }
                st.session_state.user_profile = updated_profile
                
                with st.spinner("Processing Dataset & Applying Strict Algorithm..."):
                    time.sleep(1) 
                    plans = generate_workout_json(df, updated_profile)
                    st.session_state.all_json_plans = plans
                    st.success("✅ Precision Plan Generated Successfully!")

# --- DISPLAY OUTPUT ---
if st.session_state.all_json_plans:
    df_loaded = load_data(DATASET_FILE)
    advisor = FitnessAdvisor(df_loaded)
    best_tip = "Consistency > Intensity"
    
    # 1. Tabs Display (First)
    tabs = st.tabs(st.session_state.user_profile['days_per_week'])
    for i, day in enumerate(st.session_state.user_profile['days_per_week']):
        with tabs[i]:
            if day in st.session_state.all_json_plans:
                display_interactive_workout_day(day, st.session_state.all_json_plans[day], st.session_state.user_profile, advisor)

    # 2. Separator
    st.markdown("---")
    
    # 3. Export Options (Last)
    st.subheader("📥 Download Your Plan")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c2:
        md_text = generate_markdown_export(st.session_state.user_profile, st.session_state.all_json_plans, best_tip)
        st.download_button(label="📄 Download Plan (MD)", data=md_text, file_name=f"Plan_{datetime.now().strftime('%Y%m%d')}.md", mime="text/markdown", use_container_width=True)
    with c3:
        json_export = {"profile": st.session_state.user_profile, "plans_json": st.session_state.all_json_plans}
        json_text = json.dumps(json_export, indent=4)
        st.download_button(label="📋 Download Plan (JSON)", data=json_text, file_name=f"Plan_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json", use_container_width=True)