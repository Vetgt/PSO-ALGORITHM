import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PSO Shift Scheduler", layout="wide")

# Title and Description
st.title(" PSO Shift Scheduler (Multi-Objective)")
st.markdown("""
This **Intelligent Scheduling System** utilizes **Particle Swarm Optimization (PSO)** to generate optimal staff rosters.

**Optimization Constraints & Objectives:**
1.  1. **Hard Constraint (Mandatory):** No staff shortage allowed (Shortage must be 0). Heavy penalty applied.
2.  2. **Soft Constraint (Preferred):** Minimum 6 staff required when the store is open. Moderate penalty.
3.  3. **Efficiency (Cost):** Minimize total staff count while satisfying constraints.
4.  4. **Shift Rules:** Staff works for **8 consecutive hours** (16 Periods). Two shifts available: **Morning** & **Evening**.
""")

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
DEFAULT_FILENAME = 'dataset.csv' 

# Shift Configuration (Assumption: 1 Period = 30 Minutes)
# Total operations = 28 Periods (14 Hours)
SHIFTS_CONFIG = {
    "Morning": {"start": 0,  "duration": 16}, # Starts Period 0, Ends Period 16 (8 Hours)
    "Evening": {"start": 12, "duration": 16}  # Starts Period 12, Ends Period 28 (Overlap in the afternoon)
}

@st.cache_data
def load_data(file_path):
    data_dict = {}
    # If using default file
    if os.path.exists(file_path):
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, header=None, engine='openpyxl')
            else:
                df = pd.read_csv(file_path, header=None)
        except:
            return None
    else:
        return None
    
    current_dept = None
    temp_data = []
    
    # Parsing logic (AMPL style format)
    for index, row in df.iterrows():
        first_col = str(row[0])
        if "[" in first_col and "]" in first_col:
            if current_dept is not None and temp_data:
                data_dict[current_dept] = np.array(temp_data, dtype=float)
                temp_data = []
            match = re.search(r'\[(\d+)', first_col)
            if match:
                current_dept = int(match.group(1))
        elif first_col.isdigit() and int(float(first_col)) in range(1, 8):
            raw_values = row.iloc[1:29].values 
            clean_values = np.maximum(np.nan_to_num(pd.to_numeric(raw_values, errors='coerce'), nan=0.0), 0)
            temp_data.append(clean_values)
            
    if current_dept is not None and temp_data:
        data_dict[current_dept] = np.array(temp_data, dtype=float)
    return data_dict

def load_uploaded_data(uploaded_file):
    data_dict = {}
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file, header=None)
            
        current_dept = None
        temp_data = []
        for index, row in df.iterrows():
            first_col = str(row[0])
            if "[" in first_col and "]" in first_col:
                if current_dept is not None and temp_data:
                    data_dict[current_dept] = np.array(temp_data, dtype=float)
                    temp_data = []
                match = re.search(r'\[(\d+)', first_col)
                if match:
                    current_dept = int(match.group(1))
            elif first_col.isdigit() and int(float(first_col)) in range(1, 8):
                raw_values = row.iloc[1:29].values 
                clean_values = np.maximum(np.nan_to_num(pd.to_numeric(raw_values, errors='coerce'), nan=0.0), 0)
                temp_data.append(clean_values)
        if current_dept is not None and temp_data:
            data_dict[current_dept] = np.array(temp_data, dtype=float)
        return data_dict
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# ==========================================
# 2. PARTICLE & PSO STRUCTURE
# ==========================================
class Particle:
    def __init__(self, n_days, n_shifts, min_staff, max_staff):
        # Particle predicts STAFF COUNT per Shift (not per hour)
        # Matrix Size: [7 Days x 2 Shift Types]
        self.position = np.random.uniform(min_staff, max_staff, (n_days, n_shifts))
        self.velocity = np.zeros((n_days, n_shifts))
        
        self.pbest_position = self.position.copy()
        self.pbest_score = float('inf')

class PSO_Shift_Scheduler:
    def __init__(self, demand_matrix, n_particles, iterations, w, c1, c2, max_staff_limit):
        self.demand = demand_matrix
        self.rows, self.cols = demand_matrix.shape
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_staff_limit = max_staff_limit # Upper limit for staff count
        
        self.shift_names = list(SHIFTS_CONFIG.keys())
        self.n_shifts = len(self.shift_names)
        
        # Initialize Swarm
        self.swarm = [Particle(self.rows, self.n_shifts, 0, self.max_staff_limit) for _ in range(n_particles)]
        
        self.gbest_position = np.zeros((self.rows, self.n_shifts))
        self.gbest_score = float('inf')
        self.history = []

    def decode_schedule(self, particle_position):
        """
        Decodes [Morning Count, Evening Count] into [28-Period Schedule].
        This ensures staff works 8 consecutive hours (hard-coded duration).
        """
        staff_counts = np.round(particle_position).astype(int)
        full_schedule = np.zeros((self.rows, self.cols))
        
        for day_idx in range(self.rows):
            for shift_idx, shift_name in enumerate(self.shift_names):
                count = staff_counts[day_idx, shift_idx]
                
                # Get shift config
                cfg = SHIFTS_CONFIG[shift_name]
                start = cfg['start']
                end = min(start + cfg['duration'], self.cols)
                
                # Add staff to that time window
                full_schedule[day_idx, start:end] += count
                
        return full_schedule, staff_counts

    def calculate_fitness(self, position):
        # 1. Decode particle position to hourly schedule
        schedule_matrix, staff_counts_matrix = self.decode_schedule(position)
        
        # --- MULTI-OBJECTIVE FUNCTIONS ---
        
        # A. HARD CONSTRAINT: Shortage
        # PENALTY: 100,000 per missing man-hour.
        shortage_matrix = np.maximum(0, self.demand - schedule_matrix)
        total_shortage = np.sum(shortage_matrix)
        hard_penalty = total_shortage * 100000 
        
        # B. SOFT CONSTRAINT: Min Staff (Store coverage)
        # PENALTY: 500 if staff < 6 during active hours.
        active_periods = self.demand > 0
        under_min_staff = (schedule_matrix < 6) & active_periods
        soft_penalty = np.sum(under_min_staff) * 500
        
        # C. COST EFFICIENCY: Total Staff
        # COST: 10 per staff member.
        total_staff_assigned = np.sum(staff_counts_matrix)
        efficiency_cost = total_staff_assigned * 10
        
        # Total Fitness (Lower is better)
        final_fitness = hard_penalty + soft_penalty + efficiency_cost
        
        return final_fitness

    def run(self, progress_bar=None):
        for i in range(self.iterations):
            for particle in self.swarm:
                fitness = self.calculate_fitness(particle.position)
                
                # Update Personal Best
                if fitness < particle.pbest_score:
                    particle.pbest_score = fitness
                    particle.pbest_position = particle.position.copy()
                
                # Update Global Best
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = particle.position.copy()
            
            self.history.append(self.gbest_score)
            
            # Move Particles (PSO Logic)
            for particle in self.swarm:
                r1 = np.random.rand(*particle.position.shape)
                r2 = np.random.rand(*particle.position.shape)
                
                # Velocity Update: Inertia + Cognitive + Social
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.pbest_position - particle.position)) + \
                                    (self.c2 * r2 * (self.gbest_position - particle.position))
                
                # Position Update
                particle.position += particle.velocity
                
                # Clipping: Ensure staff count is within bounds
                particle.position = np.clip(particle.position, 0, self.max_staff_limit)
            
            # Update Progress Bar UI
            if progress_bar:
                progress_bar.progress((i+1)/self.iterations, text=f"Optimizing... Iteration {i+1}/{self.iterations}")
                
        return self.gbest_position

# ==========================================
# 3. UI SIDEBAR (CONTROLS)
# ==========================================
st.sidebar.header(" Control Panel")

# Load File
datasets = None
# Option 1: Load Default
if os.path.exists(DEFAULT_FILENAME):
    datasets = load_data(DEFAULT_FILENAME)
    if datasets:
        st.sidebar.success(f"Loaded: {DEFAULT_FILENAME}")

# Option 2: Upload
uploaded_file = st.sidebar.file_uploader("Or Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
if uploaded_file:
    datasets = load_uploaded_data(uploaded_file)
    if datasets:
        st.sidebar.info("Using Uploaded File")

if datasets:
    # Department Selection
    dept_options = list(datasets.keys())
    selected_dept = st.sidebar.selectbox("Select Department:", dept_options)
    
    st.sidebar.markdown("---")
    
    # Basic Parameters (UPDATED LIMITS TO 1)
    st.sidebar.subheader("PSO Parameters")
    n_particles = st.sidebar.slider("Number of Particles", 1, 100, 30, help="Min 1 for testing failure cases.")
    iterations = st.sidebar.slider("Iterations", 1, 500, 150, help="Min 1 for testing failure cases.")
    max_staff_limit = st.sidebar.slider("Max Staff Limit per Shift", 20, 100, 50, help="Upper limit to prevent overloading.")
    
    # Advanced Parameters (UPDATED LIMITS TO 0.0)
    with st.sidebar.expander(" Advanced Parameters"):
        st.markdown("**Particle Behavior Settings:**")
        w = st.slider("Inertia Weight (w)", 0.0, 1.0, 0.7, help="0.0 = Freeze, 1.0 = Chaos")
        c1 = st.slider("Cognitive (c1)", 0.0, 3.0, 1.5, help="0.0 = No Memory")
        c2 = st.slider("Social (c2)", 0.0, 3.0, 1.5, help="0.0 = No Cooperation")

    # Execution Button
    if st.button(" START OPTIMIZATION", type="primary"):
        demand_matrix = datasets[selected_dept]
        
        # Initialize PSO
        scheduler = PSO_Shift_Scheduler(demand_matrix, n_particles, iterations, w, c1, c2, max_staff_limit)
        
        # Progress Bar
        bar = st.progress(0, "Initializing swarm...")
        start_time = time.time()
        
        # Run Algorithm
        best_shift_counts = scheduler.run(bar)
        
        end_time = time.time()
        st.success(f"Optimization finished in {end_time - start_time:.2f} seconds!")
        
        # Decode Final Results
        final_schedule, final_counts = scheduler.decode_schedule(best_shift_counts)
        
        # ==========================================
        # 4. RESULTS & VISUALIZATION
        # ==========================================
        
        # --- TAB 1: CONVERGENCE GRAPH ---
        st.subheader("1. Convergence Graph (Penalty Reduction)")
        st.line_chart(scheduler.history)
        st.caption("A downward slope indicates the algorithm is successfully learning and minimizing penalties.")

        # --- TAB 2: STAFF RECOMMENDATION ---
        st.subheader("2. Recommended Staff Count (Shift Tickets)")
        st.write("This is the number of staff you need to assign for each shift block (8-hour work).")
        
        df_counts = pd.DataFrame(final_counts, 
                                 index=[f"Day {i+1}" for i in range(7)],
                                 columns=["Morning Shift (08:00-16:00)", "Evening Shift (14:00-22:00)"]).astype(int)
        st.dataframe(df_counts, use_container_width=True)

        # --- TAB 3: DETAILED SCHEDULE ---
        st.subheader("3. Detailed Schedule (28 Periods)")
        st.write("Detailed view of active staff count for every 30-minute period.")
        
        rows_label = [f"Day {i+1}" for i in range(7)]
        cols_label = [f"P{j+1}" for j in range(28)]
        df_schedule = pd.DataFrame(final_schedule, index=rows_label, columns=cols_label).astype(int)
        st.dataframe(df_schedule, use_container_width=True)
        
        # Download Button
        csv = df_schedule.to_csv().encode('utf-8')
        st.download_button(
            label=" Download Detailed Schedule (CSV)",
            data=csv,
            file_name=f'final_schedule_dept_{selected_dept}.csv',
            mime='text/csv',
        )
        
        # --- TAB 4: CONSTRAINT ANALYSIS ---
        st.markdown("---")
        st.subheader("4. Constraint Analysis & Quality Check")
        
        # Calculate Metrics
        shortage_matrix = np.maximum(0, demand_matrix - final_schedule)
        total_shortage = np.sum(shortage_matrix)
        
        active_periods = demand_matrix > 0
        under_min_staff = (final_schedule < 6) & active_periods
        soft_violations = np.sum(under_min_staff)
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Hard Constraint (Shortage)", int(total_shortage), help="Must be 0. If > 0, the schedule is invalid.")
        col2.metric("Soft Constraint (Staff < 6)", int(soft_violations), help="Should be minimized.")
        col3.metric("Total Staff Assigned", int(np.sum(final_counts)), help="Total workforce cost.")
        
        # Automated Conclusion
        if total_shortage == 0:
            st.success(" **STATUS: OPTIMAL & SAFE.** This schedule meets all demand requirements (Shortage = 0).")
        else:
            st.error(f" **STATUS: UNSAFE.** There is still a shortage of {int(total_shortage)} man-hours.")
            st.warning("Recommendation: Try increasing 'Max Staff Limit' or 'Iterations'.")

        # Heatmap Gap
        st.write("**Gap Analysis Heatmap (Red = Understaffed/Shortage):**")
        gap_matrix = final_schedule - demand_matrix
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(gap_matrix, cmap="RdBu", center=0, annot=True, fmt=".0f", cbar=True, ax=ax)
        st.pyplot(fig)
else:
    st.info("Please upload a dataset to proceed.")
