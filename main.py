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
st.set_page_config(page_title="PSO Staff Scheduler", layout="wide")
st.title("🤖 PSO Staff Scheduling Optimizer")

# ==========================================
# FILE CONFIGURATION
# ==========================================
# Ensure this matches your actual file name
DATASET_FILENAME = 'dataset.csv' 

# ==========================================
# 1. DATA LOADING & CLEANING FUNCTION
# ==========================================
@st.cache_data
def load_and_clean_data(file_path):
    data_dict = {}
    
    # Detect file type
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
    else:
        df = pd.read_csv(file_path, header=None)
    
    current_dept = None
    temp_data = []
    
    for index, row in df.iterrows():
        first_col = str(row[0])
        
        # Detect Header Block [ID,*,*]
        if "[" in first_col and "]" in first_col:
            if current_dept is not None and temp_data:
                data_dict[current_dept] = np.array(temp_data, dtype=float)
                temp_data = []
            match = re.search(r'\[(\d+)', first_col)
            if match:
                current_dept = int(match.group(1))
                
        # Detect Data Rows (Weekly 1-7)
        elif first_col.isdigit() and int(float(first_col)) in range(1, 8):
            # Cleaning Data
            raw_values = row.iloc[1:29].values 
            clean_values = pd.to_numeric(raw_values, errors='coerce')
            clean_values = np.nan_to_num(clean_values, nan=0.0)
            clean_values = np.maximum(clean_values, 0)
            temp_data.append(clean_values)
            
    if current_dept is not None and temp_data:
        data_dict[current_dept] = np.array(temp_data, dtype=float)
        
    return data_dict

# ==========================================
# 2. PSO ALGORITHM CLASS
# ==========================================
class Particle:
    def __init__(self, shape, min_val, max_val):
        self.position = np.random.uniform(min_val, max_val, shape)
        self.velocity = np.zeros(shape)
        self.pbest_position = self.position.copy()
        self.pbest_score = float('inf')

class PSO_Scheduler:
    def __init__(self, demand_matrix, n_particles, iterations, w, c1, c2):
        self.demand = demand_matrix
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rows, self.cols = demand_matrix.shape
        # Search Space
        self.min_val = 0
        self.max_val = np.max(demand_matrix) + 5
        
        self.swarm = [Particle((self.rows, self.cols), self.min_val, self.max_val) for _ in range(n_particles)]
        self.gbest_position = np.zeros((self.rows, self.cols))
        self.gbest_score = float('inf')
        self.history = []

    def fitness_function(self, position):
        schedule = np.round(position)
        total_staff = np.sum(schedule)
        shortage = np.maximum(0, self.demand - schedule)
        # Heavy penalty for shortage
        penalty_score = np.sum(shortage) * 10000 
        return total_staff + penalty_score

    def run(self, progress_bar=None):
        for i in range(self.iterations):
            for particle in self.swarm:
                fitness = self.fitness_function(particle.position)
                if fitness < particle.pbest_score:
                    particle.pbest_score = fitness
                    particle.pbest_position = particle.position.copy()
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = particle.position.copy()
            
            self.history.append(self.gbest_score)
            
            for particle in self.swarm:
                r1 = np.random.rand(*particle.position.shape)
                r2 = np.random.rand(*particle.position.shape)
                
                # PSO Movement Equation
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.pbest_position - particle.position)) + \
                                    (self.c2 * r2 * (self.gbest_position - particle.position))
                
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.min_val, self.max_val)
            
            if progress_bar:
                progress_bar.progress((i + 1) / self.iterations, text=f"Optimizing... Iteration {i+1}/{self.iterations}")
                
        return np.round(self.gbest_position)

# ==========================================
# 3. UI SIDEBAR (ENGLISH)
# ==========================================
st.sidebar.header("⚙️ Control Panel")
file_exists = os.path.exists(DATASET_FILENAME)

if not file_exists:
    st.error(f"Dataset file not found: {DATASET_FILENAME}")
    st.stop()

# --- A. BASIC SETTINGS ---
st.sidebar.subheader("1. Basic Settings")
try:
    datasets = load_and_clean_data(DATASET_FILENAME)
    dept_options = list(datasets.keys())
    
    # Hybrid Mode Logic for Dept Selection
    dept_options_ui = ["ALL DEPARTMENTS (Combine)"] + dept_options
    
    selected_option = st.sidebar.selectbox("Select Department:", dept_options)
    
    n_particles = st.sidebar.slider("Particle Count (Swarm)", 10, 100, 30, help="More particles = better search but slower.")
    iterations = st.sidebar.slider("Iterations (Duration)", 50, 500, 100, help="Higher iterations = more stable result.")

    # --- B. ADVANCED SETTINGS (HIDDEN) ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🛠️ Advanced Parameters"):
        st.markdown("*Adjust only if you understand PSO dynamics.*")
        w = st.slider("Inertia Weight (w)", 0.1, 1.0, 0.7, 0.05, help="High = Exploration, Low = Exploitation.")
        c1 = st.slider("Cognitive (c1) - Self", 0.1, 3.0, 1.5, 0.1)
        c2 = st.slider("Social (c2) - Group", 0.1, 3.0, 1.5, 0.1)

    # --- MAIN CONTENT ---
    demand_matrix = datasets[selected_option]
    
    # Info Metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Demand (Man-hours)", int(np.sum(demand_matrix)))
    col2.metric("Matrix Dimensions", f"{demand_matrix.shape}")

    if st.button("🚀 START OPTIMIZATION", type="primary"):
        # Run PSO
        pso = PSO_Scheduler(demand_matrix, n_particles, iterations, w, c1, c2)
        
        bar = st.progress(0, "Initializing...")
        start_time = time.time()
        best_schedule = pso.run(bar)
        end_time = time.time()
        
        st.success(f"Optimization completed in {end_time - start_time:.2f} seconds.")
        
        # --- VISUALIZATION ---
        
        # 1. Convergence
        st.subheader("1. Convergence Graph (Cost Reduction)")
        st.line_chart(pso.history)
        
        # 2. Daily Comparison
        st.subheader("2. Daily Comparison: Demand vs Supply")
        daily_demand = np.sum(demand_matrix, axis=0)
        daily_schedule = np.sum(best_schedule, axis=0)
        
        chart_data = pd.DataFrame({
            "Required (Demand)": daily_demand,
            "Allocated (Supply)": daily_schedule
        })
        st.line_chart(chart_data, color=["#FF4B4B", "#1E90FF"]) # Red & Blue

        # 3. Heatmap Gap
        st.subheader("3. Gap Analysis (Red = Shortage)")
        gap_matrix = best_schedule - demand_matrix
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(gap_matrix, cmap="RdBu", center=0, annot=True, fmt=".0f", cbar=True, ax=ax)
        ax.set_title(f"Staffing Gap (Negative/Red means Understaffed)")
        st.pyplot(fig)
        
        # --- 4. SUMMARY & CONCLUSION (NEW SECTION) ---
        st.markdown("---")
        st.header("📝 4. Executive Summary & Conclusion")
        
        # Calculate Stats
        total_demand_val = np.sum(demand_matrix)
        total_supply_val = np.sum(best_schedule)
        total_shortage_val = np.sum(np.maximum(0, demand_matrix - best_schedule))
        total_surplus_val = np.sum(np.maximum(0, best_schedule - demand_matrix))
        
        # Display Metrics in Columns
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Demand", int(total_demand_val))
        m2.metric("Total Allocated", int(total_supply_val))
        m3.metric("Total Shortage", int(total_shortage_val), delta_color="inverse")
        m4.metric("Total Surplus", int(total_surplus_val))
        
        # Automated Conclusion Logic
        st.write("") # Spacer
        if total_shortage_val == 0:
            # Success Case
            st.success(f"""
            ### ✅ CONCLUSION: OPTIMAL RESULT
            **Great job!** The algorithm has successfully generated a schedule that meets **100%** of the demand.
            
            * **Status:** Safe to implement.
            * **Efficiency:** The schedule covers all shifts without any understaffing.
            * **Next Step:** You can export this data or proceed with rostering specific names.
            """)
        else:
            # Failure/Warning Case
            st.error(f"""
            ### ⚠️ CONCLUSION: SUB-OPTIMAL RESULT
            **Warning:** The current schedule still has **{int(total_shortage_val)} missing shift hours**.
            
            * **Status:** Risk of understaffing (see Red areas in Heatmap).
            * **Reason:** The algorithm needs more time or more particles to solve the complexity.
            * **Recommendation:**
                1.  Increase **'Iterations'** (try {iterations + 100}).
                2.  Increase **'Particle Count'** (try {n_particles + 10}).
                3.  Click "START OPTIMIZATION" again.
            """)

except Exception as e:
    st.error(f"An error occurred: {e}")