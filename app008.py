import streamlit as st
import pandas as pd
import pulp as lp
import numpy as np
import base64
import os

# ==============================================================================
# 1. CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(page_title="🏈 Pro NFL DFS Optimizer", layout="wide")

# DK Roster Rules
ROSTER_SIZE = 9
SALARY_CAP = 50000
POSITIONS = {
    'QB': (1, 1),
    'RB': (2, 3),
    'WR': (3, 4),
    'TE': (1, 2),
    'DST': (1, 1)
}

# Session state initialization
if 'exposure_counts' not in st.session_state:
    st.session_state.exposure_counts = {}
if 'player_data' not in st.session_state:
    st.session_state.player_data = None
if 'exclusion_rules' not in st.session_state:
    st.session_state.exclusion_rules = []
if 'custom_groups' not in st.session_state:
    st.session_state.custom_groups = []
if 'group_counts' not in st.session_state:
    st.session_state.group_counts = {}


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def load_and_clean_data(file_or_path):
    try:
        preview = pd.read_csv(file_or_path, nrows=20, header=None, encoding='utf-8-sig')
        header_row_idx = None
        
        for i, row in preview.iterrows():
            row_values = [str(x).strip() for x in row.values if pd.notna(x)]
            if "Player" in row_values or "Name + ID" in row_values or "DK Pos" in row_values or "Position" in row_values:
                header_row_idx = i
                break
        
        if header_row_idx is None:
            st.error("Could not detect a valid header row. Ensure your CSV has headers like 'Player' or 'Name + ID'.")
            return None

        if hasattr(file_or_path, 'seek'):
            file_or_path.seek(0)
            
        df = pd.read_csv(file_or_path, header=header_row_idx, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()

        field_candidates = {
            'Name': ['Name', 'Player', 'Player Name', 'player', 'Name + ID'],
            'Position': ['Position', 'DK Pos', 'Pos', 'Roster Position'],
            'Team': ['Team', 'TeamAbbrev', 'Team Abbrev', 'team'],
            'Opp': ['Opp', 'Opponent', 'Game Info', 'Matchup'],
            'Salary': ['Salary', 'DK Salary', 'salary'],
            'Projection': ['Projection', 'DK Proj', 'Proj', 'Fpts', 'Median', 'fpts', 'AvgPointsPerGame'],
            'Floor': ['Floor', 'DK Floor', 'floor'],
            'Ceiling': ['Ceiling', 'DK Ceiling', 'ceiling'],
            'Value': ['Value', 'DK Value', 'value'],
            'Small_Field_Own': ['Small_Field_Own', 'Small Field', 'SmallField', 'SE Own', 'Cash Own'],
            'Large_Field_Own': ['Large_Field_Own', 'Large Field', 'LargeField', 'GPP Own', 'Ownership', '% Drafted'],
            'ID': ['ID', 'id', 'Id', 'Player ID', 'DK ID']
        }

        rename_dict = {}
        missing_fields = []

        for internal_col, candidates in field_candidates.items():
            matched = False
            for candidate in candidates:
                if candidate in df.columns:
                    rename_dict[candidate] = internal_col
                    matched = True
                    break
            if not matched:
                if internal_col in ['Small_Field_Own', 'Large_Field_Own', 'Floor', 'Ceiling', 'Value']:
                    df[internal_col] = 0.0
                elif internal_col == 'ID':
                    df['ID'] = df.index.astype(str)
                elif internal_col == 'Opp':
                    df['Opp'] = ''
                else:
                    missing_fields.append(internal_col)

        if missing_fields:
            st.error(f"Could not find matching columns for: {', '.join(missing_fields)}")
            return None

        df = df.rename(columns=rename_dict)

        if 'ID' not in rename_dict.values() and 'Name' in df.columns and df['Name'].astype(str).str.contains(r'\(\d+\)').any():
            df['ID'] = df['Name'].str.extract(r'\(\d+\)')
            df['Name'] = df['Name'].str.replace(r'\s*\(\d+\)', '', regex=True)

        df['ID'] = df['ID'].astype(str)
        df['Salary'] = df['Salary'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(0)
        df['Projection'] = pd.to_numeric(df['Projection'], errors='coerce').fillna(0)
        
        for numeric_col in ['Floor', 'Ceiling', 'Value']:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce').fillna(0)
            
        if (df['Floor'] == 0).all(): df['Floor'] = df['Projection']
        if (df['Ceiling'] == 0).all(): df['Ceiling'] = df['Projection']
        if (df['Value'] == 0).all(): 
            df['Value'] = np.where(df['Salary'] > 0, df['Projection'] / (df['Salary'] / 1000), 0)
        
        for own_col in ['Small_Field_Own', 'Large_Field_Own']:
            df[own_col] = df[own_col].astype(str).str.replace('%', '', regex=False).str.strip()
            df[own_col] = pd.to_numeric(df[own_col], errors='coerce').fillna(0)

        df['Position'] = df['Position'].astype(str).str.strip().str.upper()
        df['Position'] = df['Position'].replace({'D/ST': 'DST', 'DEF': 'DST', 'D': 'DST'})
        df['Position'] = df['Position'].apply(lambda x: x.split('/')[0] if '/' in x else x)

        df['Opp'] = df['Opp'].astype(str).str.replace('@', '', regex=False).str.strip()

        df['Lock'] = False
        df['Exclude'] = False
        df['Max Exposure (%)'] = 100 
        df['Min Exposure (%)'] = 0

        valid_pos = ['QB', 'RB', 'WR', 'TE', 'DST']
        df = df[df['Position'].isin(valid_pos)].reset_index(drop=True)
        
        st.session_state.exposure_counts = {pid: 0 for pid in df['ID'].tolist()}
        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="DK_upload_lineups.csv">Download DK Upload CSV</a>'

def get_sorted_lineup_df(df, lineup_ids):
    lineup_df = df.loc[lineup_ids].copy() 
    pos_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'DST': 7} 
    lineup_df['Pos_Rank'] = lineup_df['Position'].map(pos_map)
    lineup_df['Secondary_Sort'] = 0 
    
    for pos, limit in [('RB', 2), ('WR', 3), ('TE', 1)]:
        pos_df = lineup_df[lineup_df['Position'] == pos].sort_values(by='Projection', ascending=False)
        if not pos_df.empty:
            lineup_df.loc[pos_df.index[:limit], 'Secondary_Sort'] = 1

    flex_mask = (lineup_df['Position'].isin(['RB', 'WR', 'TE'])) & (lineup_df['Secondary_Sort'] == 0)
    if flex_mask.any():
        lineup_df.loc[flex_mask, 'Position'] = 'FLEX'
        lineup_df.loc[flex_mask, 'Pos_Rank'] = 6 
    
    return lineup_df.sort_values(by=['Pos_Rank', 'Secondary_Sort', 'Projection'], ascending=[True, False, False]).reset_index(drop=True)

def format_dk_csv_lineup(lineup_df):
    dk_template_cols = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
    try:
        pos_map = {pos: [] for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'FLEX']}
        for _, row in lineup_df.iterrows():
            pos_map[row['Position']].append(row['ID'])
            
        row = []
        row.extend(pos_map['QB'][:1])
        row.extend(pos_map['RB'][:2])
        row.extend(pos_map['WR'][:3])
        row.extend(pos_map['TE'][:1])
        row.extend(pos_map['FLEX'][:1])
        row.extend(pos_map['DST'][:1])
        
        return pd.DataFrame([row], columns=dk_template_cols)
    except Exception:
        return pd.DataFrame([], columns=dk_template_cols)

# ==============================================================================
# 3. OPTIMIZATION ENGINE
# ==============================================================================

def run_optimization(player_df, num_lineups, min_salary, stack_qb, min_pc, max_pc, randomness_pct, global_max_pct, min_low_own_players, low_own_threshold, ownership_column_name, min_unique, exposure_settings, avoid_dst_vs_offense, exclusion_rules, flex_settings, opt_metric, custom_groups):
    
    lineups = []
    df = player_df.copy()
    df = df[df['Exclude'] == False].reset_index(drop=True)
    locked_players = df[df['Lock'] == True].index.tolist()
    player_ids = df.index.tolist() 
    
    status_text = st.empty()
    bar = st.progress(0)
    
    team_stack_map = {}
    if stack_qb:
        for team in df['Team'].unique():
            qb_idxs = df[(df['Team'] == team) & (df['Position'] == 'QB')].index.tolist()
            rec_idxs = df[(df['Team'] == team) & (df['Position'].isin(['WR', 'TE']))].index.tolist()
            if qb_idxs and rec_idxs:
                team_stack_map[team] = (qb_idxs, rec_idxs)
                
    dst_opp_map = {}
    if avoid_dst_vs_offense:
        for team in df['Team'].unique():
            team_rows = df[df['Team'] == team]
            if not team_rows.empty:
                opp_team = team_rows.iloc[0]['Opp']
                opp_off_idxs = df[(df['Team'] == opp_team) & (df['Position'].isin(['QB', 'RB', 'WR', 'TE']))].index.tolist()
                dst_idx_list = df[(df['Team'] == team) & (df['Position'] == 'DST')].index.tolist()
                if dst_idx_list:
                    dst_opp_map[dst_idx_list[0]] = opp_off_idxs

    active_rules = []
    for trigger_id, exclude_ids in exclusion_rules:
        trigger_matches = df[df['ID'] == str(trigger_id)].index
        if not trigger_matches.empty:
            trigger_idx = trigger_matches[0]
            exclude_indices = [df[df['ID'] == str(exc_id)].index[0] for exc_id in exclude_ids if not df[df['ID'] == str(exc_id)].index.empty]
            if exclude_indices:
                active_rules.append((trigger_idx, exclude_indices))

    active_groups = []
    for g_idx, grp in enumerate(custom_groups):
        grp_indices = [df[df['ID'] == str(pid)].index[0] for pid in grp['player_ids'] if not df[df['ID'] == str(pid)].index.empty]
        if len(grp_indices) == len(grp['player_ids']):
            active_groups.append({
                'id': g_idx,
                'name': grp['name'],
                'indices': grp_indices,
                'min_count': int(np.ceil((grp['min_pct'] / 100.0) * num_lineups)),
                'max_count': int(np.floor((grp['max_pct'] / 100.0) * num_lineups)),
                'size': len(grp_indices)
            })

    flex_counts = {'RB': 0, 'WR': 0, 'TE': 0}
    st.session_state.group_counts = {g['name']: 0 for g in active_groups}

    for k in range(num_lineups):
        status_text.text(f"Generating Lineup {k+1} of {num_lineups}...")
        bar.progress((k + 1) / num_lineups)
        
        if randomness_pct > 0:
            current_noise = np.random.uniform(1 - (randomness_pct/100), 1 + (randomness_pct/100), len(df))
            df['Optimization_Proj'] = df[opt_metric] * current_noise
        else:
            df['Optimization_Proj'] = df[opt_metric]

        for i in player_ids:
            pid = df.loc[i, 'ID']
            min_exp_pct = df.loc[i, 'Min Exposure (%)']
            if min_exp_pct > 0:
                target_count = int(np.ceil((min_exp_pct / 100.0) * num_lineups))
                if st.session_state.exposure_counts.get(pid, 0) < target_count:
                    df.loc[i, 'Optimization_Proj'] += 1000

        prob = lp.LpProblem(f"NFL_Lineup_{k}", sense=-1) 
        player_vars = lp.LpVariable.dicts("Player", player_ids, 0, 1, "Binary")
        
        prob += lp.lpSum([df.loc[i, 'Optimization_Proj'] * player_vars[i] for i in player_ids])
        prob += lp.lpSum([df.loc[i, 'Salary'] * player_vars[i] for i in player_ids]) <= SALARY_CAP
        prob += lp.lpSum([df.loc[i, 'Salary'] * player_vars[i] for i in player_ids]) >= min_salary
        prob += lp.lpSum([player_vars[i] for i in player_ids]) == ROSTER_SIZE
        
        for pos, (min_p, max_p) in POSITIONS.items():
            current_max_p = max_p
            if pos in ['RB', 'WR', 'TE']:
                allowed_count = int(np.ceil((flex_settings[pos] / 100.0) * num_lineups))
                if flex_counts[pos] >= allowed_count:
                    current_max_p = min_p 
            pos_vars = [player_vars[i] for i in player_ids if df.loc[i, 'Position'] == pos]
            prob += lp.lpSum(pos_vars) >= min_p
            prob += lp.lpSum(pos_vars) <= current_max_p
        
        prob += lp.lpSum([player_vars[i] for i in player_ids if df.loc[i, 'Position'] in ['RB', 'WR', 'TE']]) == 7
        
        for i in locked_players:
            prob += player_vars[i] == 1
            
        if stack_qb:
            for team, (qb_idxs, rec_idxs) in team_stack_map.items():
                for qb_i in qb_idxs:
                    prob += lp.lpSum([player_vars[r] for r in rec_idxs]) - min_pc * player_vars[qb_i] >= 0
                    prob += lp.lpSum([player_vars[r] for r in rec_idxs]) - max_pc * player_vars[qb_i] <= ROSTER_SIZE

        if min_low_own_players > 0:
            low_own_idxs = df[df[ownership_column_name] < low_own_threshold].index
            prob += lp.lpSum([player_vars[i] for i in low_own_idxs]) >= min_low_own_players

        if len(lineups) > 0 and min_unique > 0:
            for prev_lineup in lineups:
                prob += lp.lpSum([player_vars[pid] for pid in prev_lineup['ID_List']]) <= (ROSTER_SIZE - min_unique)
        
        if avoid_dst_vs_offense:
            for dst_idx, opp_off_idxs in dst_opp_map.items():
                prob += lp.lpSum([player_vars[o] for o in opp_off_idxs]) + 9 * player_vars[dst_idx] <= 9

        for trig_idx, excl_indices in active_rules:
            for exc_idx in excl_indices:
                prob += player_vars[trig_idx] + player_vars[exc_idx] <= 1

        group_vars = {}
        for grp in active_groups:
            g_name = grp['name']
            curr_g_count = st.session_state.group_counts.get(g_name, 0)
            remaining_lineups = num_lineups - k
            
            z_grp = lp.LpVariable(f"Group_{grp['id']}_{k}", 0, 1, "Binary")
            group_vars[g_name] = z_grp
            
            prob += lp.lpSum([player_vars[idx] for idx in grp['indices']]) - (grp['size'] * z_grp) >= 0
            
            if curr_g_count >= grp['max_count']:
                prob += z_grp == 0
                prob += lp.lpSum([player_vars[idx] for idx in grp['indices']]) <= (grp['size'] - 1)
                
            needed = grp['min_count'] - curr_g_count
            if needed > 0 and needed >= remaining_lineups:
                prob += z_grp == 1

        for i in player_ids:
            pid = df.loc[i, 'ID']
            effective_max_pct = min(df.loc[i, 'Max Exposure (%)'], global_max_pct, exposure_settings.get(df.loc[i, 'Position'], 100)) / 100.0
            if st.session_state.exposure_counts.get(pid, 0) >= int(np.ceil(effective_max_pct * num_lineups)):
                prob += player_vars[i] == 0

        prob.solve(lp.PULP_CBC_CMD(msg=0)) 
        
        if prob.status == 1: 
            current_ids = [i for i in player_ids if player_vars[i].varValue == 1.0]
            
            n_rb = len([i for i in current_ids if df.loc[i, 'Position'] == 'RB'])
            n_wr = len([i for i in current_ids if df.loc[i, 'Position'] == 'WR'])
            n_te = len([i for i in current_ids if df.loc[i, 'Position'] == 'TE'])
            
            if n_rb > 2: flex_counts['RB'] += 1
            elif n_wr > 3: flex_counts['WR'] += 1
            elif n_te > 1: flex_counts['TE'] += 1
            
            for i in current_ids:
                pid = df.loc[i, 'ID']
                st.session_state.exposure_counts[pid] = st.session_state.exposure_counts.get(pid, 0) + 1
            
            for grp in active_groups:
                if all(idx in current_ids for idx in grp['indices']):
                    st.session_state.group_counts[grp['name']] = st.session_state.group_counts.get(grp['name'], 0) + 1
            
            sorted_df = get_sorted_lineup_df(df, current_ids)
            
            lineups.append({
                'Lineup Num': k+1,
                'Total Points': round(df.loc[current_ids, opt_metric].sum(), 2),
                'Salary': df.loc[current_ids, 'Salary'].sum(),
                'ID_List': current_ids, 
                'Lineup_DF': sorted_df,
                'DK_Upload_Row': format_dk_csv_lineup(sorted_df) 
            })
        else:
            st.warning(f"Solver failed for Lineup {k+1}. Constraints may be too strict.")
            break
            
    bar.empty()
    status_text.text("Optimization Complete!")
    return pd.DataFrame(lineups)

# ==============================================================================
# 4. MAIN APP UI
# ==============================================================================

st.title("🏈 Pro NFL DFS Optimizer")
st.markdown("Build tournament-winning lineups with stacking, diversity, and **Exposure Control**.")

with st.sidebar:
    st.header("⚙️ Generator Settings")
    num_lineups = st.number_input("Number of Lineups", 1, 150, 5)
    opt_metric = st.selectbox("Optimize Based On", ["Projection", "Ceiling", "Floor", "Value"], help="Select the stat the optimizer will maximize.")
    
    with st.expander("📊 Exposure Controls", expanded=False):
        st.markdown("**Global Max Exposure Limits**")
        global_max_pct = st.slider("Global Cap (All Players)", 0, 100, 100)
        st.markdown("---")
        st.markdown("**Position Caps (Overrides Global if Lower)**")
        exp_qb = st.number_input("Max QB %", 0, 100, 100)
        exp_rb = st.number_input("Max RB %", 0, 100, 100)
        exp_wr = st.number_input("Max WR %", 0, 100, 100)
        exp_te = st.number_input("Max TE %", 0, 100, 100)
        exp_dst = st.number_input("Max DST %", 0, 100, 100)
        exposure_settings = {'QB': exp_qb, 'RB': exp_rb, 'WR': exp_wr, 'TE': exp_te, 'DST': exp_dst}

    min_unique = st.slider("Min Unique Players per Lineup", 0, 9, 1)
    
    st.divider()
    st.header("🧠 Strategy")
    stack_qb = st.checkbox("Stack QB with WR/TE", value=True)
    if stack_qb:
        c1, c2 = st.columns(2)
        min_pc = c1.number_input("Min Stack", 1, 3, 1)
        max_pc = c2.number_input("Max Stack", 1, 4, 2)
        if max_pc < min_pc: max_pc = min_pc
    else:
        min_pc, max_pc = 0, 4
    
    avoid_dst_vs_offense = st.checkbox("Avoid DST vs Opposing Offense", value=True)
    
    with st.expander("Flex Position Allocation", expanded=False):
        st.write("Limit how often each position appears in the FLEX spot.")
        flex_rb_pct = st.slider("Max Flex RB %", 0, 100, 100)
        flex_wr_pct = st.slider("Max Flex WR %", 0, 100, 100)
        flex_te_pct = st.slider("Max Flex TE %", 0, 100, 100)
        flex_settings = {'RB': flex_rb_pct, 'WR': flex_wr_pct, 'TE': flex_te_pct}

    min_salary = st.number_input("Min Lineup Salary", 45000, 50000, 49000, step=100)
    st.divider()
    
    st.header("📢 Ownership")
    own_col = 'Small_Field_Own' if st.radio("Ownership Source", ['Small Field', 'Large Field']) == 'Small Field' else 'Large_Field_Own'
    c1, c2 = st.columns(2)
    min_low_own_players = c1.number_input("Min Low-Owned Players", 0, 9, 0)
    low_own_threshold = c2.number_input("Low Own Threshold (%)", 0.0, 100.0, 10.0)
    
    st.divider()
    st.header("🎲 Randomness")
    randomness = st.slider("Projection Randomness (%)", 0, 50, 0)

# --- FILE UPLOAD & DEFAULT LOAD LOGIC ---
uploaded_file = st.file_uploader("Upload Player CSV (Optional)", type=['csv'])
DEFAULT_FILE_PATH = "default_projections.csv" 

data_source = uploaded_file if uploaded_file else (DEFAULT_FILE_PATH if os.path.exists(DEFAULT_FILE_PATH) else None)

if data_source:
    source_name = uploaded_file.name if uploaded_file else "default"
    
    if st.session_state.player_data is None or st.session_state.get('uploaded_file_name') != source_name:
        st.session_state.player_data = load_and_clean_data(data_source)
        if st.session_state.player_data is not None:
             st.session_state.uploaded_file_name = source_name

if st.session_state.player_data is None:
    st.info("Please upload a CSV file or ensure 'default_projections.csv' is in the project folder.")

else:
    st.sidebar.divider()
    st.sidebar.header("🧩 Custom Groups & Cores")
    with st.sidebar.expander("Add Custom Stack / Group", expanded=False):
        df_display = st.session_state.player_data.copy()
        df_display['Label'] = df_display['Name'] + " (" + df_display['Position'] + ") - $" + df_display['Salary'].astype(int).astype(str)
        player_options = dict(zip(df_display['ID'], df_display['Label']))
        
        with st.form("group_form"):
            grp_name = st.text_input("Group Name", value=f"Stack {chr(65 + len(st.session_state.custom_groups))}")
            grp_players = st.multiselect("Select Group Players", options=player_options.keys(), format_func=lambda x: player_options[x])
            col_gmin, col_gmax = st.columns(2)
            min_grp_exp = col_gmin.number_input("Min Exposure %", 0, 100, 0, step=5)
            max_grp_exp = col_gmax.number_input("Max Exposure %", 0, 100, 100, step=5)
            
            submitted_grp = st.form_submit_button("Add Group")
            if submitted_grp:
                if len(grp_players) < 2:
                    st.error("Please select at least 2 players.")
                elif min_grp_exp > max_grp_exp:
                    st.error("Min Exposure cannot exceed Max Exposure.")
                else:
                    st.session_state.custom_groups.append({
                        'name': grp_name,
                        'player_ids': grp_players,
                        'min_pct': min_grp_exp,
                        'max_pct': max_grp_exp
                    })
                    st.success(f"Added {grp_name}!")
                    st.rerun()

        if st.session_state.custom_groups:
            st.markdown("**Active Stacks & Cores:**")
            for idx, grp in enumerate(st.session_state.custom_groups):
                p_names = ", ".join([player_options.get(pid, pid).split('(')[0].strip() for pid in grp['player_ids']])
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption(f"**{grp['name']}** ({grp['min_pct']}% - {grp['max_pct']}%)")
                    st.text(f"{p_names}")
                with c2:
                    if st.button("❌", key=f"del_grp_{idx}"):
                        st.session_state.custom_groups.pop(idx)
                        st.rerun()
            
            if st.button("Clear All Groups"):
                st.session_state.custom_groups = []
                st.rerun()

    st.sidebar.divider()
    st.sidebar.header("🚫 Player Exclusions")
    with st.sidebar.expander("Add Exclusion Rules", expanded=False):
        with st.form("exclusion_form"):
            st.caption("If 'Trigger Player' is selected, exclude all 'Excluded Players'.")
            trigger = st.selectbox("Trigger Player", options=player_options.keys(), format_func=lambda x: player_options[x])
            excludes = st.multiselect("Excluded Players", options=player_options.keys(), format_func=lambda x: player_options[x])
            
            submitted = st.form_submit_button("Add Rule")
            if submitted and excludes:
                st.session_state.exclusion_rules.append((trigger, excludes))
                st.success("Rule Added!")

        if st.session_state.exclusion_rules:
            st.markdown("**Active Rules:**")
            for idx, (trig, excls) in enumerate(st.session_state.exclusion_rules):
                trig_name = player_options.get(trig, trig)
                excl_names = ", ".join([player_options.get(e, e).split('(')[0] for e in excls])
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"If {trig_name.split('(')[0]} -> No {excl_names}")
                with col2:
                    if st.button("❌", key=f"del_{idx}"):
                        st.session_state.exclusion_rules.pop(idx)
                        st.rerun()
            
            if st.button("Clear All Rules"):
                st.session_state.exclusion_rules = []
                st.rerun()

    st.subheader("1. Player Pool & Constraints")
    
    cfg = {
        "Lock": st.column_config.CheckboxColumn(width='small'),
        "Exclude": st.column_config.CheckboxColumn(width='small'),
        "Projection": st.column_config.NumberColumn(format="%.2f"),
        "Ceiling": st.column_config.NumberColumn(format="%.2f"),
        "Floor": st.column_config.NumberColumn(format="%.2f"),
        "Value": st.column_config.NumberColumn(format="%.2f"),
        "Salary": st.column_config.NumberColumn(format="$%d", disabled=True),
        "Max Exposure (%)": st.column_config.NumberColumn(format="%d %%", min_value=0, max_value=100),
        "Min Exposure (%)": st.column_config.NumberColumn(format="%d %%", min_value=0, max_value=100),
        "Small_Field_Own": st.column_config.NumberColumn(format="%.1f %%"),
        "Large_Field_Own": st.column_config.NumberColumn(format="%.1f %%"),
    }
    
    cols = ['Name', 'Position', 'Team', 'Salary', 'Projection', 'Ceiling', 'Floor', 'Value', own_col, 'Min Exposure (%)', 'Max Exposure (%)', 'Lock', 'Exclude']
    display_cols = [col for col in cols if col in st.session_state.player_data.columns]

    edited_df = st.data_editor(
        st.session_state.player_data[display_cols],
        column_config=cfg,
        disabled=["Name", "Position", "Team", "Salary", own_col],
        hide_index=True,
        use_container_width=True
    )
    st.session_state.player_data.update(edited_df)
    
    st.divider()
    
    if st.button("🚀 Run Optimizer", type="primary"):
        st.session_state.exposure_counts = {pid: 0 for pid in st.session_state.player_data['ID'].tolist()}
        st.session_state.group_counts = {grp['name']: 0 for grp in st.session_state.custom_groups}
        
        with st.spinner("Optimizing..."):
            results_df = run_optimization(
                st.session_state.player_data, num_lineups, min_salary, stack_qb, min_pc, max_pc,
                randomness, global_max_pct, min_low_own_players, low_own_threshold, own_col, min_unique,
                exposure_settings, avoid_dst_vs_offense, st.session_state.exclusion_rules, flex_settings, opt_metric,
                st.session_state.custom_groups
            )
        
        if not results_df.empty:
            st.success(f"Generated {len(results_df)} Lineups optimized for {opt_metric}!")
            
            all_dk_rows = pd.concat(results_df['DK_Upload_Row'].tolist(), ignore_index=True)
            st.markdown(download_csv(all_dk_rows), unsafe_allow_html=True)
            
            st.divider()
            
            st.header("📊 Output Summary Dashboard")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Lineups", len(results_df))
            col2.metric("Avg Salary Spent", f"${results_df['Salary'].mean():.0f}")
            col3.metric(f"Avg Lineup {opt_metric}", f"{results_df['Total Points'].mean():.2f}")
            
            if st.session_state.custom_groups:
                st.subheader("🧩 Custom Group & Stack Exposures")
                grp_summary_data = []
                for grp in st.session_state.custom_groups:
                    count = st.session_state.group_counts.get(grp['name'], 0)
                    pct = (count / len(results_df)) * 100
                    p_labels = ", ".join([player_options.get(pid, pid).split('(')[0].strip() for pid in grp['player_ids']])
                    grp_summary_data.append({
                        'Group Name': grp['name'],
                        'Players': p_labels,
                        'Target Min': f"{grp['min_pct']}%",
                        'Target Max': f"{grp['max_pct']}%",
                        'Lineups': count,
                        'Exposure (%)': pct
                    })
                grp_df = pd.DataFrame(grp_summary_data)
                st.dataframe(
                    grp_df,
                    column_config={
                        "Exposure (%)": st.column_config.ProgressColumn(
                            "Actual Exposure (%)", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True, use_container_width=True
                )

            st.subheader("Player Exposure Report")
            exp_data = []
            for pid, count in st.session_state.exposure_counts.items():
                if count > 0:
                    r = st.session_state.player_data[st.session_state.player_data['ID'] == pid]
                    if not r.empty:
                        r = r.iloc[0]
                        exp_data.append({
                            'Player': r['Name'], 'Pos': r['Position'], 'Team': r['Team'],
                            'Lineups': count, 'Exposure (%)': (count/len(results_df))*100
                        })
                    
            if exp_data:
                exp_df = pd.DataFrame(exp_data).sort_values('Lineups', ascending=False)
                st.dataframe(
                    exp_df,
                    column_config={
                        "Exposure (%)": st.column_config.ProgressColumn(
                            "Exposure (%)", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True, use_container_width=True
                )

            st.divider()

            with st.expander(f"📋 View All {len(results_df)} Individual Lineups", expanded=False):
                for _, row in results_df.iterrows():
                    cols = st.columns(3)
                    cols[0].metric(f"Lineup {row['Lineup Num']}", f"{row['Total Points']:.2f} {opt_metric}")
                    cols[1].metric("Salary", f"${row['Salary']}")
                    cols[2].metric("Rem. Salary", f"${SALARY_CAP - row['Salary']}")
                    
                    display_metric = opt_metric if opt_metric in row['Lineup_DF'].columns else 'Projection'
                    st.dataframe(
                        row['Lineup_DF'][['Position', 'Name', 'Team', 'Salary', display_metric]],
                        column_config={
                            "Position": st.column_config.Column("POS", width="small"),
                            "Name": st.column_config.Column("PLAYER", width="large"),
                            "Team": st.column_config.Column("TEAM", width="small"),
                            "Salary": st.column_config.NumberColumn("SALARY ($)", format="$%d"),
                            display_metric: st.column_config.NumberColumn(display_metric.upper(), format="%.2f"),
                        },
                        hide_index=True, use_container_width=True
                    )
                    st.markdown("---")
