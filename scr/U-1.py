import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# 1. READ DATA
# ============================================

# File names
NORMALIZED_FILE = "normalised_export.csv"
INVERSION_FILE = "inversion radius .csv"

# Try different encodings
encodings_to_try = ['utf-8', 'ISO-8859-1', 'windows-1252', 'cp1256', 'latin1']

df_norm = None
df_inv = None

# Read first file (normalised_export.csv)
for encoding in encodings_to_try:
    try:
        df_norm = pd.read_csv(NORMALIZED_FILE, encoding=encoding)
        print(f"Successfully read {NORMALIZED_FILE} with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        continue

# Read second file (inversion radius .csv)
for encoding in encodings_to_try:
    try:
        df_inv = pd.read_csv(INVERSION_FILE, encoding=encoding)
        print(f"Successfully read {INVERSION_FILE} with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        continue

# Check if files were read successfully
if df_norm is None:
    raise ValueError(f"Failed to read file: {NORMALIZED_FILE} with all tried encodings")
if df_inv is None:
    raise ValueError(f"Failed to read file: {INVERSION_FILE} with all tried encodings")

# ============================================
# 2. DISPLAY ORIGINAL COLUMNS
# ============================================

print("\n" + "="*60)
print("Original columns in files (before processing):")
print("="*60)

print(f"\nColumns in normalised_export.csv:")
for i, col in enumerate(df_norm.columns):
    print(f"  {i+1:2d}. '{col}' (type: {df_norm[col].dtype})")

print(f"\nColumns in inversion radius .csv:")
for i, col in enumerate(df_inv.columns):
    print(f"  {i+1:2d}. '{col}' (type: {df_inv[col].dtype})")

# ============================================
# 3. CLEAN AND PREPARE DATA
# ============================================

# Clean column names (remove extra spaces)
df_norm.columns = df_norm.columns.str.strip()
df_inv.columns = df_inv.columns.str.strip()

print("\n" + "="*60)
print("Columns after cleaning:")
print("="*60)

print(f"\nColumns in df_norm after cleaning: {df_norm.columns.tolist()}")
print(f"Columns in df_inv after cleaning: {df_inv.columns.tolist()}")

# ============================================
# 4. FIND COMMON COLUMNS
# ============================================

# Find common columns
norm_cols_set = set(df_norm.columns)
inv_cols_set = set(df_inv.columns)
common_cols = norm_cols_set.intersection(inv_cols_set)

print(f"\nCommon columns between files: {common_cols}")

# Find columns containing "shot" in both files
shot_cols_norm = [col for col in df_norm.columns if 'shot' in col.lower()]
shot_cols_inv = [col for col in df_inv.columns if 'shot' in col.lower()]

print(f"\nColumns containing 'shot' in df_norm: {shot_cols_norm}")
print(f"Columns containing 'shot' in df_inv: {shot_cols_inv}")

# Find columns containing "time" in both files
time_cols_norm = [col for col in df_norm.columns if 'time' in col.lower()]
time_cols_inv = [col for col in df_inv.columns if 'time' in col.lower()]

print(f"\nColumns containing 'time' in df_norm: {time_cols_norm}")
print(f"Columns containing 'time' in df_inv: {time_cols_inv}")

# ============================================
# 5. PREPARE FOR MERGING
# ============================================

# If there is a common column containing "shot" in both files, rename it to 'shotno'
if shot_cols_norm and shot_cols_inv:
    # Use first column containing "shot" in each file
    shot_col_norm = shot_cols_norm[0]
    shot_col_inv = shot_cols_inv[0]
    
    print(f"\nRenaming '{shot_col_inv}' to '{shot_col_norm}' for consistency")
    df_inv = df_inv.rename(columns={shot_col_inv: shot_col_norm})
    
    # Update list after renaming
    shot_cols_inv = [shot_col_norm]
    
    # If column name is different from 'shotno', rename it to 'shotno'
    if shot_col_norm != 'shotno':
        print(f"Renaming '{shot_col_norm}' to 'shotno'")
        df_norm = df_norm.rename(columns={shot_col_norm: 'shotno'})
        df_inv = df_inv.rename(columns={shot_col_norm: 'shotno'})
else:
    # If no "shot" column found, use first common column
    if common_cols:
        common_col = list(common_cols)[0]
        print(f"\nNo 'shot' columns found, using first common column: '{common_col}'")
        # Rename this column to 'shotno' in both files
        df_norm = df_norm.rename(columns={common_col: 'shotno'})
        df_inv = df_inv.rename(columns={common_col: 'shotno'})
    else:
        # If no common column, use first column in each file
        print(f"\nNo common columns found, using first columns")
        first_col_norm = df_norm.columns[0]
        first_col_inv = df_inv.columns[0]
        df_norm = df_norm.rename(columns={first_col_norm: 'shotno'})
        df_inv = df_inv.rename(columns={first_col_inv: 'shotno'})

# Same for time columns
if time_cols_norm and time_cols_inv:
    time_col_norm = time_cols_norm[0]
    time_col_inv = time_cols_inv[0]
    
    if time_col_norm != time_col_inv:
        print(f"Renaming '{time_col_inv}' to '{time_col_norm}' for consistency")
        df_inv = df_inv.rename(columns={time_col_inv: time_col_norm})
    
    # If column name is different from 'time', rename it
    if time_col_norm != 'time':
        print(f"Renaming '{time_col_norm}' to 'time'")
        df_norm = df_norm.rename(columns={time_col_norm: 'time'})
        df_inv = df_inv.rename(columns={time_col_norm: 'time'})

print(f"\nColumns in df_norm after renaming: {df_norm.columns.tolist()}")
print(f"Columns in df_inv after renaming: {df_inv.columns.tolist()}")

# Now we should have 'shotno' and 'time' columns in both files
# Check if they exist
if 'shotno' not in df_norm.columns:
    raise KeyError("Column 'shotno' not found in df_norm after renaming")
if 'shotno' not in df_inv.columns:
    raise KeyError("Column 'shotno' not found in df_inv after renaming")

# Convert columns to appropriate types
df_norm['shotno'] = pd.to_numeric(df_norm['shotno'], errors='coerce')
df_inv['shotno'] = pd.to_numeric(df_inv['shotno'], errors='coerce')

if 'time' in df_norm.columns and 'time' in df_inv.columns:
    df_norm['time'] = pd.to_numeric(df_norm['time'], errors='coerce')
    df_inv['time'] = pd.to_numeric(df_inv['time'], errors='coerce')
else:
    print("Warning: Column 'time' not found in one or both files")

# ============================================
# 6. MERGE DATA
# ============================================

print("\n" + "="*60)
print("Merging data:")
print("="*60)

# Determine merge columns
merge_on = ['shotno']
if 'time' in df_norm.columns and 'time' in df_inv.columns:
    merge_on.append('time')
    print("Merging using: shotno and time")
else:
    print("Merging using: shotno only")

# Merge
df = pd.merge(df_norm, df_inv, on=merge_on, how='inner')

print(f"Number of rows after merging: {len(df)}")
print(f"Number of columns after merging: {len(df.columns)}")

if len(df) == 0:
    print("Warning: No matching rows found after merging!")
    print("Sample of shotno in df_norm:", df_norm['shotno'].head().tolist())
    print("Sample of shotno in df_inv:", df_inv['shotno'].head().tolist())
    
    # Try merging using shotno only (even if we have time)
    if len(merge_on) > 1:
        print("Trying to merge using shotno only...")
        df = pd.merge(df_norm, df_inv, on=['shotno'], how='inner')
        print(f"Number of rows after merging using shotno only: {len(df)}")

if len(df) == 0:
    raise ValueError("Data merging failed: No matching rows found")

# ============================================
# 7. SELECT AND EXTRACT VARIABLES
# ============================================

print("\n" + "="*60)
print("Selecting columns for analysis:")
print("="*60)

# Find required columns
# Bt (magnetic field) - could be 'B_T', 'Bt', 'B'
bt_candidates = [col for col in df.columns if any(keyword in col.upper() for keyword in ['B_T', 'BT', 'B-T', 'MAGNETIC'])]
# Ip (plasma current) - could be 'I42', 'Ip', 'I_P'
ip_candidates = [col for col in df.columns if any(keyword in col.upper() for keyword in ['I42', 'IP', 'I_P', 'I-P', 'CURRENT'])]
# R_inv (inversion radius) - could be 'R_inv', 'RINV', 'INV'
r_inv_candidates = [col for col in df.columns if any(keyword in col.upper() for keyword in ['R_INV', 'RINV', 'INV', 'RADIUS'])]

print(f"Candidates for Bt column: {bt_candidates}")
print(f"Candidates for Ip column: {ip_candidates}")
print(f"Candidates for R_inv column: {r_inv_candidates}")

# Select columns
if not bt_candidates:
    # If no candidates found, search manually
    print("\nManually searching for Bt column...")
    for col in df.columns:
        print(f"  {col}")
    bt_col = input("Enter Bt column name (magnetic field): ")
else:
    bt_col = bt_candidates[0]

if not ip_candidates:
    print("\nManually searching for Ip column...")
    for col in df.columns:
        print(f"  {col}")
    ip_col = input("Enter Ip column name (plasma current): ")
else:
    ip_col = ip_candidates[0]

if not r_inv_candidates:
    print("\nManually searching for R_inv column...")
    for col in df.columns:
        print(f"  {col}")
    r_inv_col = input("Enter R_inv column name (inversion radius): ")
else:
    r_inv_col = r_inv_candidates[0]

print(f"\nSelected columns:")
print(f"  Bt: {bt_col}")
print(f"  Ip: {ip_col}")
print(f"  R_inv: {r_inv_col}")

# Extract data
Bt = pd.to_numeric(df[bt_col], errors='coerce').values
Ip = pd.to_numeric(df[ip_col], errors='coerce').values
Y = pd.to_numeric(df[r_inv_col], errors='coerce').values

print(f"\nNumber of values in {bt_col}: {len(Bt)}")
print(f"Number of values in {ip_col}: {len(Ip)}")
print(f"Number of values in {r_inv_col}: {len(Y)}")

# Remove NaN values
mask = ~np.isnan(Bt) & ~np.isnan(Ip) & ~np.isnan(Y)
Bt = Bt[mask]
Ip = Ip[mask]
Y = Y[mask]

print(f"Number of data points after removing NaN: {len(Bt)}")

if len(Bt) == 0:
    raise ValueError("No valid data for analysis")

# Calculate X = Bt / Ip
X = Bt / Ip

print(f"\nData statistics:")
print(f"  Bt:   Mean = {np.mean(Bt):.4f}, Std = {np.std(Bt):.4f}")
print(f"  Ip:   Mean = {np.mean(Ip):.4f}, Std = {np.std(Ip):.4f}")
print(f"  R_inv: Mean = {np.mean(Y):.4f}, Std = {np.std(Y):.4f}")
print(f"  X = Bt/Ip: Mean = {np.mean(X):.4f}, Std = {np.std(X):.4f}")

# ============================================
# 8. BUILD INTERVALS
# ============================================

# Error margins (5%)
delta_x = 0.05
delta_y = 0.05

X_l = X * (1 - delta_x)
X_u = X * (1 + delta_x)

Y_l = Y * (1 - delta_y)
Y_u = Y * (1 + delta_y)

# ============================================
# 9. POINT LINEAR REGRESSION
# ============================================

A = np.vstack([X, np.ones_like(X)]).T
a_hat, b_hat = np.linalg.lstsq(A, Y, rcond=None)[0]

Y_pred = a_hat * X + b_hat

# Calculate R² coefficient
SS_res = np.sum((Y - Y_pred) ** 2)
SS_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (SS_res / SS_tot) if SS_tot != 0 else 0

# ============================================
# 10. INTERVAL REGRESSION (ENCLOSURE)
# ============================================

Y_pred_l = a_hat * X_l + b_hat
Y_pred_u = a_hat * X_u + b_hat

# Expand the band to ensure containing Y intervals
low_shift = np.min(Y_l - Y_pred_l)
up_shift = np.max(Y_u - Y_pred_u)

Y_reg_l = Y_pred_l + low_shift
Y_reg_u = Y_pred_u + up_shift

# ============================================
# 11. PLOT RESULTS
# ============================================

plt.figure(figsize=(12, 8))

# Interval data
plt.fill_between(X, Y_l, Y_u, color="gray", alpha=0.4, label="Y interval (±5%)")

# Original data points
plt.scatter(X, Y, s=30, alpha=0.6, color='blue', label="Original data")

# Point regression
plt.plot(X, Y_pred, "k--", linewidth=2, label=f"Linear regression (R² = {r_squared:.4f})")

# Interval regression
plt.plot(X, Y_reg_l, "r-", linewidth=1.5, label="Interval regression lower bound")
plt.plot(X, Y_reg_u, "r-", linewidth=1.5, label="Interval regression upper bound")

# Color area between interval regression bounds
plt.fill_between(X, Y_reg_l, Y_reg_u, color="red", alpha=0.2, label="Regression interval")

plt.xlabel(f"X = {bt_col} / {ip_col}", fontsize=14)
plt.ylabel(r_inv_col, fontsize=14)
plt.title("Interval Linear Regression - Enclosure Condition", fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(alpha=0.3)

# Add text box with parameters
textstr = '\n'.join((
    f'Equation: y = {a_hat:.4f}x + {b_hat:.4f}',
    f'R² = {r_squared:.4f}',
    f'Lower shift = {low_shift:.4f}',
    f'Upper shift = {up_shift:.4f}',
    f'Number of points = {len(X)}'
))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("interval_regression.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 12. PRINT RESULTS
# ============================================

print("\n" + "="*60)
print("INTERVAL REGRESSION RESULTS")
print("="*60)
print(f"Slope (a)              = {a_hat:.6f}")
print(f"Intercept (b)          = {b_hat:.6f}")
print(f"R² coefficient         = {r_squared:.6f}")
print(f"Lower shift            = {low_shift:.6f}")
print(f"Upper shift            = {up_shift:.6f}")
print(f"Number of data points  = {len(X)}")
print(f"Points within interval = {100 * np.mean((Y >= Y_reg_l) & (Y <= Y_reg_u)):.2f}%")
print("✔ Interval regression model built successfully")
print("✔ Plot saved as interval_regression.png")

# ============================================
# 13. SAVE RESULTS TO FILE
# ============================================

# Save analysis results
results_df = pd.DataFrame({
    'X': X,
    'X_lower': X_l,
    'X_upper': X_u,
    'Y': Y,
    'Y_lower': Y_l,
    'Y_upper': Y_u,
    'Y_pred': Y_pred,
    'Y_reg_lower': Y_reg_l,
    'Y_reg_upper': Y_reg_u
})

results_df.to_csv("regression_results.csv", index=False)
print("✔ Analysis results saved to regression_results.csv")

# ============================================
# 14. ADDITIONAL ANALYSIS
# ============================================

print("\n" + "="*60)
print("ADDITIONAL ANALYSIS")
print("="*60)

# Calculate approximate confidence intervals for parameters
n = len(X)
X_mean = np.mean(X)
S_xx = np.sum((X - X_mean) ** 2)
residuals = Y - Y_pred
sigma2 = np.sum(residuals ** 2) / (n - 2)
se_a = np.sqrt(sigma2 / S_xx)
se_b = np.sqrt(sigma2 * (1/n + X_mean**2 / S_xx))

print(f"Standard error of slope (SE_a)   = {se_a:.6f}")
print(f"Standard error of intercept (SE_b) = {se_b:.6f}")

# 95% confidence intervals (approximate)
t_val = 1.96  # For 95% confidence with large sample
print(f"\nApproximate 95% confidence intervals:")
print(f"Slope:   [{a_hat - t_val*se_a:.6f}, {a_hat + t_val*se_a:.6f}]")
print(f"Intercept:  [{b_hat - t_val*se_b:.6f}, {b_hat + t_val*se_b:.6f}]")

print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*60)