"""
GRAVITAS Prediction 2: Compute ∇²S_EE(r) and compare to NFW.

The entanglement entropy profile S_EE(r) for a mass M embedded in 
de Sitter spacetime, following the Jacobson-Verlinde framework:

1. Jacobson (1995): δS ∝ R_ab k^a k^b at each local Rindler horizon
   => S_EE relates to the gravitational potential
   
2. Verlinde (2017): In de Sitter, volume-law entropy from dark energy 
   produces an entropy displacement when baryonic matter is introduced,
   generating an additional "dark gravitational force"
   g_D(r) = sqrt(a_0 * g_B(r) / 6) where a_0 = c*H_0

3. The effective potential:
   Φ_eff(r) = -GM/r + Φ_D(r) 
   where Φ_D(r) = -sqrt(a_0*GM/6) * ln(r/r_0)

4. The ERI claim: ρ_DM(r) ∝ ∇²S_EE(r)
   If S_EE ∝ Φ_eff, then ρ_DM ∝ ∇²Φ_eff

We compute this and compare to NFW.

Author: ERI Labs, Eric Ren, April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# PHYSICAL CONSTANTS (SI)
# ============================================================
G = 6.674e-11       # m^3 kg^-1 s^-2
c = 2.998e8          # m/s
H_0 = 2.2e-18        # s^-1 (70 km/s/Mpc)
a_0 = c * H_0        # Hubble acceleration ~ 6.6e-10 m/s^2
M_sun = 1.989e30     # kg
kpc = 3.086e19       # m

# ============================================================
# GALAXY PARAMETERS (Milky Way-like)
# ============================================================
M_b = 6e10 * M_sun   # Baryonic mass ~ 6 × 10^10 solar masses
r_s_nfw = 20 * kpc   # NFW scale radius ~ 20 kpc
rho_0_nfw = 0.01 * M_sun / kpc**3  # NFW normalization (typical)

# Schwarzschild radius of central SMBH (Sgr A*)
M_bh = 4e6 * M_sun
r_schw = 2 * G * M_bh / c**2  # ~ 1.2e10 m ~ 0.04 AU

# ============================================================
# RADIAL GRID (log-spaced, 0.1 kpc to 300 kpc)
# ============================================================
r = np.logspace(np.log10(0.1 * kpc), np.log10(300 * kpc), 2000)

# ============================================================
# PART 1: NFW PROFILE (the target)
# ============================================================
def rho_nfw(r, rho_0, r_s):
    """Navarro-Frenk-White profile: ρ_0 / [(r/r_s)(1 + r/r_s)²]"""
    x = r / r_s
    return rho_0 / (x * (1 + x)**2)

# Fit rho_0 to give reasonable MW halo mass
# M_NFW(r_200) ~ 10^12 M_sun at r_200 ~ 200 kpc
# Integrate NFW: M(r) = 4π ρ_0 r_s^3 [ln(1+r/r_s) - (r/r_s)/(1+r/r_s)]
def M_nfw(r, rho_0, r_s):
    x = r / r_s
    return 4 * np.pi * rho_0 * r_s**3 * (np.log(1 + x) - x/(1+x))

# Calibrate: M(200 kpc) ~ 10^12 M_sun
r_200 = 200 * kpc
M_target = 1e12 * M_sun
x_200 = r_200 / r_s_nfw
rho_0_nfw = M_target / (4 * np.pi * r_s_nfw**3 * (np.log(1 + x_200) - x_200/(1+x_200)))

rho_nfw_arr = rho_nfw(r, rho_0_nfw, r_s_nfw)

# ============================================================
# PART 2: VERLINDE DARK MATTER PROFILE
# ============================================================
# Verlinde (2017, eq. 7.40): g_D(r) = sqrt(a_0 * g_B / 6)
# where g_B = GM_b/r² (baryonic only, treating as point mass for simplicity)

g_B = G * M_b / r**2
g_D_verlinde = np.sqrt(a_0 * g_B / 6)

# Apparent dark matter density from Verlinde:
# g_D = GM_D(r)/r² => M_D(r) = g_D * r² / G
# ρ_D = (1/4πr²) dM_D/dr

# g_D = sqrt(a_0 * G * M_b / 6) / r
# M_D = sqrt(a_0 * G * M_b / 6) * r / G = sqrt(a_0 * M_b / (6*G)) * r

coeff_verlinde = np.sqrt(a_0 * M_b / (6 * G))
# dM_D/dr = coeff_verlinde (constant!)
# ρ_D = coeff_verlinde / (4π r²)

rho_verlinde = coeff_verlinde / (4 * np.pi * r**2)

# ============================================================
# PART 3: S_EE(r) AND ITS LAPLACIAN (THE ERI COMPUTATION)
# ============================================================
# 
# The entanglement entropy profile S_EE(r) has three components:
#
# Component 1: Schwarzschild / Newtonian potential
#   S_1(r) = -α GM/r   (Jacobson: entropy ∝ potential)
#
# Component 2: Verlinde entropy displacement
#   The dark gravitational potential:
#   Φ_D(r) = -sqrt(a_0 GM_b/6) ln(r/r_0)
#   S_2(r) = -β sqrt(a_0 GM_b/6) ln(r/r_0)
#
# Component 3: De Sitter volume entropy (background)
#   S_3(r) = γ r²  (proportional to area of sphere at r)
#   This gives the "cosmological" entropy that Verlinde's displacement acts on
#
# The proportionality constants α, β, γ are set by:
#   - α: Jacobson's δQ = TδS normalization => α = c³/(4Gℏ) per unit area
#   - β: same normalization as α for the displacement piece  
#   - γ: Bekenstein-Hawking, A/(4Gℏ)
#
# For the density comparison, we only need the SHAPE of ∇²S_EE,
# not the absolute normalization (which cancels in the ρ_DM ∝ ∇²S_EE claim).
#
# So we set α = β = 1 for shape comparison.

# Define S_EE(r) [arbitrary normalization — only shape matters]
r_0 = r_s_nfw  # reference radius (natural choice: scale radius)

# Component 1: Newtonian
S_newton = -G * M_b / r

# Component 2: Verlinde displacement (the logarithmic piece)
S_verlinde_disp = -np.sqrt(a_0 * G * M_b / 6) * np.log(r / r_0)

# Full S_EE (the pieces that produce ρ ≠ 0 away from origin)
S_EE = S_newton + S_verlinde_disp

# ============================================================
# COMPUTE LAPLACIAN NUMERICALLY
# ============================================================
# ∇²f(r) = (1/r²) d/dr [r² df/dr]
# = f'' + (2/r)f'

# First derivative
dS_dr = np.gradient(S_EE, r)

# Second derivative  
d2S_dr2 = np.gradient(dS_dr, r)

# Laplacian in spherical coordinates (radial part only for spherical symmetry)
laplacian_S = d2S_dr2 + 2 * dS_dr / r

# ============================================================
# ANALYTICAL CHECK
# ============================================================
# ∇²(-GM/r) = 0 for r > 0 (harmonic, Laplacian vanishes)
# ∇²(-A ln(r)) = -A/r²   (since d/dr(ln r) = 1/r, d²/dr²(ln r) = -1/r²,
#                           and (2/r)(1/r) = 2/r², so total = -1/r² + 2/r² = 1/r²)
# Wait: ∇²(ln r) = d²/dr²(ln r) + (2/r) d/dr(ln r) = -1/r² + 2/r² = 1/r²
# So ∇²(-A ln r) = -A/r²

A_coeff = np.sqrt(a_0 * G * M_b / 6)
laplacian_analytical = -A_coeff / r**2  # This should be negative

# But we want ρ_DM > 0, so actually:
# ρ_DM ∝ -∇²S_EE (if S_EE ∝ -Φ, and ∇²Φ = 4πGρ)
# 
# Let's be careful:
# Poisson: ∇²Φ = 4πGρ
# Jacobson: S_EE ∝ Φ (with proportionality involving signs)
# 
# If S_EE = -(c³/4Gℏ) Φ (Bekenstein normalization, Φ < 0 for attractive gravity)
# Then ∇²S_EE = -(c³/4Gℏ) ∇²Φ = -(c³/4Gℏ) 4πGρ
# So ρ = -(4Gℏ/c³) ∇²S_EE / (4πG) = -(ℏ/πc³) ∇²S_EE
# i.e., ρ ∝ -∇²S_EE

rho_from_see = -laplacian_S  # proportional to dark matter density

# Analytical version
rho_analytical = A_coeff / r**2  # This is exactly ρ ∝ 1/r² (isothermal)

# ============================================================
# PART 4: THE VERDICT
# ============================================================
# NFW:       ρ ∝ 1/r at small r,  ρ ∝ 1/r³ at large r  (slopes: -1, -3)
# Isothermal: ρ ∝ 1/r² everywhere                         (slope: -2)
# Verlinde:   ρ ∝ 1/r²                                    (slope: -2)
# ∇²S_EE:    ρ ∝ 1/r²                                    (slope: -2)

# Compute log-slopes for comparison
log_r = np.log10(r / kpc)
log_rho_nfw = np.log10(rho_nfw_arr / (M_sun / kpc**3))
log_rho_see = np.log10(np.abs(rho_from_see))
log_rho_analytical = np.log10(rho_analytical)
log_rho_verlinde = np.log10(rho_verlinde / (M_sun / kpc**3))

# Compute slopes (d log ρ / d log r)
def compute_slope(log_r, log_rho):
    return np.gradient(log_rho, log_r)

slope_nfw = compute_slope(log_r, log_rho_nfw)
slope_isothermal = -2.0 * np.ones_like(r)  # constant -2

# ============================================================
# PART 5: ROTATION CURVES
# ============================================================
# v_circ(r) = sqrt(G M(r) / r) where M(r) = ∫ 4π r'² ρ(r') dr'

# NFW enclosed mass
M_enc_nfw = M_nfw(r, rho_0_nfw, r_s_nfw)

# Isothermal enclosed mass: M(r) = 4π ∫ (A/r'²) r'² dr' = 4π A r
# with A = coeff_verlinde / (4π) [in SI, need correct normalization]
# Actually: ρ_verlinde = coeff_verlinde / (4π r²)
# M_verlinde(r) = ∫_0^r 4π r'² ρ_verlinde dr' = ∫ coeff_verlinde dr' = coeff_verlinde * r
M_enc_verlinde = coeff_verlinde * r

# Baryonic enclosed mass (point mass approximation)
M_enc_baryon = M_b * np.ones_like(r)

# Total rotation velocities
v_nfw = np.sqrt(G * (M_enc_baryon + M_enc_nfw) / r) / 1000  # km/s
v_verlinde = np.sqrt(G * (M_enc_baryon + M_enc_verlinde) / r) / 1000  # km/s
v_baryon = np.sqrt(G * M_enc_baryon / r) / 1000  # km/s

# ============================================================
# PLOTTING
# ============================================================
fig = plt.figure(figsize=(16, 20))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# --- Panel 1: Density profiles ---
ax1 = fig.add_subplot(gs[0, 0])
r_kpc = r / kpc

# Normalize both to same value at r = 10 kpc for shape comparison
r_norm = 10 * kpc
idx_norm = np.argmin(np.abs(r - r_norm))

rho_nfw_norm = rho_nfw_arr / rho_nfw_arr[idx_norm]
rho_iso_norm = rho_analytical / rho_analytical[idx_norm]

ax1.loglog(r_kpc, rho_nfw_norm, 'b-', linewidth=2.5, label='NFW (ΛCDM prediction)')
ax1.loglog(r_kpc, rho_iso_norm, 'r--', linewidth=2.5, label='∇²S_EE ∝ 1/r² (ERI prediction)')
ax1.set_xlabel('r  [kpc]', fontsize=13)
ax1.set_ylabel('ρ(r) / ρ(10 kpc)  [normalized]', fontsize=13)
ax1.set_title('Dark Matter Density Profile', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_xlim(0.1, 300)
ax1.set_ylim(1e-4, 1e4)
ax1.grid(True, alpha=0.3)

# Add slope annotations
ax1.annotate('slope = −1\n(NFW inner)', xy=(0.3, 300), fontsize=10, color='blue')
ax1.annotate('slope = −2\n(ERI everywhere)', xy=(0.3, 30), fontsize=10, color='red')
ax1.annotate('slope = −3\n(NFW outer)', xy=(100, 0.005), fontsize=10, color='blue')

# --- Panel 2: Log-slope comparison ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogx(r_kpc, slope_nfw, 'b-', linewidth=2.5, label='NFW slope')
ax2.axhline(y=-2, color='r', linestyle='--', linewidth=2.5, label='∇²S_EE slope = −2')
ax2.axhline(y=-1, color='gray', linestyle=':', alpha=0.5, label='slope = −1')
ax2.axhline(y=-3, color='gray', linestyle=':', alpha=0.5, label='slope = −3')
ax2.set_xlabel('r  [kpc]', fontsize=13)
ax2.set_ylabel('d log ρ / d log r', fontsize=13)
ax2.set_title('Log-Slope of Density Profile', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_xlim(0.1, 300)
ax2.set_ylim(-4, 0)
ax2.grid(True, alpha=0.3)
ax2.fill_between(r_kpc, slope_nfw, -2, alpha=0.15, color='purple', 
                  label='Difference region')

# --- Panel 3: Rotation curves ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogx(r_kpc, v_baryon, 'g:', linewidth=2, label='Baryonic only')
ax3.semilogx(r_kpc, v_nfw, 'b-', linewidth=2.5, label='Baryon + NFW')
ax3.semilogx(r_kpc, v_verlinde, 'r--', linewidth=2.5, label='Baryon + ∇²S_EE (isothermal)')
ax3.set_xlabel('r  [kpc]', fontsize=13)
ax3.set_ylabel('v_circ  [km/s]', fontsize=13)
ax3.set_title('Rotation Curves', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.set_xlim(1, 300)
ax3.set_ylim(0, 400)
ax3.grid(True, alpha=0.3)

# --- Panel 4: S_EE(r) profile ---
ax4 = fig.add_subplot(gs[1, 1])
# Normalize S_EE for display
S_display = S_EE / np.abs(S_EE[idx_norm])
ax4.semilogx(r_kpc, S_display, 'k-', linewidth=2.5)
ax4.set_xlabel('r  [kpc]', fontsize=13)
ax4.set_ylabel('S_EE(r) / |S_EE(10 kpc)|', fontsize=13)
ax4.set_title('Entanglement Entropy Profile S_EE(r)', fontsize=15, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.annotate('Newtonian: −GM/r\n(dominates at small r)', 
             xy=(0.5, S_display[100]), fontsize=10, color='blue',
             xytext=(2, S_display[100]*1.2),
             arrowprops=dict(arrowstyle='->', color='blue'))
ax4.annotate('Verlinde: −A·ln(r)\n(dominates at large r)', 
             xy=(100, S_display[-200]), fontsize=10, color='red',
             xytext=(30, S_display[-200]*0.7),
             arrowprops=dict(arrowstyle='->', color='red'))

# --- Panel 5: The verdict ---
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

verdict_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                    COMPUTATION RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  S_EE(r) = −GM/r − √(a₀GM/6) · ln(r/r₀)    [Jacobson-Verlinde framework]

  ∇²S_EE(r) = −√(a₀GM/6) / r²                [analytically exact for r > 0]

  ρ_DM ∝ −∇²S_EE = √(a₀GM/6) / r²            [ISOTHERMAL PROFILE, slope = −2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  COMPARISON TO NFW:
  ┌──────────────────┬────────────────┬─────────────────┬───────────────────────────────────┐
  │                  │ Inner (r ≪ rₛ) │ Outer (r ≫ rₛ)  │ Rotation curve                    │
  ├──────────────────┼────────────────┼─────────────────┼───────────────────────────────────┤
  │ NFW (ΛCDM)       │ ρ ∝ 1/r        │ ρ ∝ 1/r³        │ Rises, peaks, slowly declines     │
  │ ∇²S_EE (ERI)     │ ρ ∝ 1/r²       │ ρ ∝ 1/r²        │ Flat (v = const)                  │
  │ Observed          │ Often cored    │ Uncertain       │ Flat to ~100 kpc                  │
  └──────────────────┴────────────────┴─────────────────┴───────────────────────────────────┘

  VERDICT: ∇²S_EE does NOT match NFW.

  It produces an isothermal profile (ρ ∝ 1/r²), which gives flat rotation curves
  (v = const) — matching the primary observational fact. But the spatial profile
  differs from NFW at both small r (ERI is steeper: slope −2 vs −1) and large r
  (ERI is shallower: slope −2 vs −3).

  The isothermal result is IDENTICAL to Verlinde (2017). This is not a coincidence —
  the Verlinde entropy displacement IS the ∇²S_EE computation, just expressed
  differently. The ERI framework reproduces Verlinde's emergent gravity exactly.

  IMPLICATIONS FOR THE PROGRAMME:
  • The original claim (∇²S_EE matches NFW) is FALSIFIED by this computation.
  • The revised claim: ∇²S_EE gives isothermal dark matter (ρ ∝ 1/r²), which is
    Verlinde emergent gravity in the Jacobson-Verlinde entanglement framework.
  • This gives flat rotation curves — the right answer to the right question.
  • The inner profile (slope −2 vs observed cores) may be softened by the ε-floor
    regularization in the TH(a,d) architecture — this requires further computation.
  • The outer profile (slope −2 vs NFW slope −3) predicts MORE dark matter at
    large radii than ΛCDM — testable by weak lensing at r > 200 kpc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax5.text(0.02, 0.98, verdict_text, transform=ax5.transAxes,
         fontsize=9, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('GRAVITAS Prediction 2: ∇²S_EE(r) vs NFW\nERI Labs · April 2026', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/home/claude/gravitas_prediction2.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()

# ============================================================
# NUMERICAL SUMMARY
# ============================================================
print("=" * 70)
print("GRAVITAS PREDICTION 2: NUMERICAL RESULTS")
print("=" * 70)
print()
print(f"Galaxy parameters (Milky Way-like):")
print(f"  Baryonic mass: M_b = {M_b/M_sun:.1e} M_sun")
print(f"  NFW scale radius: r_s = {r_s_nfw/kpc:.0f} kpc")
print(f"  Hubble acceleration: a_0 = c*H_0 = {a_0:.2e} m/s²")
print()
print(f"Verlinde coefficient: sqrt(a_0*GM_b/6) = {A_coeff:.4e} m²/s²")
print()
print("DENSITY PROFILE COMPARISON AT KEY RADII:")
print(f"{'r (kpc)':<12} {'NFW slope':<12} {'ERI slope':<12} {'Match?':<10}")
print("-" * 46)
for r_check in [0.5, 1, 5, 10, 20, 50, 100, 200]:
    idx = np.argmin(np.abs(r - r_check * kpc))
    slope_n = slope_nfw[idx]
    slope_e = -2.0
    match = "~" if abs(slope_n - slope_e) < 0.3 else "NO"
    print(f"{r_check:<12.1f} {slope_n:<12.2f} {slope_e:<12.2f} {match:<10}")

print()
print("ROTATION VELOCITY COMPARISON AT KEY RADII:")
print(f"{'r (kpc)':<12} {'v_NFW (km/s)':<15} {'v_ERI (km/s)':<15} {'v_baryon':<12}")
print("-" * 54)
for r_check in [5, 10, 20, 50, 100, 200]:
    idx = np.argmin(np.abs(r - r_check * kpc))
    print(f"{r_check:<12.1f} {v_nfw[idx]:<15.1f} {v_verlinde[idx]:<15.1f} {v_baryon[idx]:<12.1f}")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print()
print("∇²S_EE(r) = isothermal (ρ ∝ 1/r², slope = -2)")
print("NFW       = cuspy inner (slope -1), steep outer (slope -3)")
print()
print("RESULT: ∇²S_EE does NOT match NFW.")
print()
print("∇²S_EE reproduces Verlinde emergent gravity exactly.")
print("It gives flat rotation curves (the primary observational fact).")
print("It does NOT reproduce the NFW spatial profile.")
print()
print("The original claim must be REVISED:")
print("  OLD: ρ_DM ∝ ∇²S_EE matches NFW")
print("  NEW: ρ_DM ∝ ∇²S_EE gives isothermal dark matter = Verlinde (2017)")
print()
print("TESTABLE PREDICTION (new):")
print("  At r > 200 kpc, isothermal predicts MORE mass than NFW.")
print("  Weak lensing measurements at these radii distinguish the two.")
print("=" * 70)
