"""Clean S-JEPA architecture diagram - ample spacing, no overlap."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(20, 13))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')

# Color palette - minimal, clean
C_INPUT  = '#E8E8E8'
C_ENC    = '#B8D4E8'
C_LATENT = '#FFE0B0'
C_PRED   = '#C8E6C8'
C_LOSS   = '#F4C8C8'
C_CRITIC = '#E0C8E8'
C_DEC    = '#D8D8D8'

def box(x, y, w, h, label, color, fontsize=11, bold=True):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                        linewidth=1.5, edgecolor='black', facecolor=color)
    ax.add_patch(b)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight)

def arrow(x1, y1, x2, y2, label='', style='->', color='black', label_offset=(0.1, 0.15)):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                         mutation_scale=18, linewidth=1.5, color=color)
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha='center', va='center', fontsize=9, style='italic')

# ============== TITLE ==============
ax.text(10, 12.7, 'S-JEPA Architecture',
        ha='center', va='center', fontsize=18, fontweight='bold')
ax.text(10, 12.25, 'Latent-Space Predictive World Model with Cross-Domain Transfer',
        ha='center', va='center', fontsize=11, style='italic', color='#555')

# ============== INPUTS (TOP) ==============
y_in = 10.3
box(0.5,  y_in, 2.5, 0.9, 'State  x', C_INPUT)
box(8.5,  y_in, 2.5, 0.9, 'Action  a', C_INPUT)
box(16.5, y_in, 2.5, 0.9, 'Next State  y', C_INPUT)

# ============== ENCODERS / EMBEDDER ==============
y_enc = 8.5
box(0.5,  y_enc, 2.5, 1.0, 'Encoder\nE_θ  (MLP)', C_ENC)
box(8.5,  y_enc, 2.5, 1.0, 'Action Embedder\nA_θ', C_ENC)
box(16.5, y_enc, 2.5, 1.0, 'Target Encoder\nE_ξ  (EMA)', C_ENC)

arrow(1.75, y_in, 1.75, y_enc + 1.0)
arrow(9.75, y_in, 9.75, y_enc + 1.0)
arrow(17.75, y_in, 17.75, y_enc + 1.0)


# ============== LATENTS ==============
y_lat = 6.6
box(0.5,  y_lat, 2.5, 1.0, 'Latent  z_x\n(256-D)', C_LATENT)
box(8.5,  y_lat, 2.5, 1.0, 'Action Latent\n(32-D)', C_LATENT)
box(16.5, y_lat, 2.5, 1.0, 'Target Latent  z_y\n(256-D)', C_LATENT)

arrow(1.75, y_enc, 1.75, y_lat + 1.0)
arrow(9.75, y_enc, 9.75, y_lat + 1.0)
arrow(17.75, y_enc, 17.75, y_lat + 1.0)

# ============== PREDICTOR ==============
y_pred = 4.6
box(5.5, y_pred, 6.0, 1.1, 'Predictor   P_φ  (MLP)\n(state latent, action latent) → next state latent',
    C_PRED, fontsize=11)

# arrows from z_x and action latent into predictor
arrow(1.75, y_lat, 5.5, y_pred + 0.55)
arrow(9.75, y_lat, 9.75, y_pred + 1.1)

# ============== PREDICTED LATENT ==============
y_pp = 2.7
box(5.5, y_pp, 6.0, 0.9, 'Predicted Next Latent  ẑ_y', C_LATENT)
arrow(8.5, y_pred, 8.5, y_pp + 0.9)

# ============== LOSS ==============
y_loss = 0.8
box(5.5, y_loss, 6.0, 1.1,
    'JEPA Loss\n‖ẑ_y − sg[z_y]‖²  +  λ · SIGReg',
    C_LOSS, fontsize=11)
arrow(8.5, y_pp, 8.5, y_loss + 1.1)

# Target latent → loss (long curved arrow with stop-gradient marker)
ax.annotate('', xy=(11.5, y_loss + 0.55), xytext=(16.5, y_lat + 0.5),
            arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5,
                            connectionstyle="arc3,rad=0.3"))
ax.text(14.7, 3.4, 'sg [·]\n(stop-gradient)', ha='center', va='center',
        fontsize=9, color='#666', style='italic')

# ============== CRITIC (right side, below latents) ==============
box(13.5, y_pred, 5.0, 1.1,
    'Critic   C_ψ\nF(s, a) = energy score',
    C_CRITIC, fontsize=11)
arrow(11.5, y_pp + 0.45, 13.5, y_pred + 0.55)
ax.text(13.5, 3.85, 'Used at inference\nto rank actions',
        ha='center', va='center', fontsize=9, color='#666', style='italic')

# ============== DECODER (bottom right, eval only) ==============
box(13.5, y_loss, 5.0, 1.1,
    'Decoder  D_ψ  (eval only)\nẑ → reconstructed state',
    C_DEC, fontsize=10)
arrow(11.5, y_pp + 0.2, 13.5, y_loss + 0.85, color='#888')

# ============== LEGEND - frozen vs trained =============
legend_x, legend_y, legend_w, legend_h = 0.3, 0.3, 4.7, 2.4
ax.add_patch(mpatches.Rectangle((legend_x, legend_y), legend_w, legend_h,
                                 linewidth=1.0, edgecolor='black',
                                 facecolor='white'))
ax.text(legend_x + legend_w/2, legend_y + legend_h - 0.3,
        'Cross-Domain Transfer',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(legend_x + 0.2, legend_y + legend_h - 0.75,
        'Re-train per domain:',
        ha='left', va='center', fontsize=9, fontweight='bold')
ax.text(legend_x + 0.4, legend_y + legend_h - 1.10,
        'Encoder E_θ',
        ha='left', va='center', fontsize=9)
ax.text(legend_x + 0.4, legend_y + legend_h - 1.45,
        'Action Embedder A_θ',
        ha='left', va='center', fontsize=9)
ax.text(legend_x + 0.2, legend_y + legend_h - 1.85,
        'Frozen across domains:',
        ha='left', va='center', fontsize=9, fontweight='bold')
ax.text(legend_x + 0.4, legend_y + legend_h - 2.20,
        'Predictor P_φ, Critic C_ψ',
        ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('sjepa_architecture_clean.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print("Saved: sjepa_architecture_clean.png")
