"""
This module contains functions to plot 1-D static tolerances.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ultra.config import CONFIG_ULTRA


def plot_mus_all_hexrings(mu1, mu2, mu3, mu4, mu5, c0, out_dir, save=False):
    """Plots static segment level tolerances for all modes, all SCDA designs.

    Parameters
    ----------
    mu1 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 1-HexRingTelescope.
    mu2 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 2-HexRingTelescope.
    mu3 : numpy.ndarray
        Each element  represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 3-HexRingTelescope.
    mu4 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 4-HexRingTelescope.
    mu5 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 5-HexRingTelescope.
    c0 : float
        The set-target contrast for which the above tolerances were calculated.
    out_dir : str
        path where the plot will be saved
    save : bool, default False
        whether to save the plot
    """
    plt.figure(figsize=(10, 10))
    plt.title("Modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=20)
    plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
    plt.xlabel("Mode Index", fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
    plt.plot(mu1 * 1e3, label="1-HexRingTelescope")
    plt.plot(mu2 * 1e3, label="2-HexRingTelescope")
    plt.plot(mu3 * 1e3, label="3-HexRingTelescope")
    plt.plot(mu4 * 1e3, label="4-HexRingTelescope")
    plt.plot(mu5 * 1e3, label="5-HexRingTelescope")
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, 'mus_1d_multi_modes_%s.png' % c0))


def plot_single_thermal_mode_all_hex(mu1, mu2, mu3, mu4, mu5, c0, mode, out_dir, save=False, inner_segments=False):
    """Plots static tolerance coefficients for a single Harris mode, for all SCDA designs.

    Parameters
    ----------
    mu1 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 1-HexRingTelescope.
    mu2 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 2-HexRingTelescope.
    mu3 : numpy.ndarray
        Each element  represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 3-HexRingTelescope.
    mu4 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 4-HexRingTelescope.
    mu5 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per segment level aberration mode
        for the 5-HexRingTelescope.
    c0 : float
        The set-target contrast for which the above tolerances were calculated.
    mode : str
        name of the segment level zernike/harris thermal aberration
        "Faceplates Silvered" or "Piston", "Bulk" or "Tip", "Gradiant Radial" or "Tilt",
        "Gradiant X lateral" or "Defocus", or "Gradiant Z axial" or "Astig"
    out_dir : str
        path where the plot will be saved
    save : bool, default False
        whether to save the plot
    inner_segments : bool, default False
        whether to plot tolerances for the inner segments only
    """
    # for 1-HexRingTelescope
    mus1_table = np.zeros([5, 7])
    for qq in range(5):
        for kk in range(7):
            mus1_table[qq, kk] = mu1[qq + kk * 5]

    # for 2-HexRingTelescope
    mus2_table = np.zeros([5, 19])
    for qq in range(5):
        for kk in range(19):
            mus2_table[qq, kk] = mu2[qq + kk * 5]

    # for 3-HexRingTelescope
    mus3_table = np.zeros([5, 31])
    for qq in range(5):
        for kk in range(31):
            mus3_table[qq, kk] = mu3[qq + kk * 5]

    # for 4-HexRingTelescope
    mus4_table = np.zeros([5, 55])
    for qq in range(5):
        for kk in range(55):
            mus4_table[qq, kk] = mu4[qq + kk * 5]

    # for 5-HexRingTelescope
    mus5_table = np.zeros([5, 85])
    for qq in range(5):
        for kk in range(85):
            mus5_table[qq, kk] = mu5[qq + kk * 5]

    if mode == "Faceplates Silvered" or mode == "Piston":
        num = 0
    elif mode == "Bulk" or mode == "Tip":
        num = 1
    elif mode == "Gradiant Radial" or mode == "Tilt":
        num = 2
    elif mode == "Gradiant X lateral" or mode == "Defocus":
        num = 3
    elif mode == "Gradiant Z axial" or mode == "Astig":
        num = 4

    # For 1 Hex:
    ring1_mean = 0
    for i in range(1, 7):
        ring1_mean = ring1_mean + mus1_table[num, i] * 1e3
    print("1Hex:", "ring1:", ring1_mean / 6)

    # For 2 Hex:
    ring1_mean = 0
    for i in range(1, 7):
        ring1_mean = ring1_mean + mus2_table[num, i] * 1e3
    ring2_mean = 0
    for i in range(7, 19):
        ring2_mean = ring2_mean + mus2_table[num, i] * 1e3

    print("2Hex:", "ring1:", ring1_mean / 6, "ring2:", ring2_mean / 12)

    # For 3 Hex:
    ring1_mean = 0
    for i in range(1, 7):
        ring1_mean = ring1_mean + mus3_table[num, i] * 1e3
    ring2_mean = 0
    for i in range(7, 19):
        ring2_mean = ring2_mean + mus3_table[num, i] * 1e3
    ring3_mean = 0
    for i in range(19, 31):
        ring3_mean = ring3_mean + mus3_table[num, i] * 1e3
    print("3Hex:", "ring1:", ring1_mean / 6, "ring2:", ring2_mean / 12, "ring3:", ring3_mean / 12)

    # For 4 Hex:
    ring1_mean = 0
    for i in range(1, 7):
        ring1_mean = ring1_mean + mus4_table[num, i] * 1e3
    ring2_mean = 0
    for i in range(7, 19):
        ring2_mean = ring2_mean + mus4_table[num, i] * 1e3
    ring3_mean = 0
    for i in range(19, 37):
        ring3_mean = ring3_mean + mus4_table[num, i] * 1e3
    ring4_mean = 0
    for i in range(37, 55):
        ring4_mean = ring4_mean + mus4_table[num, i] * 1e3
    print("4Hex:", "ring1:", ring1_mean / 6, "ring2:", ring2_mean / 12, "ring3:", ring3_mean / 12, "ring4:", ring4_mean / 18)

    # For 5 Hex:
    ring1_mean = 0
    for i in range(1, 7):
        ring1_mean = ring1_mean + mus5_table[num, i] * 1e3
    ring2_mean = 0
    for i in range(7, 19):
        ring2_mean = ring2_mean + mus5_table[num, i] * 1e3
    ring3_mean = 0
    for i in range(19, 37):
        ring3_mean = ring3_mean + mus5_table[num, i] * 1e3
    ring4_mean = 0
    for i in range(37, 61):
        ring4_mean = ring4_mean + mus5_table[num, i] * 1e3
    ring5_mean = 0
    for i in range(61, 85):
        ring5_mean = ring5_mean + mus5_table[num, i] * 1e3
    print("5Hex:", "ring1:", ring1_mean / 6, "ring2:", ring2_mean / 12, "ring3:", ring3_mean / 12, "ring4:", ring4_mean / 24, "ring5:", ring5_mean / 24)

    plt.figure(figsize=(10, 10))
    plt.title(str(mode) + " modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
    plt.ylabel("Weight per segment (in units of pm)", fontsize=20)
    plt.xlabel("Segment Number", fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
    plt.plot(mus1_table[num] * 1e3, label="1-HexRingTelescope", marker="o")
    plt.plot(mus2_table[num] * 1e3, label="2-HexRingTelescope", marker="s")
    plt.plot(mus3_table[num] * 1e3, label="3-HexRingTelescope", marker="p")
    plt.plot(mus4_table[num] * 1e3, label="4-HexRingTelescope", marker="P")
    plt.plot(mus5_table[num] * 1e3, label="5-HexRingTelescope", marker="H")
    if inner_segments:
        plt.xlabel("Inner Segment Number", fontsize=20)
        plt.yticks(np.arange(np.min(mus3_table[num] * 1e3), np.max(mus4_table[num] * 1e3), 0.5))
        plt.ylim(1, 5)
        plt.xlim(0, 15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, str(mode) + '_mus_%s.png' % c0))


if __name__ == '__main__':

    c_target = 1e-11

    # Get thermal tolerance coefficients for corresponding c_target
    mus_data_path = CONFIG_ULTRA.get('local_path', 'local_data_path')
    mus1 = np.genfromtxt(os.path.join(mus_data_path, 'mus_Hex_1_1e-11.csv'), delimiter=',')
    mus2 = np.genfromtxt(os.path.join(mus_data_path, 'mus_Hex_2_1e-11.csv'), delimiter=',')
    mus3 = np.genfromtxt(os.path.join(mus_data_path, 'mus_Hex_3_1e-11.csv'), delimiter=',')
    mus4 = np.genfromtxt(os.path.join(mus_data_path, 'mus_Hex_4_1e-11.csv'), delimiter=',')
    mus5 = np.genfromtxt(os.path.join(mus_data_path, 'mus_Hex_5_1e-11.csv'), delimiter=',')

    plot_dir = CONFIG_ULTRA.get('local_path', 'local_analysis_path')

    plot_single_thermal_mode_all_hex(mus1, mus2, mus3, mus4, mus5, c_target,
                                     mode="Gradiant Z axial", out_dir=plot_dir, save=True, inner_segments=False)
    plot_mus_all_hexrings(mus1, mus2, mus3, mus4, mus5, c_target, plot_dir, save=True)
