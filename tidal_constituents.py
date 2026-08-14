# Source: https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/9410170/harcon.json
CONSTITUENTS = {
    # ── Diurnal ───────────────────────────────────────────────────────────
    "O1":   25.8194,   # Principal lunar diurnal
    "K1":   23.9345,   # Luni-solar diurnal
    "S1":   24.0000,   # Solar diurnal
    "P1":   24.0659,   # Solar diurnal
    "Q1":   26.8684,   # Larger lunar elliptic diurnal
    "2Q1":  28.0063,   # Larger lunar elliptic diurnal 2
    "rho1": 26.7230,   # Larger lunar evectional diurnal
    "J1":   23.0985,   # Smaller lunar elliptic diurnal
    "OO1":  22.3061,   # Second overtide of O1
    "M1":   24.8412,   # Smaller lunar elliptic diurnal

    # ── Semidiurnal ───────────────────────────────────────────────────────
    "M2":      12.4206,   # Principal lunar semidiurnal
    "S2":      12.0000,   # Principal solar semidiurnal
    "N2":      12.6583,   # Larger lunar elliptic semidiurnal
    "K2":      11.9672,   # Luni-solar semidiurnal
    "L2":      12.1916,   # Smaller lunar elliptic semidiurnal
    "lambda2": 12.2218,   # Smaller lunar evectional semidiurnal
    "2N2":     12.9055,   # Second larger lunar elliptic
    "mu2":     12.8717,   # Variational (lunar evectional)
    "nu2":     12.6260,   # Larger lunar evectional
    "T2":      12.0164,   # Larger solar elliptic
    "R2":      11.9840,   # Smaller solar elliptic
    "2SM2":    11.6070,   # Shallow water semidiurnal

    # ── Shallow-water (overtides and compound tides) ─────────────────────
    "M3":    8.2803,   # Lunar terdiurnal — significant in shallow water
    "MK3":   8.1774,   # Compound: M2 + K1
    "2MK3":  8.3864,   # Compound: 2M2 - K1
    "M4":    6.2103,   # Shallow-water overtide of M2
    "MS4":   6.1033,   # Shallow-water quarter-diurnal
    "MN4":   6.2695,   # Shallow-water quarter-diurnal
    "S4":    6.0000,   # Shallow-water overtide of S2
    "M6":    4.1402,   # Shallow-water sixth-diurnal
    "S6":    4.0000,   # Shallow-water sixth-diurnal solar
    "M8":    3.1051,   # Shallow-water eighth-diurnal

    # ── Long period ───────────────────────────────────────────────────────
    "Sa":   8766.0,    # Solar annual
    "Ssa":  4383.0,    # Solar semi-annual
    "MSf":   354.37,   # Lunisolar synodic fortnightly
    "Mm":    661.3,    # Lunar monthly
    "Mf":    327.9,    # Lunisolar fortnightly
}
