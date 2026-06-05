import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================
# INPUT VARIABLES
# =========================

bp = ctrl.Antecedent(
    np.arange(80, 201, 1),
    'bp'
)

heart_rate = ctrl.Antecedent(
    np.arange(40, 181, 1),
    'heart_rate'
)

temperature = ctrl.Antecedent(
    np.arange(34, 42, 0.1),
    'temperature'
)

resp_rate = ctrl.Antecedent(
    np.arange(10, 41, 1),
    'resp_rate'
)


# =========================
# OUTPUT VARIABLE
# =========================

risk = ctrl.Consequent(
    np.arange(0, 101, 1),
    'risk'
)


# =========================
# MEMBERSHIP FUNCTIONS
# =========================

bp['normal'] = fuzz.trimf(
    bp.universe,
    [80, 100, 120]
)

bp['elevated'] = fuzz.trimf(
    bp.universe,
    [115, 130, 145]
)

bp['high'] = fuzz.trimf(
    bp.universe,
    [140, 170, 200]
)


heart_rate['normal'] = fuzz.trimf(
    heart_rate.universe,
    [50, 75, 100]
)

heart_rate['high'] = fuzz.trimf(
    heart_rate.universe,
    [90, 120, 180]
)


temperature['normal'] = fuzz.trimf(
    temperature.universe,
    [35, 36.5, 37.5]
)

temperature['high'] = fuzz.trimf(
    temperature.universe,
    [37, 38.5, 41]
)


resp_rate['normal'] = fuzz.trimf(
    resp_rate.universe,
    [12, 16, 20]
)

resp_rate['high'] = fuzz.trimf(
    resp_rate.universe,
    [18, 25, 40]
)


risk['low'] = fuzz.trimf(
    risk.universe,
    [0, 20, 40]
)

risk['moderate'] = fuzz.trimf(
    risk.universe,
    [30, 50, 70]
)

risk['high'] = fuzz.trimf(
    risk.universe,
    [60, 80, 100]
)


# =========================
# FUZZY RULES
# =========================

rule1 = ctrl.Rule(
    bp['normal'] &
    heart_rate['normal'] &
    temperature['normal'] &
    resp_rate['normal'],

    risk['low']
)

rule2 = ctrl.Rule(
    bp['elevated'] |
    heart_rate['high'] |
    temperature['high'],

    risk['moderate']
)

rule3 = ctrl.Rule(
    bp['high'] &
    heart_rate['high'],

    risk['high']
)

rule4 = ctrl.Rule(
    bp['high'] &
    temperature['high'] &
    resp_rate['high'],

    risk['high']
)

# DEFAULT FALLBACK RULE
rule5 = ctrl.Rule(
    bp['elevated'] &
    resp_rate['normal'],

    risk['moderate']
)


# =========================
# CONTROL SYSTEM
# =========================

risk_ctrl = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5
])


# =========================
# MAIN FUNCTION
# =========================

def calculate_risk(

    bp_value,
    hr_value,
    temp_value,
    resp_value
):

    # CREATE NEW SIMULATION EACH TIME
    risk_simulator = ctrl.ControlSystemSimulation(
        risk_ctrl
    )

    # INPUTS
    risk_simulator.input['bp'] = bp_value

    risk_simulator.input[
        'heart_rate'
    ] = hr_value

    risk_simulator.input[
        'temperature'
    ] = temp_value

    risk_simulator.input[
        'resp_rate'
    ] = resp_value


    # COMPUTE
    risk_simulator.compute()


    # SAFE OUTPUT HANDLING
    try:

        score = risk_simulator.output['risk']

    except:

        score = 50


    # RISK LEVEL
    if score < 40:

        level = "LOW"

    elif score < 70:

        level = "MODERATE"

    else:

        level = "HIGH"


    return {

        "risk_score":
            round(float(score), 2),

        "risk_level":
            level
    }
