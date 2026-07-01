from fuzzy_logic import calculate_risk


result = calculate_risk(

    bp_value=145,

    hr_value=115,

    temp_value=38.5,

    resp_value=26
)

print("\n===== FUZZY RESULT =====\n")

print(result)
