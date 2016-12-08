'''
    Statistics needed for POMDP model
'''

# Sensitivity and specificity statistics for self-diagnosis
SDStats = {
    "sens": 0.44,
    "spec": 0.92
}

# Sensitivity and specificity statistics for Mammography
MStats = {
    # Age group 40-49
    0: {
        "sens": 0.722,
        "spec": 0.889
    },
    # Age group 50-54
    1: {
        "sens": 0.722,
        "spec": 0.893
    },
    # Age group 55-59
    2: {
        "sens": 0.81,
        "spec": 0.893
    },
    # Age group 60-69
    3: {
        "sens": 0.81,
        "spec": 0.897
    },
    # Age group 70+
    4: {
        "sens": 0.862,
        "spec": 0.897
    }
}

# Transition Probabilities
# matrix structure:
#               healthy, in-situ, invasive, post-in-situ, post-invasive, death
# healthy
# in-situ
# invasive
# post-in-situ
# post-invasive
# death

# Healthy -> in-situ: same for each (.001)
# In-situ -> invasive: same for each (.58)
# Invasive -> death: same for each (.1)
# Invasive -> invasive: same for each (1 - .1 = .9)
# Healthy -> healthy: 1 - the rest in row
# In-situ -> in-situ: 1 - the rest in row

# -1's denote probabilities that are age-specific

TMatrix = [
[-1, .001, -1, 0, 0, -1],
[0, -1, .58, 0, 0, -1],
[0, 0, .9, 0, 0, .1],
[0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 1]
]

# Healthy -> death / In-situ -> death probabilities
# http://www.cdc.gov/nchs/data/nvsr/nvsr65/nvsr65_08.pdf
# data provided for every five years from ages 40 to 100; assume uniform and
# divide by 10 to get probability for each timestep
death_probs_from_lit = [.009868, 0.015637, 0.024290, 0.035348, 0.049706, 0.071703, 0.109255, 0.170906, 0.271119, 0.426073, 0.614941, 0.786733]
death_probs = [dp / 10.0 for dp in death_probs_from_lit]

# Healthy -> invasive probabilities
# https://seer.cancer.gov/csr/1975_2013/results_merged/topic_lifetime_risk.pdf
# data provided for every ten years from ages 40 to 100; assume uniform and
# divide by 20 to get probability for each timestep
healthy_inv_from_lit = [.0147, .0230, .0347, .0395, .030, .030]
healthy_inv_probs = [hip / 20.0 for hip in healthy_inv_from_lit]
