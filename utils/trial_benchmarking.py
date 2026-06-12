import numpy as np

def standardized_difference_hr(hr_emul, ci_low_emul, ci_high_emul,
                               hr_rct, ci_low_rct, ci_high_rct):
    # log estimates
    log_hr_emul = np.log(hr_emul)
    log_hr_rct = np.log(hr_rct)

    # derive SE from 95% CI on log scale
    se_emul = (np.log(ci_high_emul) - np.log(ci_low_emul)) / (2 * 1.96)
    se_rct = (np.log(ci_high_rct) - np.log(ci_low_rct)) / (2 * 1.96)

    # standardized difference
    std_diff = (log_hr_emul - log_hr_rct) / np.sqrt(se_emul**2 + se_rct**2)

    return std_diff

def standardized_difference_risk_difference(
    rd_emul, ci_low_emul, ci_high_emul,
    rd_rct, ci_low_rct, ci_high_rct
):
    se_emul = (ci_high_emul - ci_low_emul) / (2 * 1.96)
    se_rct = (ci_high_rct - ci_low_rct) / (2 * 1.96)

    z = (rd_emul - rd_rct) / np.sqrt(se_emul**2 + se_rct**2)

    return z