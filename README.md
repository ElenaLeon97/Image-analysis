# Image-analysis

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning/Preparation](#data-cleaningpreparation)
- [Data Analysis](#data-analysis)
- [Results/Findings](#resultsfindings)
- [Recommendations](#recommendations)
- [Limitations](#limitations)
- [References](#references)

---

### Project Overview
---

This project investigates whether the presence of people in crowdfunding campaign images increases donation amounts. Using YOLOv5 object detection, natural language metrics, and linear regression, the goal is to assess if emotional cues or visual cues (like a human face) drive greater donor engagement.

---

### Data Sources
---

- **Crowdfunding campaign dataset**: Campaign-level metadata (e.g., amount raised, sentiment features, word count).
- **Campaign images**: Manually collected images associated with each campaign.
- **Derived data**:
  - `is_person`: Binary indicator for human presence in campaign image (via YOLOv5).
  - Sentiment and word count features (`posemo`, `negemo`, `WC`) from prior text analysis.

---

### Tools
---

- **Python** (Colab)
- **Libraries**: `torch`, `pandas`, `numpy`, `statsmodels`, `os`
- **YOLOv5s** from Ultralytics (via `torch.hub`)
- **Google Drive** for dataset storage and access

---

### Data Cleaning/Preparation
---

- Used YOLOv5 to detect people in images and assign an `is_person` flag.
- Merged image results with campaign metadata using a unique campaign ID.
- Logged the `raised_USD` variable to normalize the distribution.
- Imputed and cleaned inconsistencies in filenames, image detection outputs, and UID mappings.

---

### Data Analysis
---

- Built an OLS regression model:
```python
# Model specification
raised_USD ~ is_person + posemo + negemo + WC
```
Key Regression Output:

|Variable|Coef.|Std. Err.|t|P>t|[0.025     0.975]
|--------------|--------|-----------|--------|------|--------------------|
| Intercept | 6.0058 | 0.340 | 17.660 | 0.000 | [5.336, 6.676] |
| is_person | 0.4214 | 0.238 | 1.767 | 0.078 | [-0.048, 0.891] |
| posemo | 0.0283 | 0.048 | 0.594 | 0.553 | [-0.065, 0.122] |
| negemo | 0.0791 | 0.073 | 1.087 | 0.278 | [-0.064, 0.222] |
| WC | 0.0020 | 0.001 | 2.543 | 0.012 | [0.000, 0.004] |

R²: 0.050
Adj. R²: 0.033
F-statistic: 3.021 (p = 0.0187)
Observations: 237

- Assessed statistical significance of human presence (`is_person`) in relation to log-transformed amount raised.
- Evaluated additional predictors: positive emotion (`posemo`), negative emotion (`negemo`), and word count (`WC`).
---

### Results/Findings
---

- **Presence of a person** in the campaign image showed a positive but borderline-significant relationship with raised funds (*p = 0.078*).
- **Word count (WC)** was the only variable with a significant positive effect (*p = 0.012*).
- Sentiment variables (`posemo`, `negemo`) were not significant in this model.
- The model explains approximately 5% of the variance in fundraising success (*R² = 0.05*).

---

### Recommendations
---

- Consider including people in campaign visuals to potentially increase donor engagement.
- Focus on well-structured campaign text with sufficient detail (as reflected in word count).
- Further explore interaction effects between image content and textual sentiment.

---

### Limitations
---

- Model has low explanatory power; other unobserved factors likely influence donation behavior.
- Person detection assumes one image per campaign and may not reflect all visuals used.
- Sentiment features derived externally and may not capture emotional nuance accurately.

---

*This project was conducted for educational purposes for the Data (Science) & Analytics Learning Community.*
