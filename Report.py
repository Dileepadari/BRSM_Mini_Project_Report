from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel
from pingouin import rm_anova, ttest, bayesfactor_ttest

# Load dataset
file_path = "cleaned_merged_data.csv"  # Ensure correct file path
df = pd.read_csv(file_path)

# Rename columns
df.rename(columns={
    "Intended.or.Unintended.Outcome": "Intentionality",
    "Reaction.Time.on.Outcome.Response": "ReactionTime",
    "Response...Short.or.Long": "PerceivedDuration",
    "Stimulus.Delay": "StimulusDelay",
    "Objective.Duration.of.Stimulus": "ObjectiveDuration"
}, inplace=True)

# Convert categorical variables
df["Intentionality"] = df["Intentionality"].map({1: "Intended", 0: "Unintended"})
df["PerceivedDuration"] = df["PerceivedDuration"].map({"short": 0, "long": 1})

### 1. REPEATED MEASURES ANOVA ###
anova_model = rm_anova(
    data=df,
    dv="ReactionTime",
    within=["Intentionality", "StimulusDelay"],
    subject="Subject.ID"
)
print("Repeated Measures ANOVA Results for Reaction Time:")
print(anova_model)

anova_duration = rm_anova(
    data=df,
    dv="PerceivedDuration",
    within=["Intentionality", "StimulusDelay"],
    subject="Subject.ID"
)
print("Repeated Measures ANOVA Results for Perceived Duration:")
print(anova_duration)

### 2. BAYESIAN PAIRED T-TESTS (Fixed Code) ###
# Compute paired t-test for 250ms delay
t_stat_250, p_value_250 = ttest_rel(
    df[df["StimulusDelay"] == 250]["PerceivedDuration"],
    df[df["StimulusDelay"] == 250]["PerceivedDuration"]
)
bf_250 = bayesfactor_ttest(t_stat_250, len(df[df["StimulusDelay"] == 250]))

# Compute paired t-test for 1000ms delay
t_stat_1000, p_value_1000 = ttest_rel(
    df[df["StimulusDelay"] == 1000]["PerceivedDuration"],
    df[df["StimulusDelay"] == 1000]["PerceivedDuration"]
)
bf_1000 = bayesfactor_ttest(t_stat_1000, len(df[df["StimulusDelay"] == 1000]))

print(f"Bayesian Paired t-test (250ms Delay): BF10 = {bf_250}")
print(f"Bayesian Paired t-test (1000ms Delay): BF10 = {bf_1000}")

### 3. REGRESSION ANALYSIS ###
reg_model = smf.ols("PerceivedDuration ~ ReactionTime + StimulusDelay", data=df).fit()
print(reg_model.summary())

### 4. DATA VISUALIZATIONS ###

# 4.1 Psychometric Curve
sns.lmplot(
    x="ObjectiveDuration", 
    y="PerceivedDuration", 
    hue="Intentionality", 
    col="StimulusDelay",
    data=df, 
    logistic=True
)
plt.xlabel("Objective Duration (ms)")
plt.ylabel("Proportion of 'Long' Responses")
plt.suptitle("Psychometric Curve: Perceived Duration")
plt.savefig("psychometric_curve.png")
plt.show()

# 4.2 Box Plot for Reaction Time
plt.figure(figsize=(8, 5))
sns.boxplot(x="StimulusDelay", y="ReactionTime", hue="Intentionality", data=df)
plt.xlabel("Stimulus Delay (ms)")
plt.ylabel("Reaction Time (ms)")
plt.title("Reaction Time Distribution")
plt.savefig("reaction_time_boxplot.png")
plt.show()

# 4.3 Bar Chart for Proportion of "Long" Responses
proportion_data = df.groupby(["Intentionality", "StimulusDelay"])["PerceivedDuration"].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="StimulusDelay", y="PerceivedDuration", hue="Intentionality", data=proportion_data)
plt.xlabel("Stimulus Delay (ms)")
plt.ylabel("Proportion of 'Long' Responses")
plt.title("Effect of Intentionality on Perceived Duration")
plt.savefig("long_response_proportion.png")
plt.show()

# 4.4 Correlation Heatmap
corr_matrix = df[["ReactionTime", "StimulusDelay", "PerceivedDuration", "ObjectiveDuration"]].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()


# Load dataset
file_path = "cleaned_merged_data.csv"  
df = pd.read_csv(file_path)

# Convert "short"/"long" responses to binary (0 = short, 1 = long)
df["PerceivedDuration"] = df["Response...Short.or.Long"].map({"short": 0, "long": 1})

# Convert Objective Duration into categorical bins (Short: <=500ms, Long: >500ms)
df["ObjectiveCategory"] = np.where(df["Objective.Duration.of.Stimulus"] <= 500, 0, 1)

# Confusion Matrix
conf_matrix = confusion_matrix(df["ObjectiveCategory"], df["PerceivedDuration"])
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Short", "Long"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: Estimated vs. Actual Duration")
plt.savefig("confusion_matrix.png")
plt.show()

# Distribution of Estimation Errors
df["Error"] = df["PerceivedDuration"] - df["ObjectiveCategory"]

plt.figure(figsize=(8, 5))
sns.histplot(df["Error"], bins=20, kde=True)
plt.xlabel("Estimation Error (Perceived - Actual)")
plt.ylabel("Frequency")
plt.title("Distribution of Estimation Errors")
plt.savefig("error_distribution.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(
    x="Stimulus.Delay", 
    y="PerceivedDuration", 
    data=df, 
    estimator=np.mean, 
    ci=95, 
    palette="coolwarm"
)
plt.xlabel("Stimulus Delay (ms)")
plt.ylabel("Proportion of 'Long' Responses")
plt.title("Effect of Delay on Perceived Duration")
plt.savefig("delay_effect.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.regplot(
    x="Objective.Duration.of.Stimulus", 
    y="PerceivedDuration", 
    data=df, 
    logistic=True,
    scatter_kws={"alpha": 0.3}
)
plt.xlabel("Objective Stimulus Duration (ms)")
plt.ylabel("Proportion of 'Long' Responses")
plt.title("Influence of Objective Duration on Perceived Duration")
plt.savefig("objective_vs_perceived.png")
plt.show()


plt.figure(figsize=(12, 5))
sns.boxplot(
    x="Subject.ID", 
    y="PerceivedDuration", 
    hue="Stimulus.Delay", 
    data=df, 
    palette="coolwarm"
)
plt.xticks(rotation=90)
plt.xlabel("Participant ID")
plt.ylabel("Proportion of 'Long' Responses")
plt.title("Individual Differences in Delay Effects on Perceived Duration")
plt.legend(title="Stimulus Delay (ms)")
plt.savefig("individual_differences.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(
    df[df["Stimulus.Delay"] == 250]["PerceivedDuration"], 
    label="250ms Delay", 
    fill=True, 
    alpha=0.5
)
sns.kdeplot(
    df[df["Stimulus.Delay"] == 1000]["PerceivedDuration"], 
    label="1000ms Delay", 
    fill=True, 
    alpha=0.5
)
plt.xlabel("Proportion of 'Long' Responses")
plt.ylabel("Density")
plt.title("Effect of Delay on Perceived Duration Estimates")
plt.legend()
plt.savefig("delay_distribution.png")
plt.show()
