# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Function to load and merge all data files
def load_all_data(base_path):
    # Lists to store dataframes
    exp1_data = []
    exp2_data = []
    
    # Process Experiment 1 (Within Subjects) data
    exp1_path = os.path.join(base_path, 'Experiment 1 (Within Subjects)')
    for file in os.listdir(exp1_path):
        if file.endswith('.csv'):
            file_path = os.path.join(exp1_path, file)
            df = pd.read_csv(file_path)
            df['Experiment'] = 'Within Subjects'
            exp1_data.append(df)
    
    # Process Experiment 2 (Between Subjects) data
    exp2_path = os.path.join(base_path, 'Experiment 2 (Between Subjects)')
    for file in os.listdir(exp2_path):
        if file.endswith('.csv'):
            file_path = os.path.join(exp2_path, file)
            df = pd.read_csv(file_path)
            df['Experiment'] = 'Between Subjects'
            exp2_data.append(df)
    
    # Concatenate all dataframes
    all_exp1 = pd.concat(exp1_data, ignore_index=True)
    all_exp2 = pd.concat(exp2_data, ignore_index=True)
    all_data = pd.concat([all_exp1, all_exp2], ignore_index=True)
    
    # Clean column names and convert data types
    all_data.columns = [col.strip() for col in all_data.columns]
    
    # Convert 'Response - Short or Long' to binary
    print(all_data['Response - Short or Long'].value_counts())
    all_data['Response_Binary'] = all_data['Response - Short or Long'].apply(lambda x: 0 if (x.strip() == 'Short' or x.strip() == 'short' ) else 1)
    
    print(all_data['Response_Binary'].value_counts())
    
    # Create a binary column for objective duration (for accuracy analysis)
    # Based on the midpoint between standard durations (250ms and 850ms)
    midpoint = (250 + 850) / 2  # 550ms
    all_data['Objective_Binary'] = (all_data['Objective Duration of Stimulus'] > midpoint).astype(int)
    
    # Create an accuracy column
    all_data['Accurate'] = (all_data['Response_Binary'] == all_data['Objective_Binary']).astype(int)
    
    return all_data

def clean_data(df):
    # Drop rows with NaN values in specific columns
    if 'Unnamed: 13' in df.columns:
        df = df.drop(columns=['Unnamed: 13'])
    df = df.dropna(subset=['Intended or Unintended Outcome', 'Stimulus Delay', 
                           'Reaction Time on Outcome Response', 'Objective Duration of Stimulus'])
    
    # Convert categorical columns to 'category' dtype
    df['Intended or Unintended Outcome'] = df['Intended or Unintended Outcome'].astype('category')
    df['Stimulus Delay'] = df['Stimulus Delay'].astype('category')
    
    # FOR EVERY VALUE and fileds USE .STRpIP() TO REMOVE SPACEs
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df

# Load the data
data_path = './'  # Path to the data directory
df = load_all_data(data_path)
df = clean_data(df)

# print(df) to txt
with open('data_summary.txt', 'w') as f:
    f.write("Data Summary:\n")
    f.write(str(df.describe()))
    f.write("\n\nMissing Values:\n")
    f.write(str(df.isnull().sum()))

# Perform One-way ANOVA on Response Time by Intentionality
oneway_intention = pg.anova(data=df, dv='Reaction Time on Outcome Response', 
                           between='Intended or Unintended Outcome', detailed=True)

# Perform One-way ANOVA on Response Time by Delay
oneway_delay = pg.anova(data=df, dv='Reaction Time on Outcome Response', 
                       between='Stimulus Delay', detailed=True)

# Perform Two-way ANOVA (Factorial) on Response Time 
factorial_rt = pg.anova(data=df, dv='Reaction Time on Outcome Response', 
                       between=['Intended or Unintended Outcome', 'Stimulus Delay'], 
                       detailed=True)

# Perform Two-way ANOVA on Accuracy
factorial_acc = pg.anova(data=df, dv='Accurate', 
                        between=['Intended or Unintended Outcome', 'Stimulus Delay'], 
                        detailed=True)

# Print results
print("One-way ANOVA: Effect of Intentionality on Reaction Time")
print(oneway_intention)
print("\nOne-way ANOVA: Effect of Delay on Reaction Time")
print(oneway_delay)
print("\nTwo-way ANOVA: Interaction between Intentionality and Delay on Reaction Time")
print(factorial_rt)
print("\nTwo-way ANOVA: Interaction between Intentionality and Delay on Accuracy")
print(factorial_acc)


# This code snippet is performing a Generalized Linear Model (GLM) using logistic regression to predict binary response outcomes. Here's a breakdown of what each step is doing:
# Perform Generalized Linear Model (GLM) for binary response outcome
# Create a binary logistic regression model to predict "long" responses
X = df[['Intended or Unintended Outcome', 'Stimulus Delay', 
        'Reaction Time on Outcome Response', 'Objective Duration of Stimulus']]
X = sm.add_constant(X)  # Add intercept
y = df['Response_Binary']

# Ensure all columns in X are numeric and drop rows with NaN or infinite values
X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]  # Ensure y matches the cleaned X

# Check if there are any remaining NaN values in the target variable
if y.isnull().any():
    raise ValueError("Target variable 'y' contains NaN values after cleaning. Please check the data.")

# Fit logistic regression model
glm_model = sm.GLM(y, X, family=sm.families.Binomial())
glm_results = glm_model.fit()

# Print summary
print("Generalized Linear Model (Logistic Regression) Results:")
print(glm_results.summary())

# Calculate odds ratios and confidence intervals
odds_ratios = np.exp(glm_results.params)
conf_int = np.exp(glm_results.conf_int())

# Create a DataFrame with odds ratios and confidence intervals
odds_df = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    '2.5%': conf_int[0],
    '97.5%': conf_int[1]
})

print("\nOdds Ratios and 95% Confidence Intervals:")
print(odds_df)

# Visualize odds ratios with confidence intervals
plt.figure(figsize=(10, 6))
odds_df_plot = odds_df.drop('const')  # Remove intercept for better visualization
odds_df_plot.plot(y='Odds Ratio', yerr=[odds_df_plot['Odds Ratio'] - odds_df_plot['2.5%'], 
                                      odds_df_plot['97.5%'] - odds_df_plot['Odds Ratio']], 
                 kind='bar', capsize=5)
plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
plt.title('Odds Ratios for Predicting "Long" Responses with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('odds_ratios.png', dpi=300)

print(df.columns)
# Regression analysis with interaction effects
# Create interaction terms
df['Intention_Delay_Interaction'] = df['Intended or Unintended Outcome'].cat.codes * df['Stimulus Delay'].cat.codes
df['Intention_Duration_Interaction'] = df['Intended or Unintended Outcome'].cat.codes * df['Objective Duration of Stimulus']

# Fit linear regression model for perceived duration with interaction terms
# Ensure categorical variables are properly encoded
df['Intended or Unintended Outcome'] = df['Intended or Unintended Outcome'].astype('category')
df['Stimulus Delay'] = df['Stimulus Delay'].astype('category')

# Replace spaces in column names with underscores for compatibility
df.columns = df.columns.str.replace(' ', '_')

interaction_formula = ('Response_Binary ~ C(Intended_or_Unintended_Outcome) + '
                     'Stimulus_Delay + Objective_Duration_of_Stimulus + '
                     'Reaction_Time_on_Outcome_Response + '
                     'C(Intended_or_Unintended_Outcome):Stimulus_Delay + '
                     'C(Intended_or_Unintended_Outcome):Objective_Duration_of_Stimulus')

interaction_model = ols(interaction_formula, data=df).fit()
print("Regression Model with Interaction Effects:")
print(interaction_model.summary())

# Create plots to visualize interaction effects
# Interaction between Intentionality and Delay
plt.figure(figsize=(10, 6))
grouped = df.groupby(['Intended_or_Unintended_Outcome', 'Stimulus_Delay'])['Response_Binary'].mean().reset_index()
wide_format = grouped.pivot(index='Stimulus_Delay', 
                          columns='Intended_or_Unintended_Outcome', 
                          values='Response_Binary')

plt.plot(wide_format.index, wide_format[0], 'b-', label='Unintended Outcomes')
plt.plot(wide_format.index, wide_format[1], 'r-', label='Intended Outcomes')
plt.xlabel('Stimulus Delay (ms)')
plt.ylabel('Proportion of "Long" Responses')
plt.title('Interaction between Intentionality and Delay on Perceived Duration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('intention_delay_interaction.png', dpi=300)

# Visualize the interaction between intentionality and objective duration
plt.figure(figsize=(10, 6))
# Create duration bins for better visualization
df['Duration_Bin'] = pd.cut(df['Objective_Duration_of_Stimulus'], 
                          bins=[200, 350, 450, 550, 650, 750, 850], 
                          labels=['250-350', '350-450', '450-550', '550-650', '650-750', '750-850'])

interaction_data = df.groupby(['Intended_or_Unintended_Outcome', 'Duration_Bin'])['Response_Binary'].mean().reset_index()
wide_format = interaction_data.pivot(index='Duration_Bin', 
                                   columns='Intended_or_Unintended_Outcome', 
                                   values='Response_Binary')

plt.plot(range(len(wide_format.index)), wide_format[0], 'b-o', label='Unintended Outcomes')
plt.plot(range(len(wide_format.index)), wide_format[1], 'r-o', label='Intended Outcomes')
plt.xticks(range(len(wide_format.index)), wide_format.index, rotation=45)
plt.xlabel('Objective Duration (ms)')
plt.ylabel('Proportion of "Long" Responses')
plt.title('Interaction between Intentionality and Objective Duration on Perceived Duration')
plt.legend()    
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('intention_duration_interaction.png', dpi=300)

print("uiduhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

# Mixed Effects Modeling
import statsmodels.formula.api as smf

# Create a mixed-effects model with subject as random effect
mixed_model_formula = ('Response_Binary ~ C(Intended_or_Unintended_Outcome) * Stimulus_Delay + '
                     'Objective_Duration_of_Stimulus + Reaction_Time_on_Outcome_Response')

# Extract subject ID from the Subject ID column
df['Subject'] = df['Subject_ID'].astype(str).str.extract(r'(\w+\d+)')[0].fillna('Unknown')

# Fit mixed-effects model
mixed_model = smf.mixedlm(mixed_model_formula, df, groups=df['Subject'])
mixed_model_results = mixed_model.fit()

print("Mixed Effects Model Results:")
print(mixed_model_results.summary())

# Analyze random effects variation across subjects
random_effects = pd.DataFrame({'Subject': mixed_model_results.random_effects.keys(),
                             'Random Effect': [re[0] for re in mixed_model_results.random_effects.values()]})

# Visualize random effects across subjects
plt.figure(figsize=(12, 6))
sns.barplot(x='Subject', y='Random Effect', data=random_effects.sort_values('Random Effect'))
plt.title('Individual Differences in Temporal Perception (Random Effects by Subject)')
plt.xlabel('Subject_ID')
plt.ylabel('Random Effect Size')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('random_effects.png', dpi=300)

# Compute intraclass correlation coefficient (ICC)
subject_var = mixed_model_results.cov_re.iloc[0, 0]
residual_var = mixed_model_results.scale
icc = subject_var / (subject_var + residual_var)

print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.4f}")
print(f"This indicates that {icc*100:.2f}% of the variance in perceived duration is attributable to individual differences between subjects.")



print("uiduh jojwocjhoisaxjc wqdicsdi vucsuivw huivb")

# Normality check for reaction times
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
stats.probplot(df['Reaction_Time_on_Outcome_Response'], plot=plt)
plt.title('Q-Q Plot for Reaction Times')

plt.subplot(1, 2, 2)
sns.histplot(df['Reaction_Time_on_Outcome_Response'], kde=True)
plt.title('Distribution of Reaction Times')
plt.tight_layout()
plt.savefig('normality_check.png', dpi=300)

# Test for normality
k2, p = stats.normaltest(df['Reaction_Time_on_Outcome_Response'])
print(f"D'Agostino-Pearson test for normality: k² = {k2:.2f}, p = {p:.10f}")

# If not normal, try log transformation
df['Log_RT'] = np.log(df['Reaction_Time_on_Outcome_Response'])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
stats.probplot(df['Log_RT'], plot=plt)
plt.title('Q-Q Plot for Log-Transformed Reaction Times')

plt.subplot(1, 2, 2)
sns.histplot(df['Log_RT'], kde=True)
plt.title('Distribution of Log-Transformed Reaction Times')
plt.tight_layout()
plt.savefig('log_transformed_rt.png', dpi=300)

# Test for normality of log-transformed data
k2_log, p_log = stats.normaltest(df['Log_RT'])
print(f"D'Agostino-Pearson test for normality (log-transformed): k² = {k2_log:.2f}, p = {p_log:.10f}")

# Homogeneity of variance check (Levene's test)
levene_intention = stats.levene(
    df[df['Intended_or_Unintended_Outcome'] == 0]['Reaction_Time_on_Outcome_Response'],
    df[df['Intended_or_Unintended_Outcome'] == 1]['Reaction_Time_on_Outcome_Response']
)

levene_delay = stats.levene(
    df[df['Stimulus_Delay'] == 250]['Reaction_Time_on_Outcome_Response'],
    df[df['Stimulus_Delay'] == 1000]['Reaction_Time_on_Outcome_Response']
)

print(f"\nLevene's test for homogeneity of variance (Intentionality): W = {levene_intention[0]:.2f}, p = {levene_intention[1]:.10f}")
print(f"Levene's test for homogeneity of variance (Delay): W = {levene_delay[0]:.2f}, p = {levene_delay[1]:.10f}")

# Check model diagnostics for the regression model
residuals = interaction_model.resid
fitted = interaction_model.fittedvalues

plt.figure(figsize=(12, 10))

# Residuals vs Fitted plot
plt.subplot(2, 2, 1)
plt.scatter(fitted, residuals, alpha=0.1)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# Residuals Q-Q plot
plt.subplot(2, 2, 2)
stats.probplot(residuals, plot=plt)
plt.title('Q-Q Plot of Residuals')

# Scale-Location plot
plt.subplot(2, 2, 3)
plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.1)
plt.xlabel('Fitted values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location Plot')

# Residuals vs Leverage
plt.subplot(2, 2, 4)
influence = interaction_model.get_influence()
leverage = influence.hat_matrix_diag
plt.scatter(leverage, residuals, alpha=0.1)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Leverage')
plt.ylabel('Residuals')
plt.title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig('model_diagnostics.png', dpi=300)

print("opsdjocisdjcio ksaocosdjcop ioj")

# Analysis to address Question 1
# Split data by delay condition
df_250ms = df[df['Stimulus_Delay'] == 250]
df_1000ms = df[df['Stimulus_Delay'] == 1000]

# Compute proportion of "long" responses by intentionality for each delay
prop_long_250ms = df_250ms.groupby('Intended_or_Unintended_Outcome')['Response_Binary'].mean()
prop_long_1000ms = df_1000ms.groupby('Intended_or_Unintended_Outcome')['Response_Binary'].mean()

# T-tests to compare intended vs. unintended for each delay
ttest_250ms = stats.ttest_ind(
    df_250ms[df_250ms['Intended_or_Unintended_Outcome'] == 1]['Response_Binary'],
    df_250ms[df_250ms['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
)

ttest_1000ms = stats.ttest_ind(
    df_1000ms[df_1000ms['Intended_or_Unintended_Outcome'] == 1]['Response_Binary'],
    df_1000ms[df_1000ms['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
)

print("Question 1: Does intentionality influence temporal perception differently for short vs. long delays?")
print("\nProportion of 'long' responses for 250ms delay:")
print(f"Unintended outcomes: {prop_long_250ms[0]:.4f}")
print(f"Intended outcomes: {prop_long_250ms[1]:.4f}")
print(f"Difference: {prop_long_250ms[1] - prop_long_250ms[0]:.4f}")
print(f"T-test: t = {ttest_250ms[0]:.4f}, p = {ttest_250ms[1]:.4f}")

print("\nProportion of 'long' responses for 1000ms delay:")
print(f"Unintended outcomes: {prop_long_1000ms[0]:.4f}")
print(f"Intended outcomes: {prop_long_1000ms[1]:.4f}")
print(f"Difference: {prop_long_1000ms[1] - prop_long_1000ms[0]:.4f}")
print(f"T-test: t = {ttest_1000ms[0]:.4f}, p = {ttest_1000ms[1]:.4f}")

# Visualize the interaction
plt.figure(figsize=(10, 6))
delay_labels = ['250ms', '1000ms']
intended_props = [prop_long_250ms[1], prop_long_1000ms[1]]
unintended_props = [prop_long_250ms[0], prop_long_1000ms[0]]

x = np.arange(len(delay_labels))
width = 0.35

plt.bar(x - width/2, unintended_props, width, label='Unintended_Outcomes')
plt.bar(x + width/2, intended_props, width, label='Intended_Outcomes')

plt.xlabel('Action-Outcome Delay')
plt.ylabel('Proportion of "Long" Responses')
plt.title('Effect of Intentionality on Perceived Duration by Delay Condition')
plt.xticks(x, delay_labels)
plt.legend()

# Add p-values to the plot
plt.text(0, max(intended_props[0], unintended_props[0]) + 0.02, 
         f"p = {ttest_250ms[1]:.4f}", ha='center')
plt.text(1, max(intended_props[1], unintended_props[1]) + 0.02, 
         f"p = {ttest_1000ms[1]:.4f}", ha='center')

plt.tight_layout()
plt.savefig('intentionality_by_delay.png', dpi=300)

# Compute effect sizes (Cohen's d)
from scipy import stats

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / s_pooled
    return d

d_250ms = cohens_d(
    df_250ms[df_250ms['Intended_or_Unintended_Outcome'] == 1]['Response_Binary'],
    df_250ms[df_250ms['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
)

d_1000ms = cohens_d(
    df_1000ms[df_1000ms['Intended_or_Unintended_Outcome'] == 1]['Response_Binary'],
    df_1000ms[df_1000ms['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
)

print(f"\nEffect size (Cohen's d) for 250ms delay: {d_250ms:.4f}")
print(f"Effect size (Cohen's d) for 1000ms delay: {d_1000ms:.4f}")












# Question 2: How does the type of stimulus (intended vs. unintended) affect duration estimates?
print("\nQuestion 2: How does the type of stimulus (intended vs. unintended) affect duration estimates?")

# Group by stimulus duration and intention
stimulus_analysis = df.groupby(['Objective_Duration_of_Stimulus', 'Intended_or_Unintended_Outcome'])['Response_Binary'].agg(['mean', 'count', 'std']).reset_index()

# Reshape for easier comparison
wide_stimulus = stimulus_analysis.pivot(index='Objective_Duration_of_Stimulus', 
                                      columns='Intended_or_Unintended_Outcome',
                                      values='mean')
wide_stimulus.columns = ['Unintended', 'Intended']
wide_stimulus['Difference'] = wide_stimulus['Intended'] - wide_stimulus['Unintended']

print("Proportion of 'long' responses by objective duration and intentionality:")
print(wide_stimulus)

# Visualize the effect of stimulus type by objective duration
plt.figure(figsize=(12, 6))
plt.plot(wide_stimulus.index, wide_stimulus['Intended'], 'ro-', label='Intended Outcomes')
plt.plot(wide_stimulus.index, wide_stimulus['Unintended'], 'bo-', label='Unintended Outcomes')
plt.xlabel('Objective Duration (ms)')
plt.ylabel('Proportion of "Long" Responses')
plt.title('Effect of Stimulus Type on Duration Estimates by Objective Duration')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Indifference Point')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stimulus_type_effect.png', dpi=300)

# Calculate the Point of Subjective Equality (PSE) for each intentionality condition
# We'll use logistic regression to model the psychometric function
from sklearn.linear_model import LogisticRegression

def calculate_pse(df, condition_value):
    # Filter data for the condition
    condition_data = df[df['Intended_or_Unintended_Outcome'] == condition_value]
    
    # Prepare X and y
    X = condition_data[['Objective_Duration_of_Stimulus']]
    y = condition_data['Response_Binary']
    
    # Fit logistic regression
    logit = LogisticRegression()
    logit.fit(X, y)
    
    # Calculate PSE (where p(long) = 0.5)
    pse = (np.log(1/0.5 - 1) - logit.intercept_[0]) / logit.coef_[0][0]
    
    return pse.item()

# Calculate PSE for intended and unintended conditions
pse_intended = calculate_pse(df, 1)
pse_unintended = calculate_pse(df, 0)

print(f"\nPoint of Subjective Equality (PSE):")
print(f"Intended outcomes: {pse_intended:.2f} ms")
print(f"Unintended outcomes: {pse_unintended:.2f} ms")
print(f"Difference (Temporal Expansion): {pse_unintended - pse_intended:.2f} ms")

# Plot psychometric functions
plt.figure(figsize=(10, 6))
durations = np.linspace(200, 900, 100)

# For intended outcomes
X_intended = df[df['Intended_or_Unintended_Outcome'] == 1][['Objective_Duration_of_Stimulus']]
y_intended = df[df['Intended_or_Unintended_Outcome'] == 1]['Response_Binary']
logit_intended = LogisticRegression().fit(X_intended, y_intended)
p_long_intended = logit_intended.predict_proba(durations.reshape(-1, 1))[:, 1]

# For unintended outcomes
X_unintended = df[df['Intended_or_Unintended_Outcome'] == 0][['Objective_Duration_of_Stimulus']]
y_unintended = df[df['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
logit_unintended = LogisticRegression().fit(X_unintended, y_unintended)
p_long_unintended = logit_unintended.predict_proba(durations.reshape(-1, 1))[:, 1]

plt.plot(durations, p_long_intended, 'r-', label='Intended Outcomes')
plt.plot(durations, p_long_unintended, 'b-', label='Unintended Outcomes')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='PSE Threshold')
plt.axvline(x=pse_intended, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=pse_unintended, color='b', linestyle='--', alpha=0.5)
plt.xlabel('Objective Duration (ms)')
plt.ylabel('Proportion of "Long" Responses')
plt.title('Psychometric Functions for Intended vs. Unintended Outcomes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('psychometric_functions.png', dpi=300)

# Question 3: What role does causality play in temporal binding?
print("\nQuestion 3: What role does causality play in temporal binding?")

# Create a causality strength proxy based on reaction time
# Faster reaction times might indicate stronger causal beliefs
df['Causality_Strength'] = 1 / (df['Reaction_Time_on_Outcome_Response'] / 1000)  # Inverse RT in seconds

# Bin causality strength for visualization
df['Causality_Bin'] = pd.qcut(df['Causality_Strength'], 5, labels=False)

# Analyze temporal expansion effect by causality strength
causality_analysis = df.groupby(['Causality_Bin', 'Intended_or_Unintended_Outcome'])['Response_Binary'].mean().reset_index()
causality_wide = causality_analysis.pivot(index='Causality_Bin', 
                                       columns='Intended_or_Unintended_Outcome', 
                                       values='Response_Binary')
causality_wide.columns = ['Unintended', 'Intended']
causality_wide['Difference'] = causality_wide['Intended'] - causality_wide['Unintended']

print("Temporal expansion effect by causality strength quintile:")
print(causality_wide)

# Visualize the relationship between causality strength and temporal expansion
plt.figure(figsize=(10, 6))
plt.bar(causality_wide.index, causality_wide['Difference'])
plt.xlabel('Causality Strength Quintile (1=Weakest, 5=Strongest)')
plt.ylabel('Temporal Expansion Effect (Intended - Unintended)')
plt.title('Relationship Between Causality Strength and Temporal Expansion')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('causality_effect.png', dpi=300)

# Regression analysis of causality effect
causality_model = ols('Response_Binary ~ C(Intended_or_Unintended_Outcome) * Causality_Strength + Objective_Duration_of_Stimulus', data=df).fit()
print("\nRegression analysis of causality effect:")
print(causality_model.summary().tables[1])

# Question 4: Are there gender or age-related differences in temporal perception?
print("\nQuestion 4: Are there gender or age-related differences in temporal perception?")

# Gender analysis
gender_analysis = df.groupby(['Gender', 'Intended_or_Unintended_Outcome'])['Response_Binary'].agg(['mean', 'count']).reset_index()
gender_wide = gender_analysis.pivot(index='Gender', 
                                  columns='Intended_or_Unintended_Outcome', 
                                  values='mean')
gender_wide.columns = ['Unintended', 'Intended']
gender_wide['Difference'] = gender_wide['Intended'] - gender_wide['Unintended']

print("Gender differences in temporal expansion effect:")
print(gender_wide)

# Visualization of gender differences
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(gender_wide.index))

plt.bar(x - bar_width/2, gender_wide['Unintended'], bar_width, label='Unintended Outcomes')
plt.bar(x + bar_width/2, gender_wide['Intended'], bar_width, label='Intended Outcomes')
plt.xlabel('Gender')
plt.ylabel('Proportion of "Long" Responses')
plt.title('Gender Differences in Temporal Perception')
plt.xticks(x, gender_wide.index)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gender_differences.png', dpi=300)

# Statistical test for gender differences
male_effect = gender_wide.loc['M', 'Difference']
female_effect = gender_wide.loc['F', 'Difference']
print(f"\nTemporal expansion effect (Intended - Unintended):")
print(f"Male participants: {male_effect:.4f}")
print(f"Female participants: {female_effect:.4f}")

# Age analysis
# Extract age information if available
if 'Subject_Age' in df.columns:
    # Create age groups
    df['Age_Group'] = pd.cut(df['Subject_Age'], bins=[18, 25, 35, 100], labels=['18-25', '26-35', '36+'])
    # Analyze by age group
    age_analysis = df.groupby(['Age_Group', 'Intended_or_Unintended_Outcome'])['Response_Binary'].mean().reset_index()
    age_wide = age_analysis.pivot(index='Age_Group', 
                                columns='Intended_or_Unintended_Outcome', 
                                values='Response_Binary')
    age_wide.columns = ['Unintended', 'Intended']
    age_wide['Difference'] = age_wide['Intended'] - age_wide['Unintended']
    
    print("\nAge-related differences in temporal expansion effect:")
    print(age_wide)
    
    # Visualization of age differences
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(age_wide.index))
    
    plt.bar(x - bar_width/2, age_wide['Unintended'], bar_width, label='Unintended Outcomes')
    plt.bar(x + bar_width/2, age_wide['Intended'], bar_width, label='Intended Outcomes')
    plt.xlabel('Age Group')
    plt.ylabel('Proportion of "Long" Responses')
    plt.title('Age-Related Differences in Temporal Perception')
    plt.xticks(x, age_wide.index)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('age_differences.png', dpi=300)
else:
    print("\nAge information not available in the dataset for analysis.")

# Question 5: How does reaction time correlate with duration estimates?
print("\nQuestion 5: How does reaction time correlate with duration estimates?")

# Calculate correlation between reaction time and duration estimates
correlation = df.groupby('Subject')[['Reaction_Time_on_Outcome_Response', 'Response_Binary']].corr().iloc[0::2, 1].reset_index()
if 'Reaction_Time_on_Outcome_Response' in df.columns:
    try:
        correlation = df.groupby('Subject')[['Reaction_Time_on_Outcome_Response', 'Response_Binary']].corr().iloc[0::2, 1].reset_index()
        if 'level_1' in correlation.columns and 'Reaction_Time_on_Outcome_Response' in correlation.columns:
            correlation = correlation[correlation['level_1'] == 'Response_Binary'][['Subject', 'Reaction_Time_on_Outcome_Response']]
            correlation.columns = ['Subject', 'Correlation']
        else:
            print("Required columns are missing in the correlation result. Please check the DataFrame structure.")
            correlation = pd.DataFrame(columns=['Subject', 'Correlation'])  # Return an empty DataFrame
    except Exception as e:
        print(f"An error occurred while calculating correlation: {e}")
        correlation = pd.DataFrame(columns=['Subject', 'Correlation'])  # Return an empty DataFrame
else:
    print("The column 'Reaction_Time_on_Outcome_Response' is not in the DataFrame. Please check the DataFrame structure.")
    correlation = pd.DataFrame(columns=['Subject', 'Correlation'])  # Return an empty DataFrame

print("Correlation between reaction time and duration estimates by subject:")
print(correlation)
print(correlation.describe())

# Visualize the distribution of correlations
plt.figure(figsize=(10, 6))
plt.hist(correlation['Correlation'], bins=15, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.title('Distribution of Correlations Between Reaction Time and Duration Estimates')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reaction_time_correlation.png', dpi=300)

# Test if the mean correlation is significantly different from zero
t_stat, p_val = stats.ttest_1samp(correlation['Correlation'], 0)
print(f"\nOne-sample t-test on correlations: t = {t_stat:.4f}, p = {p_val:.4f}")

# Examine how reaction time predicts duration estimates for intended vs. unintended outcomes
reaction_time_model = ols('Response_Binary ~ C(Intended_or_Unintended_Outcome) * Reaction_Time_on_Outcome_Response + Objective_Duration_of_Stimulus', data=df).fit()
print("\nRegression analysis of reaction time effect:")
print(reaction_time_model.summary().tables[1])

# Question 6: Does the experimental design (within-subjects vs. between-subjects) influence results?
print("\nQuestion 6: Does the experimental design (within-subjects vs. between-subjects) influence results?")

# Separate data by experiment type
within_subjects = df[df['Experiment'] == 'Within Subjects']
between_subjects = df[df['Experiment'] == 'Between Subjects']

# Function to analyze experiment type effect
def analyze_by_experiment_design(data, design_name):
    # Split by intention
    intended = data[data['Intended_or_Unintended_Outcome'] == 1]['Response_Binary']
    unintended = data[data['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
    
    # Compute proportions and run t-test
    prop_intended = np.mean(intended)
    prop_unintended = np.mean(unintended)
    t_stat, p_val = stats.ttest_ind(intended, unintended)
    
    # Effect size
    d = cohens_d(intended, unintended)
    
    return {
        'design': design_name,
        'prop_intended': prop_intended,
        'prop_unintended': prop_unintended,
        'difference': prop_intended - prop_unintended,
        't_stat': t_stat,
        'p_value': p_val,
        'cohens_d': d
    }

# Analyze both experimental designs
within_results = analyze_by_experiment_design(within_subjects, 'Within-Subjects')
between_results = analyze_by_experiment_design(between_subjects, 'Between-Subjects')

# Print results
print("\nWithin-Subjects Design:")
print(f"Proportion of 'long' responses for intended outcomes: {within_results['prop_intended']:.4f}")
print(f"Proportion of 'long' responses for unintended outcomes: {within_results['prop_unintended']:.4f}")
print(f"Difference (temporal expansion): {within_results['difference']:.4f}")
print(f"t-test: t = {within_results['t_stat']:.4f}, p = {within_results['p_value']:.4f}")
print(f"Effect size (Cohen's d): {within_results['cohens_d']:.4f}")

print("\nBetween-Subjects Design:")
print(f"Proportion of 'long' responses for intended outcomes: {between_results['prop_intended']:.4f}")
print(f"Proportion of 'long' responses for unintended outcomes: {between_results['prop_unintended']:.4f}")
print(f"Difference (temporal expansion): {between_results['difference']:.4f}")
print(f"t-test: t = {between_results['t_stat']:.4f}, p = {between_results['p_value']:.4f}")
print(f"Effect size (Cohen's d): {between_results['cohens_d']:.4f}")

# Visualization of experimental design effects
plt.figure(figsize=(10, 6))
labels = ['Within-Subjects', 'Between-Subjects']
intended_props = [within_results['prop_intended'], between_results['prop_intended']]
unintended_props = [within_results['prop_unintended'], between_results['prop_unintended']]
differences = [within_results['difference'], between_results['difference']]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot bars for intended and unintended proportions
ax1.bar(x - width/2, unintended_props, width, label='Unintended Outcomes', color='blue', alpha=0.7)
ax1.bar(x + width/2, intended_props, width, label='Intended Outcomes', color='red', alpha=0.7)
ax1.set_ylabel('Proportion of "Long" Responses')
ax1.set_ylim(0, 0.7)

# Add a second y-axis for the difference
ax2 = ax1.twinx()
ax2.plot(x, differences, 'ko-', label='Difference (Effect Size)')
ax2.set_ylabel('Temporal Expansion Effect')
ax2.set_ylim(0, 0.1)

# Add labels and legend
ax1.set_xlabel('Experimental Design')
ax1.set_title('Effect of Experimental Design on Temporal Expansion')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add p-values and effect sizes
for i, design in enumerate([within_results, between_results]):
    plt.text(i, 0.02, f"p = {design['p_value']:.4f}\nd = {design['cohens_d']:.4f}", 
             ha='center', va='bottom', color='black')

plt.tight_layout()
plt.savefig('experimental_design_effect.png', dpi=300)



# # Continuing with the analysis of the remaining questions...

# # Question 2: How does the type of stimulus (intended vs. unintended) affect duration estimates?
# print("\nQuestion 2: How does the type of stimulus (intended vs. unintended) affect duration estimates?")

# # Group by stimulus duration and intention
# stimulus_analysis = df.groupby(['Objective_Duration_of_Stimulus', 'Intended_or_Unintended_Outcome'])['Response_Binary'].agg(['mean', 'count', 'std']).reset_index()

# # Reshape for easier comparison
# wide_stimulus = stimulus_analysis.pivot(index='Objective_Duration_of_Stimulus', 
#                                       columns='Intended_or_Unintended_Outcome',
#                                       values='mean')
# wide_stimulus.columns = ['Unintended', 'Intended']
# wide_stimulus['Difference'] = wide_stimulus['Intended'] - wide_stimulus['Unintended']

# print("Proportion of 'long' responses by objective duration and intentionality:")
# print(wide_stimulus)

# # Visualize the effect of stimulus type by objective duration
# plt.figure(figsize=(12, 6))
# plt.plot(wide_stimulus.index, wide_stimulus['Intended'], 'ro-', label='Intended Outcomes')
# plt.plot(wide_stimulus.index, wide_stimulus['Unintended'], 'bo-', label='Unintended Outcomes')
# plt.xlabel('Objective Duration (ms)')
# plt.ylabel('Proportion of "Long" Responses')
# plt.title('Effect of Stimulus Type on Duration Estimates by Objective Duration')
# plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Indifference Point')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('stimulus_type_effect.png', dpi=300)

# # Calculate the Point of Subjective Equality (PSE) for each intentionality condition
# # We'll use logistic regression to model the psychometric function
# from sklearn.linear_model import LogisticRegression

# def calculate_pse(df, condition_value):
#     # Filter data for the condition
#     condition_data = df[df['Intended_or_Unintended_Outcome'] == condition_value]
    
#     # Prepare X and y
#     X = condition_data[['Objective_Duration_of_Stimulus']]
#     y = condition_data['Response_Binary']
    
#     # Fit logistic regression
#     logit = LogisticRegression()
#     logit.fit(X, y)
    
#     # Calculate PSE (where p(long) = 0.5)
#     pse = (np.log(1/0.5 - 1) - logit.intercept_[0]) / logit.coef_[0][0]
    
#     return pse.item()

# # Calculate PSE for intended and unintended conditions
# pse_intended = calculate_pse(df, 1)
# pse_unintended = calculate_pse(df, 0)

# print(f"\nPoint of Subjective Equality (PSE):")
# print(f"Intended outcomes: {pse_intended:.2f} ms")
# print(f"Unintended outcomes: {pse_unintended:.2f} ms")
# print(f"Difference (Temporal Expansion): {pse_unintended - pse_intended:.2f} ms")

# # Plot psychometric functions
# plt.figure(figsize=(10, 6))
# durations = np.linspace(200, 900, 100)

# # For intended outcomes
# X_intended = df[df['Intended_or_Unintended_Outcome'] == 1][['Objective_Duration_of_Stimulus']]
# y_intended = df[df['Intended_or_Unintended_Outcome'] == 1]['Response_Binary']
# logit_intended = LogisticRegression().fit(X_intended, y_intended)
# p_long_intended = logit_intended.predict_proba(durations.reshape(-1, 1))[:, 1]

# # For unintended outcomes
# X_unintended = df[df['Intended_or_Unintended_Outcome'] == 0][['Objective_Duration_of_Stimulus']]
# y_unintended = df[df['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
# logit_unintended = LogisticRegression().fit(X_unintended, y_unintended)
# p_long_unintended = logit_unintended.predict_proba(durations.reshape(-1, 1))[:, 1]

# plt.plot(durations, p_long_intended, 'r-', label='Intended Outcomes')
# plt.plot(durations, p_long_unintended, 'b-', label='Unintended Outcomes')
# plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='PSE Threshold')
# plt.axvline(x=pse_intended, color='r', linestyle='--', alpha=0.5)
# plt.axvline(x=pse_unintended, color='b', linestyle='--', alpha=0.5)
# plt.xlabel('Objective Duration (ms)')
# plt.ylabel('Proportion of "Long" Responses')
# plt.title('Psychometric Functions for Intended vs. Unintended Outcomes')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('psychometric_functions.png', dpi=300)

# # Question 3: What role does causality play in temporal binding?
# print("\nQuestion 3: What role does causality play in temporal binding?")

# # Create a causality strength proxy based on reaction time
# # Faster reaction times might indicate stronger causal beliefs
# df['Causality_Strength'] = 1 / (df['Reaction_Time_on_Outcome_Response'] / 1000)  # Inverse RT in seconds

# # Bin causality strength for visualization
# df['Causality_Bin'] = pd.qcut(df['Causality_Strength'], 5, labels=False)

# # Analyze temporal expansion effect by causality strength
# causality_analysis = df.groupby(['Causality_Bin', 'Intended_or_Unintended_Outcome'])['Response_Binary'].mean().reset_index()
# causality_wide = causality_analysis.pivot(index='Causality_Bin', 
#                                        columns='Intended_or_Unintended_Outcome', 
#                                        values='Response_Binary')
# causality_wide.columns = ['Unintended', 'Intended']
# causality_wide['Difference'] = causality_wide['Intended'] - causality_wide['Unintended']

# print("Temporal expansion effect by causality strength quintile:")
# print(causality_wide)

# # Visualize the relationship between causality strength and temporal expansion
# plt.figure(figsize=(10, 6))
# plt.bar(causality_wide.index, causality_wide['Difference'])
# plt.xlabel('Causality Strength Quintile (1=Weakest, 5=Strongest)')
# plt.ylabel('Temporal Expansion Effect (Intended - Unintended)')
# plt.title('Relationship Between Causality Strength and Temporal Expansion')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('causality_effect.png', dpi=300)

# # Regression analysis of causality effect
# causality_model = ols('Response_Binary ~ C(Intended_or_Unintended_Outcome) * Causality_Strength + Objective_Duration_of_Stimulus', data=df).fit()
# print("\nRegression analysis of causality effect:")
# print(causality_model.summary().tables[1])

# # Question 4: Are there gender or age-related differences in temporal perception?
# print("\nQuestion 4: Are there gender or age-related differences in temporal perception?")

# # Gender analysis
# gender_analysis = df.groupby(['Gender', 'Intended_or_Unintended_Outcome'])['Response_Binary'].agg(['mean', 'count']).reset_index()
# gender_wide = gender_analysis.pivot(index='Gender', 
#                                   columns='Intended_or_Unintended_Outcome', 
#                                   values='mean')
# gender_wide.columns = ['Unintended', 'Intended']
# gender_wide['Difference'] = gender_wide['Intended'] - gender_wide['Unintended']

# print("Gender differences in temporal expansion effect:")
# print(gender_wide)

# # Visualization of gender differences
# plt.figure(figsize=(10, 6))
# bar_width = 0.35
# x = np.arange(len(gender_wide.index))

# plt.bar(x - bar_width/2, gender_wide['Unintended'], bar_width, label='Unintended Outcomes')
# plt.bar(x + bar_width/2, gender_wide['Intended'], bar_width, label='Intended Outcomes')
# plt.xlabel('Gender')
# plt.ylabel('Proportion of "Long" Responses')
# plt.title('Gender Differences in Temporal Perception')
# plt.xticks(x, gender_wide.index)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('gender_differences.png', dpi=300)

# # Statistical test for gender differences
# male_effect = gender_wide.loc['M', 'Difference']
# female_effect = gender_wide.loc['F', 'Difference']
# print(f"\nTemporal expansion effect (Intended - Unintended):")
# print(f"Male participants: {male_effect:.4f}")
# print(f"Female participants: {female_effect:.4f}")

# # Age analysis
# # Extract age information if available
# if 'Subject Age' in df.columns:
#     # Create age groups
#     df['Age_Group'] = pd.cut(df['Subject Age'], bins=[18, 25, 35, 100], labels=['18-25', '26-35', '36+'])
    
#     # Analyze by age group
#     age_analysis = df.groupby(['Age_Group', 'Intended_or_Unintended_Outcome'])['Response_Binary'].mean().reset_index()
#     age_wide = age_analysis.pivot(index='Age_Group', 
#                                 columns='Intended_or_Unintended_Outcome', 
#                                 values='Response_Binary')
#     age_wide.columns = ['Unintended', 'Intended']
#     age_wide['Difference'] = age_wide['Intended'] - age_wide['Unintended']
    
#     print("\nAge-related differences in temporal expansion effect:")
#     print(age_wide)
    
#     # Visualization of age differences
#     plt.figure(figsize=(10, 6))
#     bar_width = 0.35
#     x = np.arange(len(age_wide.index))
    
#     plt.bar(x - bar_width/2, age_wide['Unintended'], bar_width, label='Unintended Outcomes')
#     plt.bar(x + bar_width/2, age_wide['Intended'], bar_width, label='Intended Outcomes')
#     plt.xlabel('Age Group')
#     plt.ylabel('Proportion of "Long" Responses')
#     plt.title('Age-Related Differences in Temporal Perception')
#     plt.xticks(x, age_wide.index)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('age_differences.png', dpi=300)
# else:
#     print("\nAge information not available in the dataset for analysis.")

# # Question 5: How does reaction time correlate with duration estimates?
# print("\nQuestion 5: How does reaction time correlate with duration estimates?")

# # Calculate correlation between reaction time and duration estimates
# correlation = df.groupby('Subject')[['Reaction_Time_on_Outcome_Response', 'Response_Binary']].corr().iloc[0::2, 1].reset_index()
# correlation = correlation[correlation['level_1'] == 'Response_Binary'][['Subject', 'Reaction_Time_on_Outcome_Response']]
# correlation.columns = ['Subject', 'Correlation']

# print("Correlation between reaction time and duration estimates by subject:")
# print(correlation.describe())

# # Visualize the distribution of correlations
# plt.figure(figsize=(10, 6))
# plt.hist(correlation['Correlation'], bins=15, edgecolor='black')
# plt.axvline(x=0, color='red', linestyle='--')
# plt.xlabel('Correlation Coefficient')
# plt.ylabel('Frequency')
# plt.title('Distribution of Correlations Between Reaction Time and Duration Estimates')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('reaction_time_correlation.png', dpi=300)

# # Test if the mean correlation is significantly different from zero
# t_stat, p_val = stats.ttest_1samp(correlation['Correlation'], 0)
# print(f"\nOne-sample t-test on correlations: t = {t_stat:.4f}, p = {p_val:.4f}")

# # Examine how reaction time predicts duration estimates for intended vs. unintended outcomes
# reaction_time_model = ols('Response_Binary ~ C(Intended_or_Unintended_Outcome) * Reaction_Time_on_Outcome_Response + Objective_Duration_of_Stimulus', data=df).fit()
# print("\nRegression analysis of reaction time effect:")
# print(reaction_time_model.summary().tables[1])

# Question 6: Does the experimental design (within-subjects vs. between-subjects) influence results?
print("\nQuestion 6: Does the experimental design (within-subjects vs. between-subjects) influence results?")

# Separate data by experiment type
within_subjects = df[df['Experiment'] == 'Within Subjects']
between_subjects = df[df['Experiment'] == 'Between Subjects']

# Function to analyze experiment type effect
def analyze_by_experiment_design(data, design_name):
    # Split by intention
    intended = data[data['Intended_or_Unintended_Outcome'] == 1]['Response_Binary']
    unintended = data[data['Intended_or_Unintended_Outcome'] == 0]['Response_Binary']
    
    # Compute proportions and run t-test
    prop_intended = np.mean(intended)
    prop_unintended = np.mean(unintended)
    t_stat, p_val = stats.ttest_ind(intended, unintended)
    
    # Effect size
    d = cohens_d(intended, unintended)
    
    return {
        'design': design_name,
        'prop_intended': prop_intended,
        'prop_unintended': prop_unintended,
        'difference': prop_intended - prop_unintended,
        't_stat': t_stat,
        'p_value': p_val,
        'cohens_d': d
    }

# Analyze both experimental designs
within_results = analyze_by_experiment_design(within_subjects, 'Within-Subjects')
between_results = analyze_by_experiment_design(between_subjects, 'Between-Subjects')

# Print results
print("\nWithin-Subjects Design:")
print(f"Proportion of 'long' responses for intended outcomes: {within_results['prop_intended']:.4f}")
print(f"Proportion of 'long' responses for unintended outcomes: {within_results['prop_unintended']:.4f}")
print(f"Difference (temporal expansion): {within_results['difference']:.4f}")
print(f"t-test: t = {within_results['t_stat']:.4f}, p = {within_results['p_value']:.4f}")
print(f"Effect size (Cohen's d): {within_results['cohens_d']:.4f}")

print("\nBetween-Subjects Design:")
print(f"Proportion of 'long' responses for intended outcomes: {between_results['prop_intended']:.4f}")
print(f"Proportion of 'long' responses for unintended outcomes: {between_results['prop_unintended']:.4f}")
print(f"Difference (temporal expansion): {between_results['difference']:.4f}")
print(f"t-test: t = {between_results['t_stat']:.4f}, p = {between_results['p_value']:.4f}")
print(f"Effect size (Cohen's d): {between_results['cohens_d']:.4f}")

# Visualization of experimental design effects
plt.figure(figsize=(10, 6))
labels = ['Within-Subjects', 'Between-Subjects']
intended_props = [within_results['prop_intended'], between_results['prop_intended']]
unintended_props = [within_results['prop_unintended'], between_results['prop_unintended']]
differences = [within_results['difference'], between_results['difference']]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot bars for intended and unintended proportions
ax1.bar(x - width/2, unintended_props, width, label='Unintended Outcomes', color='blue', alpha=0.7)
ax1.bar(x + width/2, intended_props, width, label='Intended Outcomes', color='red', alpha=0.7)
ax1.set_ylabel('Proportion of "Long" Responses')
ax1.set_ylim(0, 0.7)

# Add a second y-axis for the difference
ax2 = ax1.twinx()
ax2.plot(x, differences, 'ko-', label='Difference (Effect Size)')
ax2.set_ylabel('Temporal Expansion Effect')
ax2.set_ylim(0, 0.1)

# Add labels and legend
ax1.set_xlabel('Experimental Design')
ax1.set_title('Effect of Experimental Design on Temporal Expansion')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add p-values and effect sizes
for i, design in enumerate([within_results, between_results]):
    plt.text(i, 0.02, f"p = {design['p_value']:.4f}\nd = {design['cohens_d']:.4f}", 
             ha='center', va='bottom', color='black')

plt.tight_layout()
plt.savefig('experimental_design_effect.png', dpi=300)

# Additional analysis: Mixed-model comparing the two experimental designs
experiment_model = ols('Response_Binary ~ C(Intended_or_Unintended_Outcome) * C(Experiment) + Objective_Duration_of_Stimulus', data=df).fit()
print("\nInteraction between intentionality and experimental design:")
print(experiment_model.summary().tables[1])

# Additional analysis: Interaction between delay and experimental design
interaction_model = ols('Response_Binary ~ C(Intended_or_Unintended_Outcome) * C(Experiment) * C(Stimulus_Delay) + Objective_Duration_of_Stimulus', data=df).fit()
print("\nThree-way interaction between intentionality, experimental design, and delay:")
print(interaction_model.summary().tables[1])
