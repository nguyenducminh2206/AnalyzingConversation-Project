import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, kstest, anderson
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv('projectdata.csv')

# Part 1
print('Part 1')
print()

newsgroup_names = {
    1: 'alt.atheism',
    2: 'comp.graphics',
    3: 'comp.os.ms-windows.misc',
    4: 'comp.sys.ibm.pc.hardware',
    5: 'comp.sys.mac.hardware',
    6: 'comp.windows.x',
    7: 'misc.forsale',
    8: 'rec.autos',
    9: 'rec.motorcycles',
    10: 'rec.sport.baseball',
    11: 'rec.sport.hockey',
    12: 'sci.crypt',
    13: 'sci.electronics',
    14: 'sci.med',
    15: 'sci.space',
    16: 'soc.religion.christian',
    17: 'talk.politics.guns',
    18: 'talk.politics.mideast',
    19: 'talk.politics.misc',
    20: 'talk.religion.misc'
}

word_groups = {
    'freedom': [12, 17, 18, 19],
    'nation': [17, 18, 19],
    'logic': [1, 13, 19, 20],
    'normal': [2, 6, 13, 14],
    'program': [2, 6, 19, 5]
}

results = {}

def analyze_word(data, word, groups):
    stats = {}
    
    for group in groups:
        group_name = newsgroup_names[group]
        group_data = data[data['groupID'] == group][word]
        mean = group_data.mean()
        median = group_data.median()
        std_dev = group_data.std()
        quantiles = group_data.quantile([0.001, 0.999])
        stats[group_name] = {
            'mean': round(mean, 2),
            'median': median,
            'std_dev': round(std_dev, 2),
            '0.1% quantile': round(quantiles.iloc[0], 2),
            '99.9% quantile': round(quantiles.iloc[1], 2)
        }
        
        # Plot histogram
        plt.hist(group_data, bins=50, alpha=0.6, label=group_name)
    
    plt.title(f'Histogram of occurrences for word: {word}')
    plt.xlabel('Occurrences')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return stats

# Perform analysis for each word
for word, groups in word_groups.items():
    results[word] = analyze_word(data, word, groups)

# Print results
for word, stats in results.items():
    print(f"\nStatistics for the word '{word}':")
    for group, values in stats.items():
        print(f"\nNewsgroup: {group}")
        for stat, value in values.items():
            print(f"{stat}: {value}")
print()

# Part 2
print('Part 2')
print()

# Calculate the length of each message
word_columns = data.columns[5:]  # Assuming the first 5 columns are not word counts
data['message_length'] = data[word_columns].sum(axis=1)

# Define newsgroups for analysis
newsgroups = {
    'rec.sport.baseball': 10,
    'rec.sport.hockey': 11,
    'rec.autos': 8,
    'rec.motorcycles': 9
}

# Plot histograms of message lengths
def plot_message_lengths(newsgroup_ids, titles):
    plt.figure(figsize=(14, 6))
    for i, (group, title) in enumerate(zip(newsgroup_ids, titles)):
        group_data = data[data['groupID'] == group]['message_length']
        plt.subplot(1, 2, i+1)
        plt.hist(group_data, bins=50, alpha=0.7)
        plt.title(f'Message Lengths in {title}')
        plt.xlabel('Message Length')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot histograms of the logarithms of message lengths
def plot_log_message_lengths(newsgroup_ids, titles):
    plt.figure(figsize=(14, 6))
    for i, (group, title) in enumerate(zip(newsgroup_ids, titles)):
        group_data = data[data['groupID'] == group]['message_length']
        log_group_data = np.log(group_data)
        plt.subplot(1, 2, i+1)
        plt.hist(log_group_data, bins=50, alpha=0.7)
        plt.title(f'Logarithm of Message Lengths in {title}')
        plt.xlabel('Log(Message Length)')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Perform t-test on log message lengths between two newsgroups
def t_test_log_message_lengths(group1, group2):
    data1 = np.log(data[data['groupID'] == group1]['message_length'])
    data2 = np.log(data[data['groupID'] == group2]['message_length'])
    t_stat, p_val = ttest_ind(data1, data2, equal_var=False)  # Perform Welch's t-test
    return t_stat, p_val

# Plot histograms for rec.sport.baseball and rec.sport.hockey
plot_message_lengths([newsgroups['rec.sport.baseball'], newsgroups['rec.sport.hockey']], ['rec.sport.baseball', 'rec.sport.hockey'])
plot_log_message_lengths([newsgroups['rec.sport.baseball'], newsgroups['rec.sport.hockey']], ['rec.sport.baseball', 'rec.sport.hockey'])

# T-test for rec.sport.baseball and rec.sport.hockey
t_stat, p_val = t_test_log_message_lengths(newsgroups['rec.sport.baseball'], newsgroups['rec.sport.hockey'])
print(f'T-test between rec.sport.baseball and rec.sport.hockey: t-statistic={t_stat:.4f}, p-value={p_val:.4f}')

# Plot histograms for rec.autos and rec.motorcycles
plot_message_lengths([newsgroups['rec.autos'], newsgroups['rec.motorcycles']], ['rec.autos', 'rec.motorcycles'])
plot_log_message_lengths([newsgroups['rec.autos'], newsgroups['rec.motorcycles']], ['rec.autos', 'rec.motorcycles'])

# T-test for rec.autos and rec.motorcycles
t_stat, p_val = t_test_log_message_lengths(newsgroups['rec.autos'], newsgroups['rec.motorcycles'])
print(f'T-test between rec.autos and rec.motorcycles: t-statistic={t_stat:.4f}, p-value={p_val:.4f}')

print()

# Part 3
print('Part 3')
print()

# Plot histogram of secsfrommidnight
plt.figure(figsize=(12, 6))
plt.hist(data['secsfrommidnight'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Histogram of Posting Times (secsfrommidnight)')
plt.xlabel('Seconds from Midnight')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Transform secsfrommidnight to secsfrom8am (subtract 8 hours * 3600 seconds)
data['secsfrom8am'] = data['secsfrommidnight'] - (8 * 3600)
data['secsfrom8am'] = data['secsfrom8am'] % (24 * 3600)  # Ensure values are within 0 to 86400

# Plot histogram of secsfrom8am
plt.figure(figsize=(12, 6))
plt.hist(data['secsfrom8am'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Histogram of Transformed Posting Times (secsfrom8am)')
plt.xlabel('Seconds from 8 AM')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Compute mean, median, and standard deviation of secsfrom8am
mean_secsfrom8am = data['secsfrom8am'].mean()
median_secsfrom8am = data['secsfrom8am'].median()
std_secsfrom8am = data['secsfrom8am'].std()

print(f'Mean of secsfrom8am: {mean_secsfrom8am}')
print(f'Median of secsfrom8am: {median_secsfrom8am}')
print(f'Standard Deviation of secsfrom8am: {std_secsfrom8am}')

# Compare the transformed posting times of messages in comp.graphics and soc.religion.christian
comp_graphics = data[data['groupID'] == 2]['secsfrom8am']
soc_religion_christian = data[data['groupID'] == 16]['secsfrom8am']

# Perform a t-test
t_stat, p_value = ttest_ind(comp_graphics, soc_religion_christian, equal_var=False)

print(f'T-test statistics: t = {t_stat}, p = {p_value}')
if p_value < 0.05:
    print("There is a statistically significant difference in the mean transformed posting times.")
else:
    print("There is no statistically significant difference in the mean transformed posting times.")
    
print()

# Part 4
print('Part 4')
print()

# Compute the correlation between 'jpeg' and 'gif' over all messages
correlation_jpeg_gif_all = data['jpeg'].corr(data['gif'])
print(f"Correlation between 'jpeg' and 'gif' over all messages: {correlation_jpeg_gif_all:.4f}")

# Compute the correlation between 'write' and 'sale' over all messages
correlation_write_sale_all = data['write'].corr(data['sale'])
print(f"Correlation between 'write' and 'sale' over all messages: {correlation_write_sale_all:.4f}")

# Filter messages of the newsgroup comp.graphics (groupID == 2)
comp_graphics_data = data[data['groupID'] == 2]

# Compute the correlation between 'jpeg' and 'gif' in the newsgroup comp.graphics
correlation_jpeg_gif_comp_graphics = comp_graphics_data['jpeg'].corr(comp_graphics_data['gif'])
print(f"Correlation between 'jpeg' and 'gif' in comp.graphics: {correlation_jpeg_gif_comp_graphics:.4f}")

print()

# Part 5
print('Part 5')
print()

# Handling deprecated options for inf values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Test for normality of the sentiment values over the entire dataset
# Using Kolmogorov-Smirnov test for larger dataset
stat, p_value = kstest(data['meanvalences'], 'norm', args=(data['meanvalences'].mean(), data['meanvalences'].std()))
print(f'Kolmogorov-Smirnov test for normality: stat = {stat}, p = {p_value}')
if p_value > 0.05:
    print("The sentiment values are normally distributed.")
else:
    print("The sentiment values are not normally distributed.")

# Compute distributions for sentiment in different newsgroups
grouped = data.groupby('groupID')['meanvalences']
distributions = grouped.agg(['mean', 'median', 'std'])
distributions['25%'] = grouped.quantile(0.25)
distributions['75%'] = grouped.quantile(0.75)
print(distributions)

# Plot histograms for the sentiment values in different newsgroups
plt.figure(figsize=(12, 8))
for group_id in grouped.groups:
    sns.histplot(data[data['groupID'] == group_id]['meanvalences'], kde=True, label=f'Group {group_id}')
plt.legend(title='Newsgroups')
plt.title('Histograms of Sentiment Values in Different Newsgroups')
plt.xlabel('Sentiment (meanvalences)')
plt.ylabel('Frequency')
plt.show()

# Identify the three most positive and three most negative newsgroups
sorted_distributions = distributions.sort_values(by='mean')
most_negative = sorted_distributions.head(3)
most_positive = sorted_distributions.tail(3)
print("Three most negative newsgroups:")
print(most_negative)
print("Three most positive newsgroups:")
print(most_positive)

# Perform statistical tests between pairs of newsgroups
def perform_ttest(group1, group2):
    data1 = data[data['groupID'] == group1]['meanvalences']
    data2 = data[data['groupID'] == group2]['meanvalences']
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    return t_stat, p_value

# Newsgroup pairs to test
newsgroup_pairs = [
    (4, 5),  # comp.sys.ibm.pc.hardware vs comp.sys.mac.hardware
    (10, 11),  # rec.sport.baseball vs rec.sport.hockey
    (8, 9)  # rec.autos vs rec.motorcycles
]

# Dictionary to map groupID to newsgroup names
newsgroup_names = {
    4: "comp.sys.ibm.pc.hardware",
    5: "comp.sys.mac.hardware",
    10: "rec.sport.baseball",
    11: "rec.sport.hockey",
    8: "rec.autos",
    9: "rec.motorcycles"
}

for group1, group2 in newsgroup_pairs:
    t_stat, p_value = perform_ttest(group1, group2)
    print(f'T-test between {newsgroup_names[group1]} and {newsgroup_names[group2]}: t = {t_stat}, p = {p_value}')
    if p_value < 0.05:
        print("There is a statistically significant difference in sentiment between the two newsgroups.")
    else:
        print("There is no statistically significant difference in sentiment between the two newsgroups.")

print()

# Part 6

print('Part 6')
print()

# a)
print('Part a')
print()

# Filter data for the two newsgroups
newsgroups_of_interest = [2, 15]  # 2: comp.graphics, 15: sci.space
filtered_data = data[data['groupID'].isin(newsgroups_of_interest)].copy()

# Create the response variable
filtered_data.loc[filtered_data['groupID'] == 2, 'response'] = 1  # 1 for comp.graphics
filtered_data.loc[filtered_data['groupID'] == 15, 'response'] = -1  # -1 for sci.space


# Use the occurrence count of the word 'jpeg' as the input variable
X = filtered_data[['jpeg']].values  # Feature matrix
y = filtered_data['response'].values  # Response variable

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the response variable
predictions = model.predict(X)

# Compute the mean squared error
mse = mean_squared_error(y, predictions)

print(f"Mean Squared Error of the prediction: {mse:.2f}")
print()

# b) 
print('Part b')
print()

# Filter data for the two newsgroups: comp.graphics (2) and sci.space (15)
data_subset = data[data['groupID'].isin([2, 15])].copy()

# Create the response variable
data_subset.loc[data_subset['groupID'] == 2, 'response'] = 1
data_subset.loc[data_subset['groupID'] == 15, 'response'] = -1

# Step (a) - Using 'jpeg' Alone
X_jpeg = data_subset[['jpeg']]
y = data_subset['response']

# Train the linear regression model
model_jpeg = LinearRegression()
model_jpeg.fit(X_jpeg, y)

# Predict the response variable
y_pred_jpeg = model_jpeg.predict(X_jpeg)

# Compute the mean squared error
mse_jpeg = mean_squared_error(y, y_pred_jpeg)
print(f'Mean Squared Error using "jpeg": {mse_jpeg:.4f}')

# Step (b) - Using 'jpeg' and 'earth'
X_jpeg_earth = data_subset[['jpeg', 'earth']]

# Train the linear regression model
model_jpeg_earth = LinearRegression()
model_jpeg_earth.fit(X_jpeg_earth, y)

# Predict the response variable
y_pred_jpeg_earth = model_jpeg_earth.predict(X_jpeg_earth)

# Compute the mean squared error
mse_jpeg_earth = mean_squared_error(y, y_pred_jpeg_earth)
print(f'Mean Squared Error using "jpeg" and "earth": {mse_jpeg_earth:.4f}')

print()

# Part c
print('Part c')
print()

# Assuming the group counts are columns in the data
groups = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7', 'group8']
X = filtered_data[groups].values  # Feature matrix

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the response variable
predictions = model.predict(X)

# Compute the mean squared error
mse = mean_squared_error(y, predictions)

print(f"Mean Squared Error of the prediction using word groups: {mse:.4f}")

# Determine which groups helped the most in the prediction
coefficients = model.coef_
for group, coef in zip(groups, coefficients):
    print(f"Coefficient for {group}: {coef:.4f}")