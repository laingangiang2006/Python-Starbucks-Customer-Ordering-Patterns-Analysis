import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
df = pd.read_csv("/Users/laingangiang/Downloads/Starbucks Customer Ordering Patterns/starbucks_customer_ordering_patterns.csv")

df.rename(columns={"customer_id": "Customer ID", 
                   "order_id": "Order ID", 
                   "order_date": "Order Date", 
                   "order_time": "Order Time", 
                   "order_channel": "Order Channel", 
                   "store_id": "Store ID", 
                   "store_location_type": "Store Location Type", 
                   "region": "Region", 
                   "customer_age_group": "Customer Age Group", 
                   "customer_gender": "Customer Gender", 
                   "is_rewards_member": "Is Rewards Member",
                   "cart_size": "Cart Size", 
                   "day_of_week": "Day of Week",
                   "customer_gender": "Customer Gender",
                   "cart_size": "Cart Size",
                   "num_customizations": "Number of Customizations",
                   "total_spend": "Total Spending",
                   "drink_category": "Drink Category",
                   "fulfillment_time_min": "Fulfillment Time",
                   "customer_satisfaction": "Customer Satisfaction"}, 
                   inplace=True)
df.drop_duplicates("Order ID", inplace=True)

# Display the structure of the dataset
print(df.info())

# Display the first and last few rows of the dataset
print(df.head(10))
print(df.tail(10))

# Check for missing values
print(df.isnull().sum())

# Basic statistics of the dataset
print(df.describe())

# Count the number of orders for each day of the week
orders_per_day = df["Day of Week"].value_counts()
orders_per_day.index.name = "Orders per day"

print(orders_per_day)

# Count the number of unique customer ids for each day of the week
unique_customer_count_per_day = df.groupby("Day of Week")["Customer ID"].nunique()
unique_customer_count_per_day.index.name = "Unique Customer Count per day"

print(unique_customer_count_per_day)

# Pivot table
pivot_table1 = pd.concat([orders_per_day, unique_customer_count_per_day], axis=1)
pivot_table1.columns=("Order per day", "Number of Unique Customers")

print(pivot_table1)

# Visualization of the number of orders and customer count per day of the week
y_min = pivot_table1.min().min()
y_max = pivot_table1.max().max()

pivot_table1.plot(kind="bar", width=0.6, figsize=(12, 6))
plt.title("Number of Orders and Customers per Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.yticks(np.arange(0, y_max + 1000, 1000))
plt.tight_layout()
plt.show()

# Number of visits
visits_per_customer = df.groupby("Customer ID")["Order ID"].nunique()
visits_per_customer.index.name = "Customer ID"
visit_distribution = visits_per_customer.value_counts().sort_index()

# Distribution of visits per customer
plt.figure(figsize=(10, 6))
plt.bar(visit_distribution.index, visit_distribution.values, color="steelblue")
plt.title("Distribution of Visits per Customer")
plt.xlabel("Number of Visits")
plt.ylabel("Number of Customers")
plt.xticks(np.arange(1, visits_per_customer.max() + 1, 1))
plt.tight_layout()
plt.show()

# Number of visits by age group
visits_per_age_group = df.groupby("Customer Age Group")["Order ID"].nunique().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(visits_per_age_group.index, visits_per_age_group.values, color="steelblue")
plt.title("Distribution of Visits per Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Unique Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Number of Orders by Time of Day
df["Order Time"] = pd.to_datetime(df["Order Time"], format="%H:%M")
df["Hour"] = df["Order Time"].dt.hour

bins = [0, 6, 12, 17, 21, 24]
labels = ["Early morning (0–6)", "Morning (6–12)", "Afternoon (12–17)", "Evening (17–21)", "Night (21–24)"]

df["Time of order"] = pd.cut(df["Hour"], bins = bins, labels = labels, include_lowest = True)
df["Time of order"].value_counts().reindex(labels).plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Number of Orders by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Number of Orders by Store Location Type
visits_per_location_type = df.groupby("Store Location Type")["Order ID"].nunique().sort_index()
visits_per_location_type.plot(kind='pie', autopct='%1.1f%%', figsize=(7, 7))

plt.xlabel("")
plt.ylabel("")
plt.title("Order number by Store Location Type")
plt.show()

# Order number by Store Location Type
avg_satisfaction_by_channel = df.groupby("Order Channel")["Customer Satisfaction"].mean().sort_index()

avg_satisfaction_by_channel.plot(kind="bar", figsize=(10, 6), color="steelblue")
plt.title("Average Customer Satisfaction by Order Channel")
plt.xlabel("Order Channel")
plt.ylabel("Average Satisfaction Score")
plt.xticks(rotation=45)
plt.ylim(3, 4)
plt.tight_layout()
plt.show()

# Drink Category by Customer Gender
drink_cat_by_gender = pd.crosstab(df["Drink Category"], df["Customer Gender"])
drink_cat_by_gender.plot(kind="bar", figsize=(10, 6))
plt.title("Drink Category by Gender")
plt.xlabel("Drink Category")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Distribution of Visits by Order Channel
visits_per_ord_channel = df.groupby("Order Channel")["Order ID"].nunique().sort_index()

plt.figure(figsize=(10, 6))
plt.pie(visits_per_ord_channel.values, labels=visits_per_ord_channel.index, colors=["steelblue", "coral", "mediumseagreen", "gold"], autopct="%1.1f%%")
plt.title("Distribution of Visits by Order Channel")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of Visits by Day of Week
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
visits_per_day = orders_per_day.reindex(day_order)  # ← use orders_per_day here
colors = ["steelblue", "coral", "mediumseagreen", "gold", "orchid", "tomato", "skyblue"]
plt.figure(figsize=(10, 6))
plt.pie(visits_per_day.values, labels=visits_per_day.index, colors=colors, autopct="%1.1f%%")
plt.title("Distribution of Visits by Day of Week")
plt.tight_layout()
plt.show()
