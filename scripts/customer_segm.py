# %% [markdown]
# # PROYECTO FINAL

# %% [markdown]
# ### 0. Librerias

# %%
pip install openpyxl

# %%
pip install scikit-learn matplotlib seaborn


# %%
import numpy as np
import pickle, base64, requests, os

# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns


# %%

from sklearn.ensemble import HistGradientBoostingRegressor

excel=pd.read_excel('C:\\ER\\ML1\\archive\\Online Retail.xlsx')

# %%
##lo neceitamos en csv par subirlo al Github
# excel.to_csv('online_retail.csv', index=False)
# # df=pd.read_csv('online_retail.csv')

# %%
df=excel.copy()

# %%
# Exploración inicial
print(df.head())
print(df.info())


# %% [markdown]
# ### 2. Preparación Data

# %%
df.isnull().sum()

# %%
df = df.drop(columns=["Description"])
df.dropna(inplace = True)


# %%
df.isnull().sum()

# %%
df.shape

# %%
df[df["Quantity"]<0].head()


# %%
df["Total"] = df["Quantity"] * df["UnitPrice"]

# %%
# df[~(df["InvoiceNo"].astype(str).str.isdigit())].head()

df["Cancel"] = (df["Quantity"]<0)
df["Cancel"] = ~(df["InvoiceNo"].astype(str).str.isdigit())

# %%
df

# %%
aggreg = df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate", "Cancel"]).agg(
    InvoiceTotal = ("Total", "sum"),
    ProductCount = ("Total", "count")
).reset_index()

# %%
def calculateInvoices(CancellationValues):
    return CancellationValues.map({True: -1, False: 1}).sum()

lastInvoiceDate = aggreg[aggreg["Cancel"] == False]["InvoiceDate"].max()

def findTimeInterval(InvoiceDate):
    return lastInvoiceDate - InvoiceDate.max()

total_data = aggreg.groupby(["CustomerID"]).agg(
    CustomerInvoiceTotal = ("InvoiceTotal", "sum"),
    CustomerInvoiceCount = ("Cancel", calculateInvoices),
    TimeIntervalAfterLastInvoice = ("InvoiceDate", findTimeInterval)
).reset_index()

# %%

total_data["RecencyDays"] = total_data["TimeIntervalAfterLastInvoice"].dt.days

# %%
print(total_data.shape)
print(total_data.info())
print(total_data.describe())

# %% [markdown]
# ### 3.1 K-means 

# %%

plt.figure(figsize=(15,5))
sns.histplot(total_data["CustomerInvoiceTotal"], kde=True, bins=30)
plt.title("Histogram of Monetary Metric")
plt.xlabel("Monetary")


# %%
sns.histplot(total_data["CustomerInvoiceCount"], kde=True, bins=30)
plt.title("Histogram of Frequency Metric")
plt.xlabel("Frequency")


# %%
plt.figure(figsize=(15,5))
sns.histplot(total_data["RecencyDays"], kde=True, bins=30)
plt.title("Histogram of Recency Metric")
plt.xlabel("Recency")
plt.show()


# %%
plt.figure(figsize = (8,6))
sns.scatterplot(data = total_data, x = "CustomerInvoiceTotal", y = "CustomerInvoiceCount")
plt.title("Monetary and Frequency Metrics of customers", fontweight="bold")
plt.xlabel("Monetary")
plt.ylabel("Frequency")
plt.show()

# %%
plt.figure(figsize = (8,6))
sns.scatterplot(data = total_data, x = "RecencyDays", y = "CustomerInvoiceCount")
plt.title("Monetary and Frequency Metrics of customers", fontweight="bold")
plt.xlabel("days")
plt.ylabel("Frequency")
plt.show()

# %%
total_data

# %%

total_data = total_data[total_data["CustomerInvoiceTotal"] > 0]
total_data = total_data[total_data["CustomerInvoiceTotal"] < 15000]
total_data = total_data[total_data["CustomerInvoiceCount"] < 50]

# %%
total_data

# %%

from sklearn.preprocessing import StandardScaler

# %%

X_tot = total_data[["CustomerInvoiceTotal", "CustomerInvoiceCount", "RecencyDays"]]
scaler = StandardScaler()
X = scaler.fit_transform(X_tot[["CustomerInvoiceTotal","CustomerInvoiceCount","RecencyDays"]])

Ks = range(2, 11)
inercias, siluetas = [], []
for k in Ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inercias.append(m.inertia_)
    siluetas.append(silhouette_score(X, m.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(Ks, inercias, marker='o')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inercia (J)')
axes[0].set_title('Método del codo')
axes[0].grid(True)

axes[1].plot(Ks, siluetas, marker='o', color='darkgreen')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette promedio')
axes[1].set_title('Coeficiente de silueta')
axes[1].grid(True)
plt.tight_layout(); plt.show()

print(f'K con mayor silueta: K={Ks[int(np.argmax(siluetas))]}')
print(siluetas[1:1])


# %%

final_model = KMeans(n_clusters=2, random_state=42)


# %%

total_data["Cluster"] = final_model.fit_predict(X)

# %%


# %%
summary = total_data.groupby("Cluster").agg(
    CustomerCount = ("CustomerID", "count"),
    AverageTotalAmount = ("CustomerInvoiceTotal", "mean"),
    AverageInvoiceCount = ("CustomerInvoiceCount", "mean"),
    AverageRecency = ("RecencyDays", "mean")
).reset_index()



summary = summary.rename(columns={
    "Label": "Cluster",
    "CustomerCount": "Customer Count",
    "AverageTotalAmount": "Mean Monetary",
    "AverageInvoiceCount": "Mean Frequency",
    "AverageRecency": "Mean Recency"
})

summary = summary.round(2)

summary.head()

# %%
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))

plt.subplot(1,3,1)
sns.boxplot(data = total_data, x = "Cluster", y = "CustomerInvoiceTotal", hue = "Cluster", palette = "bright")
plt.title("Total amount of Invoices and Classes")
plt.ylabel("Total amount")
plt.subplot(1,3,2)
sns.boxplot(data = total_data, x = "Cluster", y = "CustomerInvoiceCount", hue = "Cluster", palette = "bright")
plt.title("Counts of Invoices and Classes")
plt.ylabel("Count")
plt.subplot(1,3,3)
sns.boxplot(data = total_data, x = "Cluster", y = "RecencyDays", hue = "Cluster", palette = "bright")
plt.title("Time Interval After Last Invoices of Customers and Classes")
plt.ylabel("Time Interval")
plt.tight_layout()
plt.show()


# %%
sns.scatterplot(data = total_data, x = "CustomerInvoiceTotal", y = "CustomerInvoiceCount", hue = "Cluster", palette = "bright")
plt.title("Scatter Plot of Total Amount and Invoice Counts with Classes")
plt.xlabel("Total Amount of Invoices")
plt.ylabel("Count of Invoices")
plt.show()

# %%
sns.scatterplot(data = total_data, x = "CustomerInvoiceTotal", y = "RecencyDays", hue = "Cluster", palette = "bright")
plt.title("Scatter Plot of Total Amount and Invoice Counts with Classes")
plt.xlabel("Total Amount of Invoices")
plt.ylabel("Recency Days of Invoices")
plt.show()

# %%
sns.scatterplot(data = total_data, x = "RecencyDays", y = "CustomerInvoiceCount", hue = "Cluster", palette = "bright")
plt.title("Scatter Plot of Total Amount and Invoice Counts with Classes")
plt.xlabel("Recency Days of Invoices")
plt.ylabel("Count of Invoices")
plt.show()

# %%
## guardamos pkl

with open("customer_segmentation_model.pkl", "wb") as file:
    pickle.dump(final_model, file)


