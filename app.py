import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("Market Basket Analysis & Smart Recommendation System")

# =========================
# Sidebar Settings
# =========================
st.sidebar.header("⚙️ Settings")

min_support = st.sidebar.slider("Min Support", 0.001, 1.0, 0.02, step=0.001)
min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.3, step=0.05)
min_lift = st.sidebar.slider("Min Lift (Filter weak rules)", 0.5, 5.0, 1.0, step=0.1)

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    transactions = []

    for line in uploaded_file:
        line = line.decode("utf-8").strip()
        if line:
            items = [item.strip() for item in line.split(",") if item]
            transactions.append(items)

    st.success(f"✅ Loaded {len(transactions)} transactions")

    # =========================
    # Encoding
    # =========================
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    # =========================
    # Apriori
    # =========================
    frequent_items = apriori(df, min_support=min_support, use_colnames=True)

    if not frequent_items.empty:
        rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)

        # Apply lift filter
        rules = rules[rules['lift'] >= min_lift]

        # =========================
        # Display Results
        # =========================
        st.subheader("Frequent Itemsets")
        st.dataframe(frequent_items)

        st.subheader("Association Rules (Filtered)")
        if not rules.empty:
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.warning("No strong rules found. Try lowering filters.")

        # =========================
        # GRAPH (Support vs Confidence)
        # =========================
        if not rules.empty:
            st.subheader("Rule Visualization")

            plt.figure()
            plt.scatter(rules['support'], rules['confidence'])
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.title("Support vs Confidence")

            st.pyplot(plt)

        # =========================
        # SMART RECOMMENDATION SYSTEM
        # =========================
        st.subheader("Smart Item Recommendation")

        item_list = sorted(list(te.columns_))
        selected_item = st.selectbox("Select an item", item_list)

        if selected_item:
            filtered_rules = rules[rules['antecedents'].apply(lambda x: selected_item in list(x))]

            if not filtered_rules.empty:
                # Sort by confidence + lift
                filtered_rules = filtered_rules.sort_values(by=['confidence', 'lift'], ascending=False)

                st.write("Top Recommendations:")

                recommendations = []
                for _, row in filtered_rules.head(5).iterrows():
                    recommendations.extend(list(row['consequents']))

                st.success(", ".join(set(recommendations)))

                # Show detailed table
                st.subheader("Recommendation Details")
                st.dataframe(filtered_rules[['antecedents', 'consequents', 'confidence', 'lift']])

            else:
                st.warning("No recommendations found for this item.")

        # =========================
        # DOWNLOAD BUTTON
        # =========================
        st.download_button(
            label="Download Rules",
            data=rules.to_csv(index=False),
            file_name="association_rules.csv",
            mime="text/csv"
        )

    else:
        st.error("No frequent itemsets found. Lower support value.")

else:
    st.info("Upload a dataset to start.")