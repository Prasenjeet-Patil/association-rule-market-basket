import streamlit as st
import math
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SmartCart AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
    <style>

    .main {
        background-color: #f7f7f7;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    
            
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        background: white;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

    .product-name {
        font-size: 18px;
        font-weight: 600;
        color: white;
        margin-top: 8px;
    }

    .product-price {
        font-size: 16px;
        color: #ff7a00;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 15px;
    }

    .cart-box {
        background: white;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .recommend-box {
        background: #f0fff2;
        border: 2px solid #32c766;
        padding: 12px;
        border-radius: 16px;
        text-align: center;
    }
                

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown(
    "<h1 style='text-align:center;'>🛒 SmartCart AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;font-size:18px;'>"
    "AI-Powered Grocery Recommendation System"
    "</p>",
    unsafe_allow_html=True
)

# =====================================================
# LOAD DATASET
# =====================================================
# transactions = []

# with open("groceries.csv", "r", encoding="utf-8") as file:
#     for line in file:
#         line = line.strip()

#         if line:
#             items = [item.strip().lower() for item in line.split(",") if item]
#             transactions.append(items)

@st.cache_data
def load_transactions():

    transactions = []

    with open("groceries.csv", "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line:

                items = [
                    item.strip().lower()
                    for item in line.split(",")
                    if item
                ]

                transactions.append(items)

    return transactions


transactions = load_transactions()

# # =====================================================
# # ENCODE DATA
# # =====================================================
# te = TransactionEncoder()
# te_array = te.fit(transactions).transform(transactions)
# df = pd.DataFrame(te_array, columns=te.columns_)

# # =====================================================
# # APRIORI MODEL
# # =====================================================
# frequent_items = apriori(
#     df,
#     min_support=0.01,
#     use_colnames=True
# )

# rules = association_rules(
#     frequent_items,
#     metric="confidence",
#     min_threshold=0.3
# )

# rules = rules[rules['lift'] >= 1]


@st.cache_data
def generate_rules(transactions):

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    frequent_items = apriori(
        df,
        min_support=0.005,
        use_colnames=True
    )

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.2
    )

    rules = rules[rules['lift'] >= 1]

    return te, df, rules


te, df, rules = generate_rules(transactions)

# =====================================================
# DUMMY PRICES
# =====================================================
prices = {}

for idx, item in enumerate(sorted(te.columns_)):
    prices[item] = (idx % 10 + 1) * 20

# =====================================================
# SESSION STATE
# =====================================================
if "cart" not in st.session_state:
    st.session_state.cart = {}

if "page" not in st.session_state:
    st.session_state.page = 1

# =====================================================
# SEARCH BAR
# =====================================================
search = st.text_input(
    "🔍 Search Products",
    placeholder="Search grocery items..."
)

# =====================================================
# PRODUCT SECTION
# =====================================================
st.markdown(
    "<div class='section-title'>🛍️ Available Products</div>",
    unsafe_allow_html=True
)

products = sorted(list(te.columns_))

# SEARCH FILTER
if search:
    products = [
        p for p in products
        if search.lower() in p.lower()
    ]

# =====================================================
# PAGINATION
# =====================================================
products_per_page = 12

total_pages = math.ceil(
    len(products) / products_per_page
)

start_idx = (
    (st.session_state.page - 1)
    * products_per_page
)

end_idx = start_idx + products_per_page

current_products = products[start_idx:end_idx]

# =====================================================
# PRODUCT GRID
# =====================================================
cols_per_row = 4

for i in range(0, len(current_products), cols_per_row):

    cols = st.columns(cols_per_row)

    for col_idx in range(cols_per_row):

        if i + col_idx < len(current_products):

            product = current_products[i + col_idx]

            with cols[col_idx]:

                with st.container(border=True):

                    image_path = f"images/{product}.jpg"

                    # PRODUCT IMAGE
                    if os.path.exists(image_path):
                        st.image(
                            image_path,
                            use_container_width=True
                        )
                    else:
                        st.image(
                            "images/default.webp",
                            use_container_width=True
                        )

                    # PRODUCT NAME
                    st.markdown(
                        f"""
                        <div class='product-name'>
                            {product.title()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # PRICE
                    st.markdown(
                        f"""
                        <div class='product-price'>
                            ₹{prices[product]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    quantity = st.session_state.cart.get(product, 0)

                    qcol1, qcol2, qcol3 = st.columns([1,1,1])

                    with qcol1:
                        if st.button(
                            "➖",
                            key=f"minus_{product}"
                        ):

                            if quantity > 0:

                                st.session_state.cart[product] -= 1

                                if st.session_state.cart[product] == 0:
                                    del st.session_state.cart[product]

                                st.toast(
                                    f"{product.title()} removed"
                                )

                                st.rerun()

                    with qcol2:
                        st.markdown(
                            f"""
                            <h4 style='text-align:center;'>
                                {quantity}
                            </h4>
                            """,
                            unsafe_allow_html=True
                        )

                    with qcol3:
                        if st.button(
                            "➕",
                            key=f"plus_{product}"
                        ):

                            st.session_state.cart[product] = quantity + 1

                            st.toast(
                                f"{product.title()} added to cart 🛒"
                            )

                            st.rerun()

# =====================================================
# PAGINATION CONTROLS
# =====================================================
prev_col, mid_col, next_col = st.columns([1,2,1])

with prev_col:
    if st.session_state.page > 1:
        if st.button("⬅ Previous"):
            st.session_state.page -= 1
            st.rerun()

with mid_col:
    st.markdown(
        f"<h4 style='text-align:center;'>Page {st.session_state.page} of {total_pages}</h4>",
        unsafe_allow_html=True
    )

with next_col:
    if st.session_state.page < total_pages:
        if st.button("Next ➡"):
            st.session_state.page += 1
            st.rerun()


# =====================================================
# RIGHT SIDEBAR CART
# =====================================================

if len(st.session_state.cart) > 0:

    with st.sidebar:

        st.markdown("## 🧺 Your Cart")

        total = 0

        for item, qty in st.session_state.cart.items():

            subtotal = prices[item] * qty
            total += subtotal

            cart_col1, cart_col2 = st.columns([3,1])

            with cart_col1:
                st.write(f"**{item.title()}**")
                st.caption(f"Qty: {qty}")

            with cart_col2:
                st.write(f"₹{subtotal}")

        st.markdown("---")

        st.markdown(f"### 💰 Total: ₹{total}")

        # # =====================================================
        # # RECOMMENDATIONS
        # # =====================================================

        # st.markdown("---")
        # st.markdown("## ⭐ Recommended For You")

        # recommendations = []

        # for cart_item in st.session_state.cart.keys():

        #     filtered_rules = rules[
        #         rules['antecedents'].apply(
        #             lambda x: cart_item in list(x)
        #         )
        #     ]

        #     filtered_rules = filtered_rules.sort_values(
        #         by=['confidence', 'lift'],
        #         ascending=False
        #     )

        #     for _, row in filtered_rules.iterrows():

        #         for item in row['consequents']:

        #             if (
        #                 item not in st.session_state.cart
        #                 and item not in recommendations
        #             ):
        #                 recommendations.append(item)

        # if recommendations:

        #     for item in recommendations[:6]:

        #         rec_col1, rec_col2 = st.columns([2,1])

        #         with rec_col1:

        #             image_path = f"images/{item}.jpg"

        #             if os.path.exists(image_path):
        #                 st.image(
        #                     image_path,
        #                     width=70
        #                 )

        #             st.write(f"**{item.title()}**")
        #             st.caption(f"₹{prices[item]}")

        #         with rec_col2:

        #             if st.button(
        #                 "➕ Add",
        #                 key=f"sidebar_rec_{item}"
        #             ):

        #                 current_qty = st.session_state.cart.get(item, 0)

        #                 st.session_state.cart[item] = current_qty + 1

        #                 st.rerun()

        #         st.markdown("---")

        # else:
        #     st.info("No recommendations yet.")



        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        st.markdown("---")
        st.markdown("## ⭐ Recommended For You")

        recommendations = []

        # =====================================================
        # ML-BASED RECOMMENDATIONS
        # =====================================================

        for cart_item in st.session_state.cart.keys():

            cart_item = cart_item.strip().lower()

            for _, row in rules.iterrows():

                antecedents = [str(i).strip().lower() for i in row['antecedents']]
                consequents = [str(i).strip().lower() for i in row['consequents']]

                # Check if cart item exists in antecedents
                if cart_item in antecedents:

                    for item in consequents:

                        if (
                            item not in st.session_state.cart
                            and item not in recommendations
                        ):
                            recommendations.append(item)

        # =====================================================
        # FALLBACK RECOMMENDATIONS
        # =====================================================

        if len(recommendations) < 6:

            popular_products = list(
                df.sum()
                .sort_values(ascending=False)
                .index
            )

            for item in popular_products:

                item = item.strip().lower()

                if (
                    item not in st.session_state.cart
                    and item not in recommendations
                ):
                    recommendations.append(item)

                if len(recommendations) >= 6:
                    break

        # =====================================================
        # DISPLAY RECOMMENDATIONS
        # =====================================================

        if recommendations:

            for item in recommendations[:4]:

                rec_col1, rec_col2 = st.columns([2,1])

                with rec_col1:

                    image_path = f"images/{item}.jpg"

                    if os.path.exists(image_path):
                        st.image(
                            image_path,
                            width=70
                        )

                    st.write(f"**{item.title()}**")
                    st.caption(f"₹{prices.get(item, 100)}")

                with rec_col2:

                    if st.button(
                        "➕ Add",
                        key=f"sidebar_rec_{item}"
                    ):

                        current_qty = st.session_state.cart.get(item, 0)

                        st.session_state.cart[item] = current_qty + 1

                        st.session_state.cart[product] = quantity + 1

                        st.toast(
                            f"{product.title()} added to cart 🛒"
                        )

                        st.rerun()

                st.markdown("---")

        else:
            st.info("No recommendations available.")

        # =====================================================
        # CLEAR CART
        # =====================================================

        if st.button("🗑️ Clear Cart"):

            st.session_state.cart = {}

            st.rerun()