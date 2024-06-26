import csv
import datetime
import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from openai import OpenAI
import random
import streamlit.components.v1 as components
import os
from kbcstorage.client import Client

# Setting page config
st.set_page_config(page_title="RFM DataApp")

# Constants
kbc_url = st.secrets["kbc_url"]
kbc_token = st.secrets["kbc_token"]
openai_token = st.secrets["openai_token"]
read_bucket = st.secrets["read_bucket"]
write_bucket = st.secrets["write_bucket"]
LOGO_IMAGE_PATH = os.path.abspath("./static/keboola.png")

client = Client(kbc_url, kbc_token)


@st.cache_data(ttl=60, show_spinner=False)
def hide_custom_anchor_link():
    st.markdown(
        """
        <style>
            /* Hide anchors directly inside custom HTML headers */
            h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a {
                display: none !important;
            }
            /* If the above doesn't work, it may be necessary to target by attribute if Streamlit adds them dynamically */
            [data-testid="stMarkdown"] h1 a, [data-testid="stMarkdown"] h3 a,[data-testid="stMarkdown"] h5 a,[data-testid="stMarkdown"] h2 a {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60, show_spinner=False)
def display_footer_section():
    # Inject custom CSS for alignment and style
    st.markdown(
        """
        <style>
            .footer {
                width: 100%;
                font-size: 14px;  /* Adjust font size as needed */
                color: #22252999;  /* Adjust text color as needed */
                padding: 10px 0;  /* Adjust padding as needed */
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .footer p {
                margin: 0;  /* Removes default margin for p elements */
                padding: 0;  /* Ensures no additional padding is applied */
            }
        </style>
        <div class="footer">
            <p>© Keboola 2024</p>
            <p>Version 1.0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ChangeButtonColour(widget_label, font_color, background_color, border_color):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}';
                    elements[i].style.borderColor = '{border_color}';
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)


def get_dataframe(table_name):
    """
    Reads the provided table from the specified table in Keboola Connection.

    Args:
        table_name (str): The name of the table to write the data to.

    Returns:
        The table as dataframe
    """
    table_detail = client.tables.detail(table_name)
    client.tables.export_to_file(table_id=table_name, path_name="")
    list = client.tables.list()
    with open("./" + table_detail["name"], mode="rt", encoding="utf-8") as in_file:
        lazy_lines = (line.replace("\0", "") for line in in_file)
        reader = csv.reader(lazy_lines, lineterminator="\n")
    if os.path.exists("data.csv"):
        os.remove("data.csv")
    else:
        print("The file does not exist")
    os.rename(table_detail["name"], "data.csv")
    data = pd.read_csv("data.csv")
    return data


def check_keboola_bucket(bucket_name, table_name, table_path):
    """
    Checks whether the bucket and the table exist in Keboola,
    if either doesn't exist they will be created

    Args:
        bucket_name (str): The name of the bucket to check.
        table_name (str): The name of the table to check.
        table_path (str): The local file path to write the data to before uploading.

    Returns:
        Bool: Whether everything existed
    """
    result = True
    try:
        client.buckets.detail("in.c-" + bucket_name)
        print("Bucket exists")
    except:
        client.buckets.create(bucket_name)
        result = False
        print("Bucket created")
    try:
        client.tables.detail("in.c-" + write_bucket + "." + table_name)
        print("Table exists")
    except:
        client.tables.create(
            name=table_name,
            bucket_id="in.c-" + bucket_name,
            file_path=table_path,
            primary_key=["key"],
        )
        result = False
        print("Table created")
    return result


def write_to_keboola(data, table_name, table_path, incremental):
    """
    Writes the provided data to the specified table in Keboola Connection,
    updating existing records as needed.

    Args:
        data (pandas.DataFrame): The data to write to the table.
        table_name (str): The name of the table to write the data to.
        table_path (str): The local file path to write the data to before uploading.
        incremental (bool): True if incremental

    Returns:
        None
    """

    # Write the DataFrame to a CSV file with compression
    data.to_csv(table_path, index=False, compression="gzip")
    check_keboola_bucket(write_bucket, table_name, table_path)
    # Load the CSV file into Keboola, updating existing records
    client.tables.load(
        table_id="in.c-" + write_bucket + "." + table_name,
        file_path=table_path,
        is_incremental=incremental,
    )


def load_data():
    """
    Loads data from Keboola. Assumes that the table in.c-read_bucket.order exists.
    Assumes the table has the 4 columns: "customer_id", "created_at", "total_price_usd", "financial_status"
    Processing:
    - Only takes the rows with financial_status==paid.
    - Converts the dates into a format the app is designed to read
    - Adds a count field
    - Groups data by customer ID and date
    - Renames all columns to fit the app design
    Saves the result in the session


    Returns:
        Dataframe containing the processed data from the Keboola table
    """
    if "data" not in st.session_state:
        st.session_state["data_load_timestamp"] = datetime.datetime.now()
        data = get_dataframe("in.c-" + read_bucket + ".order")
        data = data[
            ["customer_id", "created_at", "total_price_usd", "financial_status"]
        ]
        data.dropna(subset=["customer_id"], inplace=True)
        data = data[data["financial_status"] == "paid"]
        data["count"] = 1
        data["created_at"] = data["created_at"].str[:10]
        data = (
            data[["customer_id", "created_at", "total_price_usd", "count"]]
            .groupby(["customer_id", "created_at"])
            .sum()
            .reset_index()
        )
        data.rename(
            columns={
                "customer_id": "id",
                "created_at": "date",
                "count": "num_of_events",
                "total_price_usd": "value",
            },
            inplace=True,
        )
        st.session_state["data"] = data
    return st.session_state["data"]


def get_openai_response(ai_setup, prompt, api_key):
    """
    Writes the provided data to the specified table in Keboola Connection,
    updating existing records as needed.

    Args:
        ai_setup (str): The instructions to send to OpenAI. In case of a conversation this is instructions for the system.
        prompt (str): In case of a conversation this is instructions for the user.
        api_key (str): OpenAI API key

    Returns:
        The text from the response from OpenAI
    """

    open_ai_client = OpenAI(
        api_key=api_key,
    )
    random.seed(42)
    messages = [{"role": "system", "content": ai_setup}]
    if prompt:
        messages.append({"role": "user", "content": prompt})

    try:
        completion = open_ai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0
        )

        message = completion.choices[0].message.content

        # Extracting the text response from the response object
        return message

    except Exception as e:
        return f"An error occurred: {e}"


def get_segments_info_from_ai(segments):
    result = {}
    ai_setup = """
                You are an expert in RFM (Recency, Frequency, Monetary) analysis. The user will provide you with the name of a segment derived from RFM analysis. Your task is to:
                Explain the meaning of the given segment, including its characteristics and what the RFM scores indicate about customers in this segment in up to 3 sentences in one paragraph. Do not mention the monetary score or the name of the segment in your explanation
                Provide actionable tips and strategies for managing and engaging with customers in this segment to improve their loyalty, satisfaction, and overall value to the business as a list of points, choose the most common 3 points.
                Return the answer as a json of the following format:
                { "segment":" "champions",
                "explanation":  "meaning of the given segment",
                "action_points": "1. action point 1
                action point 2
                action point 3"}
                ! Remember not to mention monetary score in or the name of the segment your explanation
                ! make sure there is a new line for each action point
                ! start the explanation with the word Customers
                ! if the word value is in the user prompt explain using average rfm score
                """
    for segment in segments:
        try:
            response = json.loads(get_openai_response(ai_setup, segment, openai_token))
            result[segment] = {
                "explanation": response["explanation"],
                "action_points": response["action_points"],
            }
        except Exception as e:
            return f"An error occurred while trying to communicate with OpenAI, please refresh the page"
    return result


def get_overview_insights_from_ai(data):
    ai_setup = """
            You are an expert on RFM (Recency, Frequency, Monetary) analysis. 
            You will receive a dataframe from the user that contains RFM segments and is already segmented and aggregated. 
            The dataframe will include the following columns:
                Segment: The RFM segment.
                total_customers: The count of customers in each segment.
                num_of_events: The total number of transactions for each segment.
                value: The total monetary value for each segment.
                customer_percentage_per_month: The percentage of the total customer base that each segment represents.
                value_percentage_per_month: The percentage of the total monetary value that each segment represents.
                events_percentage_per_month: The percentage of the total transactions that each segment represents.
                
            Your task is to:
            1. Analyze the provided data, identify key patterns and insights and summarize the three most important insights in a clear and concise manner, explaining why these insights are significant, and offer an actionable action. In your answer use macro segments that will include a few of the segments grouped together
            2. Summarize the state of the company in up to 3 sentences. In your summary include whether you believe the company is in a positive state or a negative one
            Use your expertise to deliver a thorough and actionable executive summary, ensuring that your observations and recommendations are clear, concise, and data-driven.
                        the report should be in the following format:
                        "Key insights:
                        1. insight 1
                        2. insight 2
                        3. insight 3
                        ...
                        Summary:
                        Executive summary goes here

            ! remember to use the data for your recommendations
            ! remember to base you insights and summary on the data, write only things that are derived from the data            

            Here is the dataframe for analysis:
            {}
            """.format(
        data.to_string(index=False)
    )
    return get_openai_response(ai_setup, None, openai_token)


def get_movements_insights_from_ai(data):
    ai_setup = """
            You are an expert on RFM (Recency, Frequency, Monetary) analysis. 
            You will receive a dataframe from the user that contains RFM segments and is already segmented and aggregated. 
            The dataframe will include the following columns:
                Segment: The RFM segment.
                month: The month.
                total_customers: The count of customers in each segment.
                num_of_events: The total number of transactions for each segment.
                value: The total monetary value for each segment.
                customer_percentage_per_month: The percentage of the total customer base that each segment represents.
                value_percentage_per_month: The percentage of the total monetary value that each segment represents.
                events_percentage_per_month: The percentage of the total transactions that each segment represents.

            Your task is to analyze the provided data, identify key patterns and insights and summarize the most important insights in a clear and concise manner, explaining why these insights are significant, and offer an actionable action.
            
            Use your expertise to deliver a thorough and actionable executive summary, ensuring that your observations and recommendations are clear, concise, and data-driven. 
            
            Write down the two most important insights for risks and threats and two for the positive aspects.
            When applicable, you can use macro segments to include a few segments in one insight.
            When writing your insights, clearly categorize each one as either positive or negative. 
            Provide a brief explanation for why each insight falls into its respective category. 
            Keep the following in mind:
            1. A decrease in the size or value of undesirable segments (e.g., 'Low Value', 'At risk', 'Lost', 'Hibernating', 'Can't Lose', 'About to sleep') is generally positive, as it indicates fewer customers in less desirable categories.
            2. An increase in the size or value of desirable segments (e.g., 'High Value', 'Champions', 'Loyal customers', 'Potential Loyalists') is generally positive, as it indicates more customers in desirable categories.
            3. A decrease in desirable segments or an increase in undesirable segments can be considered negative.
            4. Instability and fluctuation in undesirable segments (e.g., 'Low Value', 'At risk', 'Lost', 'Hibernating', 'Can't Lose', 'About to sleep') is not considered an important insight. refrain from using this kind of insights unless you don't have enough insights to write
            
            The answer should be in the following format:
                Positive aspects:
                1. insight 1
                2. insight 2

                Negative aspects:
                1. insight 1
                2. insight 2
                
            follow this format strictly

            ! remember to use the data for your recommendations
            ! remember to base you insights on the data, write only things that are derived from the data
            ! remember to write concisely and to the point, write the least amount of words you can in order to get your point across. 
            ! remember to put the positive aspects first.    
            ! remember to follow the format provided for you answer

            Here is the dataframe for analysis:
            {}
            """.format(
        data.to_string(index=False)
    )
    return get_openai_response(ai_setup, None, openai_token)


def get_deepdive_insights_from_ai(data):
    metric = data["metric"].min()
    ai_setup = (
        """
            You are an expert on RFM (Recency, Frequency, Monetary) analysis. 
            You will receive a dataframe from the user that contains segmentation by """
        + metric
        + """ score.
            The dataframe will include the following columns:
                score_label: The segment of the score.
                month: The month.
                total_customers_per_month: total customers in the segment
                customer_percentage: percentage of total customers per segment.
                

            Your task is to analyze the provided data, identify key patterns and insights and summarize the most important insights in a clear and concise manner, explaining why these insights are significant, and offer an actionable action.
            Use your expertise to deliver a thorough and actionable executive summary, ensuring that your observations and recommendations are clear, concise, and data-driven. 

            Write down the two most important insights you found in the data.
            When applicable, you can use macro segments to include a few segments in one insight. 
            Keep in mind that the segments are ordered alphabetically from A as the best to E as the worst.
            Check your answers for inconsistencies and contradicting information. Change the insight if you find it inconsistent or contradicting.
            The answer should be in the following format:
                Key insights:
                1. insight 1
                2. insight 2

            follow this format strictly

            ! remember to use the data for your recommendations
            ! remember to base you insights on the data, write only things that are derived from the data
            ! remember to write concisely and to the point, write the least amount of words you can in order to get your point across. 
            ! remember to follow the format provided for you answer

            Here is the dataframe for analysis:
            {}
            """.format(
            data.to_string(index=False)
        )
    )
    return get_openai_response(ai_setup, None, openai_token)


def get_last_day_of_month(day):
    next_month = day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)


def get_score_label(metric, score, b12, b23, b34, b45):
    """
    Creates labels for scores to show on visualized charts

    Args:
        metric (str): which metric you want to get the label for (r/f/m)
        score (int): The score (1-5)
        b12 (str): Boundary between 1 and 2 scores
        b23 (str): Boundary between 2 and 3 scores
        b34 (str): Boundary between 3 and 4 scores
        b45 (str): Boundary between 4 and 5 scores

    Returns:
        The label as string
    """
    suffix = " Days" if metric == "r" else "" if metric == "f" else " $"
    if score == 5:
        return f"A. > {b45:,}{suffix}" if metric == "m" else f"A. 0 - {b45:,}{suffix}"
    if score == 4:
        return (
            f"B. {b34:,}{suffix} - {b45:,}{suffix}"
            if metric == "m"
            else f"B. {b45:,} - {b34:,}{suffix}"
        )
    if score == 3:
        return (
            f"C. {b23:,}{suffix} - {b34:,}{suffix}"
            if metric == "m"
            else f"C. {b34:,} - {b23:,}{suffix}"
        )
    if score == 2:
        return (
            f"D. {b12:,}{suffix} - {b23:,}{suffix}"
            if metric == "m"
            else f"D. {b23:,} - {b12:,}{suffix}"
        )
    if score == 1:
        return (
            f"E. 0{suffix} - {b12:,}{suffix}"
            if metric == "m"
            else f"E. > {b12:,}{suffix}"
        )


def get_score(val, b12, b23, b34, b45, asc):
    """
    Calculates the score based on the value and the boundaries

    Args:
        val (float): The value to score
        b12 (float): Boundary between 1 and 2 scores
        b23 (float): Boundary between 2 and 3 scores
        b34 (float): Boundary between 3 and 4 scores
        b45 (float): Boundary between 4 and 5 scores
        asc (bool): Whether the values are ascending or descending (is lower number better than higher number)

    Returns:
        The score (1-5)
    """
    if asc:
        return (
            1
            if val <= b12
            else 2 if val <= b23 else 3 if val <= b34 else 4 if val <= b45 else 5
        )
    else:
        return (
            1
            if val >= b12
            else 2 if val >= b23 else 3 if val >= b34 else 4 if val >= b45 else 5
        )


def get_simple_segment(score):
    return (
        "01. High-Value"
        if score >= 4
        else "02. Medium-Value" if 4 > score >= 2 else "03. Low-Value"
    )


def calculate_monthly_movements(data, start, end, selected_mode):
    """
    Creates a dataframe that is grouped per Segment, month to represent totals and percentages
    of each of the metrics (r/f/m) per month

    Args:
        data (dataframe): The raw data of customer-date-value
        start (date): start date filter of the data
        end (date): end date filter of the data
        selected_mode (str): Detailed or Simple, to decide which segmentation to use

    Returns:
        Segmented dataframe by months
    """
    date_range = pd.date_range(
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), freq="MS"
    )
    months_list = date_range.strftime("%Y-%m").tolist()
    monthly_rfm_list = []
    segmented_monthly_rfm_list = []

    for month in months_list:
        rfm_monthly = data[data["month"] <= month].copy()
        max_split_monthly = (
            rfm_monthly[["id", "min_date", "date"]].groupby(["id"]).max()
        )
        max_split_monthly.rename(columns={"date": "last_date"}, inplace=True)
        rfm_monthly = (
            rfm_monthly[["id", "min_month", "num_of_events", "value"]]
            .groupby(["id", "min_month"])
            .sum()
            .reset_index()
        )
        rfm_monthly = pd.merge(rfm_monthly, max_split_monthly, how="left", on=["id"])
        rfm_monthly["last_date"] = rfm_monthly.apply(
            lambda row: datetime.datetime.strptime(row["last_date"], "%Y-%m-%d"), axis=1
        )
        rfm_monthly["month"] = month
        first_of_current_month = month + "-01"
        last_of_current_month = get_last_day_of_month(
            datetime.datetime.strptime(first_of_current_month, "%Y-%m-%d")
        )
        rfm_monthly["max_date_monthly"] = min(last_of_current_month, end_datetime)
        rfm_monthly["days"] = (
            (rfm_monthly["max_date_monthly"] - rfm_monthly["last_date"])
            / np.timedelta64(1, "D")
        ).astype(int)
        rfm_monthly["days_chosen_monthly"] = (
            (rfm_monthly["max_date_monthly"] - pd.to_datetime(rfm_monthly["min_date"]))
            / np.timedelta64(1, "D")
        ).astype(int)
        rfm_monthly["frequency"] = round(
            rfm_monthly["days_chosen_monthly"] / rfm_monthly["num_of_events"], 1
        )
        rfm_monthly["r_score"] = rfm_monthly.apply(
            lambda row: get_score(
                row["days"],
                settings["r12"],
                settings["r23"],
                settings["r34"],
                settings["r45"],
                False,
            ),
            axis=1,
        )
        rfm_monthly["r_score_label"] = rfm_monthly.apply(
            lambda row: get_score_label(
                "r",
                row["r_score"],
                settings["r12"],
                settings["r23"],
                settings["r34"],
                settings["r45"],
            ),
            axis=1,
        )
        rfm_monthly["f_score"] = rfm_monthly.apply(
            lambda row: get_score(
                row["frequency"],
                settings["f12"],
                settings["f23"],
                settings["f34"],
                settings["f45"],
                False,
            ),
            axis=1,
        )
        rfm_monthly["f_score_label"] = rfm_monthly.apply(
            lambda row: get_score_label(
                "f",
                row["f_score"],
                settings["f12"],
                settings["f23"],
                settings["f34"],
                settings["f45"],
            ),
            axis=1,
        )
        rfm_monthly["m_score"] = rfm_monthly.apply(
            lambda row: get_score(
                row["value"],
                settings["m12"],
                settings["m23"],
                settings["m34"],
                settings["m45"],
                True,
            ),
            axis=1,
        )
        rfm_monthly["m_score_label"] = rfm_monthly.apply(
            lambda row: get_score_label(
                "m",
                row["m_score"],
                settings["m12"],
                settings["m23"],
                settings["m34"],
                settings["m45"],
            ),
            axis=1,
        )
        if selected_mode == "Detailed":
            rfm_monthly["rfm_score"] = (
                rfm_monthly["r_score"].astype(str)
                + rfm_monthly["f_score"].astype(str)
                + rfm_monthly["m_score"].astype(str)
            )
            rfm_monthly["rfm_segment"] = rfm_monthly["r_score"].astype(
                str
            ) + rfm_monthly["f_score"].astype(str)
            rfm_monthly["rfm_segment"] = rfm_monthly["rfm_segment"].replace(
                seg_map, regex=True
            )
            rfm_monthly["rfm_order"] = rfm_monthly["rfm_segment"].str[:2]
            rfm_monthly["Segment"] = rfm_monthly["rfm_segment"].str[4:]
        if selected_mode == "Simple":
            rfm_monthly["rfm_score"] = (
                settings["rw"] * rfm_monthly["r_score"]
                + settings["fw"] * rfm_monthly["f_score"]
                + settings["rw"] * rfm_monthly["m_score"]
            )
            rfm_monthly["rfm_segment"] = rfm_monthly.apply(
                lambda row: get_simple_segment(row["rfm_score"]), axis=1
            )
            rfm_monthly["rfm_order"] = rfm_monthly["rfm_segment"].str[:2]
            rfm_monthly["Segment"] = rfm_monthly["rfm_segment"].str[4:]
        rfm_monthly["count"] = 1
        monthly_rfm_list.append(rfm_monthly)
        rfm_monthly = (
            rfm_monthly[
                ["Segment", "rfm_order", "month", "value", "num_of_events", "count"]
            ]
            .groupby(["Segment", "rfm_order", "month"])
            .sum()
            .reset_index()
        )
        rfm_monthly.rename(columns={"count": "total_customers"}, inplace=True)
        segmented_monthly_rfm_list.append(rfm_monthly)

    segmented_result = pd.concat(segmented_monthly_rfm_list)
    segmented_result["total_customers_per_month"] = segmented_result.groupby("month")[
        "total_customers"
    ].transform("sum")
    segmented_result["total_events_per_month"] = segmented_result.groupby("month")[
        "num_of_events"
    ].transform("sum")
    segmented_result["total_value_per_month"] = segmented_result.groupby("month")[
        "value"
    ].transform("sum")
    # Calculate percentage of customers in each segment per month
    segmented_result["customer_percentage_per_month"] = (
        segmented_result["total_customers"]
        / segmented_result["total_customers_per_month"]
    )
    segmented_result["events_percentage_per_month"] = (
        segmented_result["num_of_events"] / segmented_result["total_events_per_month"]
    )
    segmented_result["value_percentage_per_month"] = (
        segmented_result["value"] / segmented_result["total_value_per_month"]
    )
    segmented_result["customers_tooltip"] = segmented_result.apply(
        lambda row: f"{row['total_customers']:,} ({round(row['customer_percentage_per_month'] * 100, 1)}%)",
        axis=1,
    )
    segmented_result["transactions_tooltip"] = segmented_result.apply(
        lambda row: f"{row['num_of_events']:,} ({round(row['events_percentage_per_month'] * 100, 1)}%)",
        axis=1,
    )
    segmented_result["value_tooltip"] = segmented_result.apply(
        lambda row: f"{round(row['value']):,} ({round(row['value_percentage_per_month'] * 100, 1)}%)",
        axis=1,
    )
    segmented_result["Label"] = segmented_result.apply(
        lambda row: f"""<b style=\'font-size:14px;\'>{row['Segment']}</b><br>
Customers: {row['total_customers']:,} ({round(row['customer_percentage_per_month'] * 100, 1)}%)<br>
Monetary value: {round(row['value']):,} ({round(row['events_percentage_per_month'] * 100, 1)}%)<br>
Transactions: {round(row['num_of_events']):,} ({round(row['value_percentage_per_month'] * 100, 1)}%)""",
        axis=1,
    )

    detailed_result = pd.concat(monthly_rfm_list)
    return segmented_result, detailed_result


def recalculate(what):
    st.session_state.pop("settings", None)
    st.session_state.pop("filtered_data", None)
    st.session_state.pop("min_date", None)
    st.session_state.pop("max_date", None)
    if what != "simple":
        st.session_state.pop("detailed_segmented_rfm", None)
        st.session_state.pop("detailed_detailed_rfm", None)
        st.session_state.pop("detailed_overview_insights", None)
        st.session_state.pop("detailed_movements_insights", None)
        st.session_state.pop("detailed_deepdive_recency_insights")
        st.session_state.pop("detailed_deepdive_frequency_insights")
        st.session_state.pop("detailed_deepdive_monetary_insights")
    st.session_state.pop("simple_detailed_rfm", None)
    st.session_state.pop("simple_segmented_rfm", None)
    st.session_state.pop("simple_overview_insights", None)
    st.session_state.pop("simple_movements_insights", None)
    st.session_state.pop("simple_segments_tips", None)
    st.session_state.pop("detailed_segments_tips", None)
    st.session_state.pop("simple_deepdive_recency_insights")
    st.session_state.pop("simple_deepdive_frequency_insights")
    st.session_state.pop("simple_deepdive_monetary_insights")


def reload_data():
    st.session_state.pop("data", None)
    st.session_state.data_load_timestamp = datetime.datetime.now()
    recalculate("all")


def update_session_state(key, value=None, function=None):
    if key not in st.session_state:  # or not st.session_state[key]:
        if function:
            if value is None:
                st.session_state[key] = function()
            else:
                st.session_state[key] = function(value)
        else:
            st.session_state[key] = value
    return st.session_state[key]


def init_data():
    """
    Uploads the data from Keboola and prepares it so the app can process it

    Returns:
        Raw dataframe
    """
    data = load_data()
    st.session_state["data_load_timestamp"] = datetime.datetime.now()
    data.dropna(subset=["id"], inplace=True)
    last_date = datetime.datetime.strptime(data["date"].max(), "%Y-%m-%d")
    data["month"] = data.apply(lambda row: row["date"][:7], axis=1)
    max_split = data[["id", "date"]].groupby(["id"]).max()
    max_split.rename(columns={"date": "last_date"}, inplace=True)
    min_split = data[["id", "date", "month"]].groupby(["id"]).min()
    min_split.rename(columns={"date": "min_date", "month": "min_month"}, inplace=True)
    sum_split = data[["id", "num_of_events", "value"]].groupby(["id"]).sum()
    sum_split.rename(
        columns={"num_of_events": "total_num_of_events", "value": "total_value"},
        inplace=True,
    )
    data = pd.merge(data, sum_split, how="left", on=["id"])
    data = pd.merge(data, max_split, how="left", on=["id"])
    data = pd.merge(data, min_split, how="left", on=["id"])
    data["days"] = (
        (last_date - pd.to_datetime(data["last_date"])) / np.timedelta64(1, "D")
    ).astype(int)
    data["days_min_date"] = (
        (last_date - pd.to_datetime(data["min_date"])) / np.timedelta64(1, "D")
    ).astype(int)
    data["frequency"] = data.apply(
        lambda row: round(row["days_min_date"] / row["total_num_of_events"], 1), axis=1
    )
    return data


def filter_data(values):
    data = values["data"]
    start = values["start"]
    end = values["end"]
    return data[
        (data["date"] <= end.strftime("%Y-%m-%d"))
        & (data["date"] >= start.strftime("%Y-%m-%d"))
    ]


def get_settings():
    """
    Uploads the saved settings from Keboola

    Returns:
        A dictionary with the settings
    """
    if check_keboola_bucket(
        write_bucket, "rfm-app-settings", f"updated_settings.csv.gz"
    ):
        settings_load = get_dataframe(
            "in.c-" + write_bucket + ".rfm-app-settings"
        ).set_index("key")
        settings = settings_load.to_dict("dict")["value"]
    else:
        settings = {}
        (
            settings["r45"],
            settings["r34"],
            settings["r23"],
            settings["r12"],
            settings["r_max"],
        ) = (
            df["days"].quantile([0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        )
        (
            settings["f45"],
            settings["f34"],
            settings["f23"],
            settings["f12"],
            settings["f_max"],
        ) = (
            df["frequency"].quantile([0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        )
        (
            settings["m12"],
            settings["m23"],
            settings["m34"],
            settings["m45"],
            settings["m_max"],
        ) = (
            df["total_value"].quantile([0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        )
        settings["rw"] = settings["fw"] = settings["mw"] = 1 / 3
    return settings


seg_map = {
    r"5[4-5]": "01. Champions",
    r"[3-4][4-5]": "02. Loyal Customers",
    r"[4-5][2-3]": "03. Potential Loyalists",
    r"51": "04. Recent Customers",
    r"41": "05. Promising",
    r"33": "06. Need Attention",
    r"3[1-2]": "07. About to Sleep",
    r"[1-2][5]": "08. Can't Lose",
    r"[1-2][3-4]": "09. At Risk",
    r"2[1-2]": "10. Hibernating",
    r"1[1-2]": "11. Lost",
}

glossary = {
    "RFM": [
        "RFM (Recency, Frequency, Monetary) is a customer segmentation technique used in marketing to evaluate and categorize a customer's value based on their purchasing behavior."
    ],
    "Recency (R)": ["The time (in days) elapsed since a customer's last purchase."],
    "Frequency (F)": [
        "How often a customer makes purchases within a specific time period."
    ],
    "Monetary (M)": [
        "The total amount of money a customer has spent over a specific period."
    ],
    "Score": ["Ranking 1-5 (5 being the highest) of a customer in a specific metric."],
    "RFM score": [
        "A combination of scores from the R,F,M metrics. Usually presented in the format of xxx (e.g 544)."
    ],
    "ADSLT": ["Average Days Since Last Transaction"],
    "Average RFM score": ["An average of the R,F,M scores."],
    "Champions": ["Customers with RFM scores 54x, 55x."],
    "Loyal Customers": ["Customers with RFM score 34x, 35x, 44x, 45x."],
    "Potential Loyalists": ["Customers with RFM score 42x, 43x, 52x, 53x."],
    "Recent Customers": ["Customers with RFM score 51x."],
    "Promising": ["Customers with RFM score 41x."],
    "Need Attention": ["Customers with RFM score 33x."],
    "About to Sleep": ["Customers with RFM score 31x, 32x."],
    "Can't Lose": ["Customers with RFM score 15x, 25x."],
    "At Risk": ["Customers with RFM score 13x, 14x, 23x, 24x."],
    "Hibernating": ["Customers with RFM score 21x, 22x."],
    "Lost": ["Customers with RFM score 11x, 12x."],
    "High-Value": ["Customers with average RFM score higher than 4."],
    "Medium-Value": ["Customers with average RFM score between 2 and 4."],
    "Low-Value": ["Customers with average RFM score lower than 2."],
}
glossary_df = pd.DataFrame.from_dict(glossary, orient="index")
glossary_df.index.names = ["Term"]
glossary_df.columns = ["Description"]

st.image(LOGO_IMAGE_PATH)
hide_img_fs = """
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        """
st.markdown(hide_img_fs, unsafe_allow_html=True)

simple_segments = ["High-Value", "Medium-Value", "Low-Value"]
detailed_segments = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "Recent Customers",
    "Promising",
    "Need Attention",
    "About to Sleep",
    "Can't Lose",
    "At Risk",
    "Hibernating",
    "Lost",
]
simple_segments_tips = {}
detailed_segments_tips = {}
segment_tips = {}
today = datetime.datetime.now()

df = update_session_state("init_data", function=init_data)
min_date = update_session_state(
    "min_date", value=datetime.datetime.strptime(df["date"].min(), "%Y-%m-%d")
)
max_date = update_session_state(
    "max_date", value=datetime.datetime.strptime(df["date"].max(), "%Y-%m-%d")
)
settings = update_session_state("settings", function=get_settings)

with st.sidebar:
    settings_tooltip = """Adjust the settings you want and press the Save Settings button to apply the changes.  
                        The next time you load the app, the last settings you saved will be used.  
                        Within the selected dates, the data for customers is calculated from their first transactions until the end date.
                        """
    st.subheader("Settings", help=settings_tooltip)

    start_date_col, end_date_col = st.columns([1, 1])
    start_date = start_date_col.date_input(
        "Start date:", value=min_date, min_value=min_date, max_value=today
    )
    end_date = end_date_col.date_input(
        "End date:", value=max_date, min_value=min_date, max_value=today
    )
    end_datetime = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    start_datetime = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    mode = st.selectbox("RFM calculation", ["Detailed", "Simple"])
    with st.expander("Adjust RFM settings"):
        if mode == "Simple":
            weights_tooltip = """Choose the weights to use for calculating the average RFM score.  
                                The sum of the weights has to be 1.
                                """
            st.text("Adjust rfm weights:", help=weights_tooltip)
            r_weight, f_weight, m_weight = st.columns([1, 1, 1])
            r_weight.text("R")
            settings["rw"] = r_weight.number_input(
                "Recency weight",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=settings["rw"],
                label_visibility="collapsed",
            )
            f_weight.text("F")
            settings["fw"] = f_weight.number_input(
                "Frequency weight",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=settings["fw"],
                label_visibility="collapsed",
            )
            m_weight.text("M")
            settings["mw"] = m_weight.number_input(
                "Monetary weight",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=settings["mw"],
                label_visibility="collapsed",
            )

        st.markdown(
            f"""
            <style>
            .centered {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                text-align: center;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        rfm_scores_tooltip = """Change the values base on which the ranking of each metrics are calculated.  
                                - The R metric represents how many days have passed since the last transaction.  
                                - The F metric represents the frequency of transactions 
                                (if the value is 7 it means that the customers buy once every 7 days on average).  
                                - The M metric represents the total value from the customers in USD.  
                                If no settings were ever saved, the default values are quintiles based on the customer data.  
                                How to use:  
                                If you choose the R metric the value 3 between ranks 4 and 5,  
                                it will mean that all the customers that had a purchase in  
                                less than 3 days their R score will be 5.
                                """
        st.text("Adjust RFM scores:", help=rfm_scores_tooltip)
        r_col, f_col, m_col = st.columns([1, 1, 1])
        r_col.markdown('<div class ="centered"><b>R</b></div>', unsafe_allow_html=True)
        r_col.markdown("""---""")
        r_col.markdown('<div class ="centered">5</div>', unsafe_allow_html=True)
        settings["r45"] = r_col.number_input(
            "Very recent",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["r45"]),
            label_visibility="collapsed",
        )
        r_col.markdown('<div class ="centered">4</div>', unsafe_allow_html=True)
        settings["r34"] = r_col.number_input(
            "Somewhat recent",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["r34"]),
            label_visibility="collapsed",
        )
        r_col.markdown('<div class ="centered">3</div>', unsafe_allow_html=True)
        settings["r23"] = r_col.number_input(
            "Moderately recent",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["r23"]),
            label_visibility="collapsed",
        )
        r_col.markdown('<div class ="centered">2</div>', unsafe_allow_html=True)
        settings["r12"] = r_col.number_input(
            "Less recent",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["r12"]),
            label_visibility="collapsed",
        )
        r_col.markdown('<div class ="centered">1</div>', unsafe_allow_html=True)

        f_col.markdown('<div class ="centered"><b>F</b></div>', unsafe_allow_html=True)
        f_col.markdown("""---""")
        f_col.markdown('<div class ="centered">5</div>', unsafe_allow_html=True)
        settings["f45"] = f_col.number_input(
            "Very frequent",
            format="%f",
            min_value=0.0,
            step=0.1,
            value=round(settings["f45"], 1),
            label_visibility="collapsed",
        )
        f_col.markdown('<div class ="centered">4</div>', unsafe_allow_html=True)
        settings["f34"] = f_col.number_input(
            "Somewhat frequent",
            format="%f",
            min_value=0.0,
            step=0.1,
            value=round(settings["f34"], 1),
            label_visibility="collapsed",
        )
        f_col.markdown('<div class ="centered">3</div>', unsafe_allow_html=True)
        settings["f23"] = f_col.number_input(
            "Moderately frequent",
            format="%f",
            min_value=0.0,
            step=0.1,
            value=round(settings["f23"], 1),
            label_visibility="collapsed",
        )
        f_col.markdown('<div class ="centered">2</div>', unsafe_allow_html=True)
        settings["f12"] = f_col.number_input(
            "Less frequent",
            format="%f",
            min_value=0.0,
            step=0.1,
            value=round(settings["f12"], 1),
            label_visibility="collapsed",
        )
        f_col.markdown('<div class ="centered">1</div>', unsafe_allow_html=True)
        max_value = df["value"].max()
        m_col.markdown('<div class ="centered"><b>M</b></div>', unsafe_allow_html=True)
        m_col.markdown("""---""")
        m_col.markdown('<div class ="centered">5</div>', unsafe_allow_html=True)
        settings["m45"] = m_col.number_input(
            "Big spenders",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["m45"]),
            label_visibility="collapsed",
        )
        m_col.markdown('<div class ="centered">4</div>', unsafe_allow_html=True)
        settings["m34"] = m_col.number_input(
            "Spenders",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["m34"]),
            label_visibility="collapsed",
        )
        m_col.markdown('<div class ="centered">3</div>', unsafe_allow_html=True)
        settings["m23"] = m_col.number_input(
            "Moderate",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["m23"]),
            label_visibility="collapsed",
        )
        m_col.markdown('<div class ="centered">2</div>', unsafe_allow_html=True)
        settings["m12"] = m_col.number_input(
            "Savers",
            format="%i",
            min_value=0,
            step=1,
            value=int(settings["m12"]),
            label_visibility="collapsed",
        )
        m_col.markdown('<div class ="centered">1</div>', unsafe_allow_html=True)

    if st.button("Save Settings"):
        settings_df = pd.DataFrame(settings.items(), columns=["key", "value"])
        if settings["rw"] + settings["fw"] + settings["mw"] != 1:
            if (
                settings["rw"] >= 0.33
                and settings["fw"] >= 0.33
                and settings["mw"] >= 0.33
            ):
                settings["rw"] = settings["fw"] = settings["mw"] = 1 / 3
                write_to_keboola(
                    settings_df, "rfm-app-settings", f"updated_settings.csv.gz", False
                )
                recalculate("simple")
            else:
                st.error("The sum of the weights does not equal 1.", icon="🚨")
        else:
            write_to_keboola(
                settings_df, "rfm-app-settings", f"updated_settings.csv.gz", False
            )
            recalculate("all")
    ChangeButtonColour("Save Settings", "#FFFFFF", "#1EC71E", "#1EC71E")

    with st.expander("Reload Data"):
        st.info(
            "Last data load: "
            + str(st.session_state.data_load_timestamp.strftime("%Y-%m-%d %X")),
            icon="ℹ️",
        )
        if st.button("Reload Data"):
            reload_data()
        ChangeButtonColour("Reload Data", "#FFFFFF", "#1EC71E", "#1EC71E")

df = update_session_state(
    "filtered_data",
    value={"data": df, "start": start_date, "end": end_date},
    function=filter_data,
)
if mode == "Simple":
    if "simple_segmented_rfm" not in st.session_state:
        segmented_rfm, detailed_rfm = calculate_monthly_movements(
            df, start_date, end_date, mode
        )
        st.session_state.simple_segmented_rfm = segmented_rfm
        st.session_state.simple_detailed_rfm = detailed_rfm
    segmented_rfm = st.session_state.simple_segmented_rfm
    detailed_rfm = st.session_state.simple_detailed_rfm
else:
    if "detailed_segmented_rfm" not in st.session_state:
        segmented_rfm, detailed_rfm = calculate_monthly_movements(
            df, start_date, end_date, mode
        )
        st.session_state.detailed_segmented_rfm = segmented_rfm
        st.session_state.detailed_detailed_rfm = detailed_rfm
    segmented_rfm = st.session_state.detailed_segmented_rfm
    detailed_rfm = st.session_state.detailed_detailed_rfm

overview_tab, movements_tab, deepdive_tab, recommendations_tab, glossary_tab = st.tabs(
    ["Overview", "Movements", "R/F/M Deepdive", "Recommendations", "Glossary"]
)

with overview_tab:
    overview_var = st.selectbox(
        "Show segment size by:", ["Customers", "Value", "Transactions"]
    )
    treemap_values = "customer_percentage_per_month"
    if overview_var == "Value":
        treemap_values = "value_percentage_per_month"
    elif overview_var == "Transactions":
        treemap_values = "events_percentage_per_month"

    current_rfm_segmentation = segmented_rfm[
        segmented_rfm["month"] == end_date.strftime("%Y-%m")
    ].copy()

    # Plot the treemap
    fig = px.treemap(
        current_rfm_segmentation,
        path=["Label"],
        values=treemap_values,
        color="Segment",
        hover_data={"Label": True},
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=20),
    )
    fig.update_traces(
        hovertemplate="%{label}<extra></extra>",
        customdata=current_rfm_segmentation[["Label"]],
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)
    overview_info = st.info("Loading insights...", icon="ℹ️")
    overview_insights_key = (
        "detailed_overview_insights"
        if mode == "Detailed"
        else "simple_overview_insights"
    )
    overview_info.info(
        update_session_state(
            overview_insights_key,
            current_rfm_segmentation[
                [
                    "Segment",
                    "total_customers",
                    "num_of_events",
                    "value",
                    "customer_percentage_per_month",
                    "value_percentage_per_month",
                    "events_percentage_per_month",
                ]
            ],
            get_overview_insights_from_ai,
        ),
        icon="ℹ️",
    )

with movements_tab:
    movements_choice, movements_chart_choice = st.columns([1, 1])
    movements_var = movements_choice.selectbox(
        "Show segment movements by:", ["Customers", "Value", "Transactions"]
    )
    movements_chart_select = movements_chart_choice.selectbox(
        "Metric to show:", ["Totals", "Percentage"]
    )

    total_y = (
        "total_customers:Q"
        if movements_var == "Customers"
        else "num_of_events:Q" if movements_var == "Transactions" else "value:Q"
    )
    total_y_tooltip = (
        "customers_tooltip:N"
        if movements_var == "Customers"
        else (
            "transactions_tooltip:N"
            if movements_var == "Transactions"
            else "value_tooltip:N"
        )
    )
    total_y_text = (
        "customer_percentage_per_month:N"
        if movements_var == "Customers"
        else (
            "transactions_percentage_per_month:N"
            if movements_var == "Transactions"
            else "value_percentage_per_month:N"
        )
    )
    # Creating a bar chart using Altair
    movements_chart = (
        alt.Chart(segmented_rfm)
        .mark_bar()
        .encode(
            x=alt.X(
                "month:N",
                sort=alt.EncodingSortField(field="month", order="ascending"),
                title="Month",
            ),
            y=alt.X(total_y, title="Customers"),
            color=alt.Color("Segment:N"),
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("Segment:N", title="Segment"),
                alt.Tooltip(total_y_tooltip, title=movements_var),
            ],
            text=alt.Text(total_y_text),
        )
        .properties(width="container")
    )

    percentage_y = (
        "customer_percentage_per_month:Q"
        if movements_var == "Customers"
        else (
            "events_percentage_per_month:Q"
            if movements_var == "Transactions"
            else "value_percentage_per_month:Q"
        )
    )
    # Create the Altair chart for percentage movements per segment
    movements_percentage_chart = (
        alt.Chart(segmented_rfm)
        .mark_bar()
        .encode(
            x=alt.X(
                "month:N",
                sort=alt.EncodingSortField(field="month", order="ascending"),
                title="Month",
            ),
            y=alt.Y(
                percentage_y,
                title="Percentage of " + movements_var,
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("Segment:N", title="Segment"),
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("Segment:N", title="Segment"),
                alt.Tooltip(percentage_y, title="Percentage", format=".2%"),
            ],
        )
        .properties(width="container")
    )

    # Display the chart in Streamlit

    if movements_chart_select == "Totals":
        st.altair_chart(movements_chart, use_container_width=True)
    else:
        st.altair_chart(movements_percentage_chart, use_container_width=True)

    # st.altair_chart(combined_chart, use_container_width=True)
    movements_info = st.info("Loading insights...", icon="ℹ️")
    movements_insights_key = (
        "detailed_movements_insights"
        if mode == "Detailed"
        else "simple_movements_insights"
    )
    movements_info.info(
        update_session_state(
            movements_insights_key,
            segmented_rfm[
                [
                    "Segment",
                    "month",
                    "total_customers",
                    "num_of_events",
                    "value",
                    "customer_percentage_per_month",
                    "value_percentage_per_month",
                    "events_percentage_per_month",
                ]
            ],
            get_movements_insights_from_ai,
        ),
        icon="ℹ️",
    )

with deepdive_tab:
    deepdive_var = st.selectbox("Metric", ["Recency", "Frequency", "Monetary"])
    bar_y = (
        "days:Q"
        if deepdive_var == "Recency"
        else (
            "percentage_transactions:Q"
            if deepdive_var == "Frequency"
            else "percentage_value:Q"
        )
    )
    bar_y_title = (
        "Average days since last transaction"
        if deepdive_var == "Recency"
        else (
            "% of total transactions"
            if deepdive_var == "Frequency"
            else "% of total value"
        )
    )
    bar_y_format = ".3" if deepdive_var == "Recency" else ".2%"
    bar_y_axis_format = ".3" if deepdive_var == "Recency" else "%"

    current_detailed_rfm = detailed_rfm[
        detailed_rfm["month"] == end_date.strftime("%Y-%m")
    ].copy()

    box_deepdive_col, bar_deepdive_col = st.columns([1, 4])
    current_detailed_rfm["score_label"] = (
        current_detailed_rfm["r_score_label"]
        if deepdive_var == "Recency"
        else (
            current_detailed_rfm["f_score_label"]
            if deepdive_var == "Frequency"
            else current_detailed_rfm["m_score_label"]
        )
    )
    deepdive_rfm = (
        current_detailed_rfm["score_label"]
        .value_counts()
        .reset_index(name="counts")
        .sort_index()
    )
    deepdive_avg = (
        current_detailed_rfm[["score_label", "days"]].groupby(["score_label"]).mean()
    )
    deepdive_sums = (
        current_detailed_rfm[["score_label", "value", "num_of_events"]]
        .groupby(["score_label"])
        .sum()
    )
    deepdive_rfm = pd.merge(deepdive_rfm, deepdive_avg, how="left", on=["score_label"])
    deepdive_rfm = pd.merge(deepdive_rfm, deepdive_sums, how="left", on=["score_label"])
    deepdive_rfm["percentage_users"] = (
        deepdive_rfm["counts"] / deepdive_rfm["counts"].sum()
    )
    deepdive_rfm["percentage_value"] = (
        deepdive_rfm["value"] / deepdive_rfm["value"].sum()
    )
    deepdive_rfm["percentage_transactions"] = (
        deepdive_rfm["num_of_events"] / deepdive_rfm["num_of_events"].sum()
    )
    barchart_title = (
        "% of total customers per segment and AVG Days Since Last Transaction (ADSLT)"
        if deepdive_var == "Recency"
        else (
            "% of total customers per segment contributing to % of transactions"
            if deepdive_var == "Frequency"
            else "% of total customers per segment contributing to % of monetary value"
        )
    )

    # Create bar chart
    bar = (
        alt.Chart(deepdive_rfm)
        .mark_bar()
        .encode(
            x=alt.X("score_label:O"),
            y=alt.Y(bar_y, title=bar_y_title, axis=alt.Axis(format=bar_y_axis_format)),
            tooltip=[
                alt.Tooltip("score_label:O", title="Score"),
                alt.Tooltip(bar_y, title=bar_y_title, format=bar_y_format),
                alt.Tooltip("percentage_users:Q", title="% of customers", format=".2%"),
            ],
        )
    )

    text = bar.mark_text(
        align="center",
        baseline="middle",
        dy=-10,  # Move text on the bars
        color="black",
    ).encode(
        x=alt.X("score_label:O", axis=alt.Axis(title=None, labels=True, ticks=False)),
        y=alt.Y(bar_y, axis=alt.Axis(title=None, labels=False, ticks=False)),
        text=alt.Text(bar_y, format=bar_y_format),
    )

    # Create line chart
    line = (
        alt.Chart(deepdive_rfm)
        .mark_line(color="red")
        .encode(
            x=alt.X("score_label:O"),
            y=alt.Y(
                "percentage_users:Q", title="% of customers", axis=alt.Axis(format="%")
            ),
            tooltip=[
                alt.Tooltip("score_label:O", title=deepdive_var),
                alt.Tooltip(bar_y, title=bar_y_title, format=bar_y_format),
                alt.Tooltip("percentage_users:Q", title="% of customers", format=".2%"),
            ],
        )
    )

    # Add text labels to line chart
    line_text = line.mark_text(
        align="center",
        baseline="bottom",
        dy=-10,  # Move text above the points
        color="black",
    ).encode(
        x=alt.X("score_label:O", axis=alt.Axis(title=None, labels=False, ticks=False)),
        y=alt.Y(
            "percentage_users:Q", axis=alt.Axis(title=None, labels=False, ticks=False)
        ),
        text=alt.Text("percentage_users:Q", format=".2%"),
    )

    # Combine bar and line charts
    combined_chart = (
        alt.layer(bar, line, text, line_text)
        .resolve_scale(y="independent")
        .properties(title=barchart_title, width="container")
        .configure_title(fontSize=12)
    )

    bar_deepdive_col.altair_chart(combined_chart, use_container_width=True)

    # Box plot of average days since last transaction per user
    boxplot_var = (
        "days"
        if deepdive_var == "Recency"
        else "num_of_events" if deepdive_var == "Frequency" else "value"
    )
    boxplot_y = (
        "days:Q"
        if deepdive_var == "Recency"
        else "num_of_events:Q" if deepdive_var == "Frequency" else "value:Q"
    )
    boxplot_y_title = (
        "Average Days Since Last Transaction"
        if deepdive_var == "Recency"
        else "Transactions per user" if deepdive_var == "Frequency" else "Total value"
    )
    boxplot_title = (
        "ADSLT per customer"
        if deepdive_var == "Recency"
        else (
            "Transactions per customer"
            if deepdive_var == "Frequency"
            else "Total value per customer"
        )
    )

    deepdive_boxplot = (
        alt.Chart(current_detailed_rfm[["id", boxplot_var]])
        .mark_boxplot(size=100)
        .encode(
            y=alt.Y(boxplot_y, title=boxplot_y_title),
            tooltip=[alt.Tooltip(boxplot_y, title=boxplot_y_title)],
        )
        .properties(title=boxplot_title, width="container")
        .configure_title(fontSize=12)
    )

    box_deepdive_col.altair_chart(deepdive_boxplot, use_container_width=True)
    detailed_rfm["score_label"] = (
        detailed_rfm["r_score_label"]
        if deepdive_var == "Recency"
        else (
            detailed_rfm["f_score_label"]
            if deepdive_var == "Frequency"
            else detailed_rfm["m_score_label"]
        )
    )
    deepdive_movements_rfm = (
        detailed_rfm[["score_label", "month", "count"]]
        .groupby(["score_label", "month"])
        .sum()
        .reset_index()
    )
    deepdive_movements_rfm["metric"] = deepdive_var
    deepdive_movements_rfm["total_customers_per_month"] = (
        deepdive_movements_rfm.groupby("month")["count"].transform("sum")
    )
    deepdive_movements_rfm["customer_percentage"] = (
        deepdive_movements_rfm["count"]
        / deepdive_movements_rfm["total_customers_per_month"]
    )
    deepdive_movements_chart = (
        alt.Chart(deepdive_movements_rfm)
        .mark_bar()
        .encode(
            x=alt.X(
                "month:N",
                sort=alt.EncodingSortField(field="month", order="ascending"),
                title="Month",
            ),
            y=alt.Y(
                "customer_percentage:Q",
                title="Percentage of Customers",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("score_label:N", title="Score"),
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("score_label:N", title="Score"),
                alt.Tooltip("customer_percentage:Q", title="Percentage", format=".2%"),
            ],
        )
        .properties(
            title="Monthly movements in percentage per segment", width="container"
        )
        .configure_title(fontSize=12)
    )

    st.altair_chart(deepdive_movements_chart, use_container_width=True)

    deepdive_info = st.info("Loading insights...", icon="ℹ️")
    deepdive_insights_key = (
        "detailed_deepdive_insights"
        if mode == "Detailed"
        else "simple_deepdive_insights"
    )
    if mode == "Detailed":
        deepdive_insights_key = (
            "detailed_deepdive_recency_insights"
            if deepdive_var == "Recency"
            else (
                "detailed_deepdive_frequency_insights"
                if deepdive_var == "Frequency"
                else "detailed_deepdive_monetary_insights"
            )
        )
    else:
        deepdive_insights_key = (
            "simple_deepdive_recency_insights"
            if deepdive_var == "Recency"
            else (
                "simple_deepdive_frequency_insights"
                if deepdive_var == "Frequency"
                else "simple_deepdive_monetary_insights"
            )
        )
    deepdive_info.info(
        update_session_state(
            deepdive_insights_key, deepdive_movements_rfm, get_deepdive_insights_from_ai
        ),
        icon="ℹ️",
    )

with recommendations_tab:
    recommendation_segments = ["All"]
    if mode == "Simple":
        segment_tips = update_session_state(
            "simple_segments_tips", simple_segments, get_segments_info_from_ai
        )
        recommendation_segments = recommendation_segments + simple_segments
    else:
        segment_tips = update_session_state(
            "detailed_segments_tips", detailed_segments, get_segments_info_from_ai
        )
        recommendation_segments = recommendation_segments + detailed_segments
    recommendations_segment_filter = st.selectbox("Segment", recommendation_segments)
    if recommendations_segment_filter == "All":
        rfm_recommendations = (
            detailed_rfm[detailed_rfm["month"] == end_date.strftime("%Y-%m")][
                ["id", "Segment", "rfm_score", "days", "num_of_events", "value"]
            ]
            .set_index("id")
            .copy()
        )
    else:
        rfm_recommendations = (
            detailed_rfm[detailed_rfm["month"] == end_date.strftime("%Y-%m")][
                ["id", "Segment", "rfm_score", "days", "num_of_events", "value"]
            ]
            .set_index("id")
            .copy()
        )
        rfm_recommendations = rfm_recommendations[
            rfm_recommendations["Segment"] == recommendations_segment_filter
        ]
    st.dataframe(
        rfm_recommendations.rename(
            columns={
                "id": "ID",
                "rfm_score": "RFM Score",
                "days": "Days",
                "num_of_events": "Transactions",
                "value": "Value",
            }
        ),
        use_container_width=True,
    )
    recommendations_info = st.info(
        "Choose a segment to see it's definitions and actionable tips.", icon="ℹ️"
    )
    if recommendations_segment_filter != "All":
        recommendations_text = (
            recommendations_segment_filter
            + ":\nSegment definition:\n"
            + segment_tips[recommendations_segment_filter]["explanation"]
            + "\nActionable tips:\n"
            + segment_tips[recommendations_segment_filter]["action_points"]
        )
        recommendations_info.info(recommendations_text, icon="ℹ️")

with glossary_tab:
    search_term = st.text_input("Search for a term:")

    if search_term:
        # Filter dataframe based on search term
        filtered_glossary = glossary_df[
            glossary_df.apply(
                lambda row: row.astype(str).str.contains(search_term, case=False).any()
                or search_term.lower() in row.name.lower(),
                axis=1,
            )
        ]
    else:
        filtered_glossary = glossary_df

    st.dataframe(filtered_glossary, use_container_width=True)

display_footer_section()
